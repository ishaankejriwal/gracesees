from __future__ import annotations

import sys
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import pandas as pd

from grace_gnn.config import (
    DATA_RAW,
    LAGS,
    RANDOM_SEED,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VAL_FRACTION,
)
from grace_gnn.correlation import region_correlation_matrix, region_correlation_pairs
from grace_gnn.data import (
    aggregate_grace_netcdf_to_mask_zips,
    find_first_file,
    list_mask_members,
    read_basin_month_csv,
)
from grace_gnn.evaluate import graph_prediction_frame, prediction_frame
from grace_gnn.experiment import ExperimentPaths, MaskExperiment
from grace_gnn.features import feature_columns, make_lagged_dataset
from grace_gnn.graph import (
    build_knn_edges_from_mask_zips,
    make_degree_matched_random_edges,
    reverse_edges,
    save_edges,
    symmetrize_edges,
)
from grace_gnn.metrics import (
    improvement_by_basin,
    metrics_by_basin,
    metrics_overall,
    prediction_diagnostics,
)
from grace_gnn.models import (
    predict_persistence,
    torch_available,
    train_correlation_neighbor,
    train_manual_gcn,
    train_mlp,
    train_random_forest,
    train_recurrent_lag_model,
    train_ridge,
    train_xgboost,
)
from grace_gnn.splits import chronological_fraction_split
from grace_gnn.validation import (
    file_fingerprint,
    read_json,
    require_matching_provenance,
    validate_basin_month,
    validate_edges,
    validate_lagged_dataset,
    validate_unique_mask_members,
    write_json,
)


def _resolve_mask_zip(mask_zip: str | Path) -> Path:
    path = Path(mask_zip)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists() and not str(mask_zip).startswith("masks"):
        candidate = ROOT / "masks" / Path(mask_zip).name
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(f"Missing mask zip: {path}")
    return path


def _selected_mask_members(experiment: MaskExperiment) -> pd.DataFrame:
    members = list_mask_members(
        [experiment.mask_zip],
        experiment.basin_name_filter,
        strict=experiment.strict_mask_names,
    )
    if experiment.basin_name_exclude and not members.empty:
        members = members[
            ~members["basin_name"].str.contains(experiment.basin_name_exclude, case=False, na=False)
        ].copy()
    validate_unique_mask_members(members)
    return members


def _basin_month_provenance(grace_nc: Path, experiment: MaskExperiment, members: pd.DataFrame) -> dict:
    provenance = {
        "experiment_region": experiment.paths.name,
        "grace_netcdf": file_fingerprint(grace_nc),
        "mask_zips": [file_fingerprint(experiment.mask_zip)],
        "mask_format": (
            "HydroBASINS .mask.csv/.mask.xyz members"
            if experiment.strict_mask_names
            else ".mask.csv/.mask.xyz members"
        ),
        "basin_name_exclude": experiment.basin_name_exclude,
        "basin_count": int(members["basin_id"].nunique()),
        "basin_ids": sorted(members["basin_id"].astype(str).unique()),
    }
    if experiment.basin_name_filter is not None or not experiment.strict_mask_names:
        provenance["basin_name_filter"] = experiment.basin_name_filter
        provenance["strict_mask_names"] = experiment.strict_mask_names
    return provenance


def _lagged_provenance(basin_month_provenance: dict) -> dict:
    return {
        "experiment_region": basin_month_provenance.get("experiment_region"),
        "source_basin_month": basin_month_provenance,
        "lags": LAGS,
    }


def summarize_basin_month_coverage(basin_month: pd.DataFrame) -> pd.DataFrame:
    data = basin_month.copy()
    data["basin_id"] = data["basin_id"].astype(str)
    data["date"] = pd.to_datetime(data["date"])
    summary = (
        data.groupby(["basin_id", "basin_name"], dropna=False)
        .agg(
            months=("date", "nunique"),
            valid_months=("twsa_cm", lambda values: int(values.notna().sum())),
            first_date=("date", "min"),
            last_date=("date", "max"),
            min_twsa_cm=("twsa_cm", "min"),
            max_twsa_cm=("twsa_cm", "max"),
        )
        .reset_index()
    )
    summary["missing_months"] = summary["months"] - summary["valid_months"]
    return summary.sort_values(["valid_months", "basin_name"]).reset_index(drop=True)


def print_basin_month_coverage(basin_month: pd.DataFrame, paths: ExperimentPaths) -> None:
    summary = summarize_basin_month_coverage(basin_month)
    output_path = paths.output_dir / "basin_month_coverage.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Saved basin-month coverage diagnostics to {output_path}")
    print("Lowest valid-month mask coverage:")
    print(summary.head(10).to_string(index=False))
    if summary["valid_months"].eq(0).all():
        raise ValueError(
            "All selected masks produced only NaN GRACE TWSA values. "
            "Check that custom mask rows are geographic lon,lat,weight values in degrees, "
            "not mode/signal IDs, raster row/column indices, projected x/y meters, or normalized coordinates."
        )


def build_basin_month(experiment: MaskExperiment, force: bool = True) -> pd.DataFrame:
    experiment.paths.ensure_dirs()
    grace_nc = find_first_file(DATA_RAW, [".nc", ".nc4"])
    if grace_nc is None:
        raise FileNotFoundError(f"No GRACE NetCDF found in {DATA_RAW}")

    members = _selected_mask_members(experiment)
    expected_basin_ids = set(members["basin_id"].astype(str))
    provenance = _basin_month_provenance(grace_nc, experiment, members)
    if experiment.paths.basin_month_csv.exists() and not force:
        if not require_matching_provenance(experiment.paths.basin_month_provenance_json, provenance):
            raise ValueError(
                f"Refusing to reuse {experiment.paths.basin_month_csv} because provenance is missing or stale. "
                "Run build_basin_month(force=True) or delete the stale processed file."
            )
        basin_month = read_basin_month_csv(experiment.paths.basin_month_csv)
        validate_basin_month(basin_month, expected_basin_ids=expected_basin_ids)
        print_basin_month_coverage(basin_month, experiment.paths)
        return basin_month

    print(f"Aggregating masks for {experiment.paths.name} from {experiment.mask_zip.name}")
    basin_month = aggregate_grace_netcdf_to_mask_zips(
        grace_nc,
        [experiment.mask_zip],
        experiment.paths.basin_month_csv,
        basin_name_filter=experiment.basin_name_filter,
        basin_name_exclude=experiment.basin_name_exclude,
        strict_mask_names=experiment.strict_mask_names,
    )
    validate_basin_month(basin_month, expected_basin_ids=expected_basin_ids)
    basin_month.to_csv(experiment.paths.basin_month_csv, index=False)
    write_json(experiment.paths.basin_month_provenance_json, provenance)
    print(f"Saved {len(basin_month):,} basin-month rows to {experiment.paths.basin_month_csv}")
    print_basin_month_coverage(basin_month, experiment.paths)
    return basin_month


def build_lagged(basin_month: pd.DataFrame, paths: ExperimentPaths) -> pd.DataFrame:
    validate_basin_month(basin_month)
    lagged = make_lagged_dataset(basin_month, LAGS, paths.lagged_dataset_csv)
    validate_lagged_dataset(lagged)
    source_basin_ids = set(basin_month["basin_id"].astype(str).unique())
    lagged_basin_ids = set(lagged["basin_id"].astype(str).unique())
    dropped_basin_ids = sorted(source_basin_ids - lagged_basin_ids)
    if dropped_basin_ids:
        print(
            "Warning: dropped "
            f"{len(dropped_basin_ids)} mask(s) with no complete lag/target rows. "
            f"Examples: {dropped_basin_ids[:5]}"
        )
    unique_dates = pd.to_datetime(lagged["date"]).nunique()
    if unique_dates < 3:
        raise ValueError(
            "Lagged dataset has fewer than three unique dates after dropping incomplete lag rows. "
            "Check that the custom masks overlap GRACE cells with valid data across enough months."
        )
    basin_month_provenance = read_json(paths.basin_month_provenance_json) or {"source": str(paths.basin_month_csv)}
    write_json(paths.lagged_dataset_provenance_json, _lagged_provenance(basin_month_provenance))
    print(
        f"Saved {len(lagged):,} lagged samples across "
        f"{lagged['basin_id'].nunique()} masks to {paths.lagged_dataset_csv}"
    )
    return lagged


def train_baselines(lagged: pd.DataFrame, paths: ExperimentPaths) -> pd.DataFrame:
    splits = chronological_fraction_split(lagged, TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION)
    total = sum(len(frame) for frame in splits.values())
    for name, frame in splits.items():
        pct = len(frame) / total if total else 0
        print(name, len(frame), f"{pct:.1%}", frame["date"].min(), frame["date"].max())

    features = feature_columns(lagged)
    prediction_parts = []
    for split_name, frame in splits.items():
        prediction_parts.append(prediction_frame(frame, predict_persistence(frame), "persistence", "none", split_name))

    models = [
        ("ridge_ar", lambda: train_ridge(splits["train"], splits["val"], splits["test"], features)),
        (
            "random_forest_ar",
            lambda: train_random_forest(splits["train"], splits["val"], splits["test"], features, seed=RANDOM_SEED),
        ),
        ("xgboost_ar", lambda: train_xgboost(splits["train"], splits["val"], splits["test"], features, seed=RANDOM_SEED)),
        ("correlation_neighbor", lambda: train_correlation_neighbor(splits["train"], splits["val"], splits["test"])),
    ]
    for model_name, trainer in models:
        try:
            _, preds = trainer()
        except Exception as exc:
            if model_name != "xgboost_ar":
                raise
            message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            print(f"Skipping xgboost_ar because XGBoost is unavailable: {message}")
            continue
        graph_type = "train_positive_corr_lag1" if model_name == "correlation_neighbor" else "none"
        for split_name, frame in splits.items():
            prediction_parts.append(prediction_frame(frame, preds[split_name], model_name, graph_type, split_name))

    corr = region_correlation_matrix(splits["train"])
    paths.correlation_matrix_csv.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(paths.correlation_matrix_csv)
    region_correlation_pairs(corr).to_csv(paths.correlation_pairs_csv, index=False)

    if torch_available():
        _, mlp_preds = train_mlp(splits["train"], splits["val"], splits["test"], features, seed=RANDOM_SEED)
        for split_name, frame in splits.items():
            prediction_parts.append(prediction_frame(frame, mlp_preds[split_name], "basin_only_nn", "none", split_name))
        recurrent_models = [
            ("gru_lag_nn", "gru"),
            ("rnn_lag_nn", "rnn"),
        ]
        for model_name, cell_type in recurrent_models:
            _, recurrent_preds = train_recurrent_lag_model(
                splits["train"],
                splits["val"],
                splits["test"],
                features,
                seed=RANDOM_SEED,
                cell_type=cell_type,
            )
            for split_name, frame in splits.items():
                prediction_parts.append(
                    prediction_frame(frame, recurrent_preds[split_name], model_name, "none", split_name)
                )
    else:
        print("PyTorch unavailable; skipping basin-only NN and GNN models.")

    predictions = pd.concat(prediction_parts, ignore_index=True)
    paths.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(paths.predictions_csv, index=False)
    return predictions


def train_gnns(lagged: pd.DataFrame, predictions: pd.DataFrame, experiment: MaskExperiment) -> pd.DataFrame:
    if not torch_available():
        return predictions

    basin_ids = sorted(lagged["basin_id"].astype(str).unique())
    basin_names = sorted(lagged["basin_name"].dropna().unique())
    real_directed = build_knn_edges_from_mask_zips(
        [experiment.mask_zip],
        basin_names,
        experiment.paths.output_dir / "edges_real_knn_directed.csv",
        k=3,
        graph_type="real_knn_directed",
        strict_mask_names=experiment.strict_mask_names,
    )
    graph_variants = {
        "real_knn_directed": real_directed,
        "real_knn_undirected": symmetrize_edges(real_directed, graph_type="real_knn_undirected"),
        "real_knn_reversed": reverse_edges(real_directed, graph_type="real_knn_reversed"),
        "random_degree_matched": make_degree_matched_random_edges(real_directed, basin_ids, seed=RANDOM_SEED),
    }
    for graph_type, edges in graph_variants.items():
        validate_edges(edges, set(basin_ids), graph_type=graph_type)
        save_edges(edges, experiment.paths.output_dir / f"edges_{graph_type}.csv")
        print(f"Saved {len(edges):,} {graph_type} edges")

    splits = chronological_fraction_split(lagged, TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION)
    for split_name, frame in splits.items():
        validate_lagged_dataset(frame)
    features = feature_columns(lagged)
    gnn_parts = []
    for graph_type, edges in graph_variants.items():
        _, preds = train_manual_gcn(
            splits["train"],
            splits["val"],
            splits["test"],
            features,
            edges,
            basin_ids,
            seed=RANDOM_SEED,
            epochs=120,
            residual=True,
        )
        for split_name, pred_df in preds.items():
            gnn_parts.append(graph_prediction_frame(pred_df, "residual_neighbor_gnn", graph_type, split_name))

    gnn_predictions = pd.concat(gnn_parts, ignore_index=True)
    combined = pd.concat([predictions, gnn_predictions], ignore_index=True)
    combined = combined.drop_duplicates(["date", "basin_id", "model_name", "graph_type", "split"], keep="last")
    combined.to_csv(experiment.paths.predictions_csv, index=False)
    return combined


def save_metrics(predictions: pd.DataFrame, paths: ExperimentPaths) -> None:
    overall = metrics_overall(predictions).sort_values(["split", "rmse_cm"])
    by_basin = metrics_by_basin(predictions, split="test").sort_values(["basin_name", "rmse_cm"])
    improvement = improvement_by_basin(by_basin)
    diagnostics = prediction_diagnostics(predictions)

    overall.to_csv(paths.metrics_overall_csv, index=False)
    by_basin.to_csv(paths.metrics_by_basin_csv, index=False)
    improvement.to_csv(paths.improvement_by_basin_csv, index=False)
    diagnostics.to_csv(paths.prediction_diagnostics_csv, index=False)
    print("Test metrics:")
    print(overall[overall["split"] == "test"].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a GRACE mask-region forecasting experiment.")
    parser.add_argument("--experiment", default="africa_l3_no_madagascar", help="Experiment/output name.")
    parser.add_argument("--mask-zip", default=None, help="Path or filename under masks/ for the mask zip.")
    parser.add_argument("--basin-name-filter", default=None, help="Keep only mask names containing this text.")
    parser.add_argument("--basin-name-exclude", default=None, help="Exclude mask names containing this text.")
    parser.add_argument("--strict-mask-names", action="store_true", help="Require HydroBASINS-style mask filenames.")
    parser.add_argument("--force", action="store_true", help="Rebuild basin-month data even when provenance matches.")
    return parser.parse_args()


def experiment_from_args(args: argparse.Namespace) -> MaskExperiment:
    if args.mask_zip is None and args.experiment == "africa_l3_no_madagascar":
        return MaskExperiment.africa_l3_default()
    if args.mask_zip is None:
        raise ValueError("--mask-zip is required for custom experiments.")
    return MaskExperiment(
        paths=ExperimentPaths.from_name(args.experiment),
        mask_zip=_resolve_mask_zip(args.mask_zip),
        basin_name_filter=args.basin_name_filter,
        basin_name_exclude=args.basin_name_exclude,
        strict_mask_names=args.strict_mask_names,
    )


def main() -> None:
    args = parse_args()
    experiment = experiment_from_args(args)
    basin_month = build_basin_month(experiment, force=args.force)
    lagged = build_lagged(basin_month, experiment.paths)
    predictions = train_baselines(lagged, experiment.paths)
    predictions = train_gnns(lagged, predictions, experiment)
    save_metrics(predictions, experiment.paths)


if __name__ == "__main__":
    main()
