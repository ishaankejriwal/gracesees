from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import pandas as pd

from grace_gnn.config import (
    AFRICA_L3_MASK_ZIP_NAME,
    BASIN_MONTH_CSV,
    BASIN_MONTH_PROVENANCE_JSON,
    DATA_RAW,
    EXPERIMENT_REGION,
    GRACE_NETCDF_NAME,
    LAGGED_DATASET_CSV,
    LAGGED_DATASET_PROVENANCE_JSON,
    LAGS,
    RANDOM_SEED,
    REGION_CORRELATION_MATRIX_CSV,
    REGION_CORRELATION_PAIRS_CSV,
    REGION_IMPROVEMENT_BY_BASIN_CSV,
    REGION_METRICS_BY_BASIN_CSV,
    REGION_METRICS_OVERALL_CSV,
    REGION_OUTPUTS,
    REGION_PREDICTION_DIAGNOSTICS_CSV,
    REGION_PREDICTIONS_CSV,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VAL_FRACTION,
    ensure_dirs,
)
from grace_gnn.correlation import region_correlation_matrix, region_correlation_pairs
from grace_gnn.data import (
    aggregate_grace_netcdf_to_mask_zips,
    find_grace_netcdf,
    list_mask_members,
    read_basin_month_csv,
)
from grace_gnn.evaluate import graph_prediction_frame, prediction_frame
from grace_gnn.features import feature_columns, filter_region, make_lagged_dataset
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


def _l3_mask_zip() -> Path:
    path = ROOT / "masks" / AFRICA_L3_MASK_ZIP_NAME
    if not path.exists():
        raise FileNotFoundError(f"Missing L3 mask zip: {path}")
    return path


def _selected_l3_mask_members() -> pd.DataFrame:
    members = list_mask_members([_l3_mask_zip()], strict=True)
    members = members[~members["basin_name"].str.contains("madagascar", case=False, na=False)].copy()
    validate_unique_mask_members(members)
    return members


def _basin_month_provenance(grace_nc: Path, mask_zip: Path, members: pd.DataFrame) -> dict:
    return {
        "experiment_region": EXPERIMENT_REGION,
        "grace_netcdf": file_fingerprint(grace_nc),
        "mask_zips": [file_fingerprint(mask_zip)],
        "mask_format": "HydroBASINS .mask.csv/.mask.xyz members",
        "basin_name_exclude": "madagascar",
        "basin_count": int(members["basin_id"].nunique()),
        "basin_ids": sorted(members["basin_id"].astype(str).unique()),
    }


def _lagged_provenance(basin_month_provenance: dict) -> dict:
    return {
        "experiment_region": EXPERIMENT_REGION,
        "source_basin_month": basin_month_provenance,
        "lags": LAGS,
    }


def build_basin_month(force: bool = True) -> pd.DataFrame:
    ensure_dirs()
    grace_nc = find_grace_netcdf(DATA_RAW, GRACE_NETCDF_NAME)

    mask_zip = _l3_mask_zip()
    members = _selected_l3_mask_members()
    expected_basin_ids = set(members["basin_id"].astype(str))
    provenance = _basin_month_provenance(grace_nc, mask_zip, members)
    if BASIN_MONTH_CSV.exists() and not force:
        if not require_matching_provenance(BASIN_MONTH_PROVENANCE_JSON, provenance):
            raise ValueError(
                f"Refusing to reuse {BASIN_MONTH_CSV} because provenance is missing or stale. "
                "Run build_basin_month(force=True) or delete the stale processed file."
            )
        basin_month = filter_region(read_basin_month_csv(BASIN_MONTH_CSV), EXPERIMENT_REGION)
        validate_basin_month(basin_month, expected_basin_ids=expected_basin_ids)
        return basin_month

    print(f"Aggregating L3 Africa masks excluding Madagascar from {mask_zip.name}")
    basin_month = aggregate_grace_netcdf_to_mask_zips(
        grace_nc,
        [mask_zip],
        BASIN_MONTH_CSV,
        basin_name_exclude="madagascar",
    )
    basin_month = filter_region(basin_month, EXPERIMENT_REGION)
    validate_basin_month(basin_month, expected_basin_ids=expected_basin_ids)
    basin_month.to_csv(BASIN_MONTH_CSV, index=False)
    write_json(BASIN_MONTH_PROVENANCE_JSON, provenance)
    print(f"Saved {len(basin_month):,} basin-month rows to {BASIN_MONTH_CSV}")
    return basin_month


def build_lagged(basin_month: pd.DataFrame) -> pd.DataFrame:
    validate_basin_month(basin_month)
    lagged = make_lagged_dataset(basin_month, LAGS, LAGGED_DATASET_CSV)
    lagged = filter_region(lagged, EXPERIMENT_REGION)
    validate_lagged_dataset(lagged, expected_basin_ids=set(basin_month["basin_id"].astype(str).unique()))
    basin_month_provenance = read_json(BASIN_MONTH_PROVENANCE_JSON) or {"source": str(BASIN_MONTH_CSV)}
    write_json(LAGGED_DATASET_PROVENANCE_JSON, _lagged_provenance(basin_month_provenance))
    print(
        f"Saved {len(lagged):,} lagged samples across "
        f"{lagged['basin_id'].nunique()} L3 basins to {LAGGED_DATASET_CSV}"
    )
    return lagged


def train_baselines(lagged: pd.DataFrame) -> pd.DataFrame:
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
        _, preds = trainer()
        graph_type = "train_positive_corr_lag1" if model_name == "correlation_neighbor" else "none"
        for split_name, frame in splits.items():
            prediction_parts.append(prediction_frame(frame, preds[split_name], model_name, graph_type, split_name))

    corr = region_correlation_matrix(splits["train"])
    REGION_CORRELATION_MATRIX_CSV.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(REGION_CORRELATION_MATRIX_CSV)
    region_correlation_pairs(corr).to_csv(REGION_CORRELATION_PAIRS_CSV, index=False)

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
    REGION_PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(REGION_PREDICTIONS_CSV, index=False)
    return predictions


def train_gnns(lagged: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if not torch_available():
        return predictions

    mask_zip = _l3_mask_zip()
    basin_ids = sorted(lagged["basin_id"].astype(str).unique())
    basin_names = sorted(lagged["basin_name"].dropna().unique())
    real_directed = build_knn_edges_from_mask_zips(
        [mask_zip],
        basin_names,
        REGION_OUTPUTS / "edges_real_knn_directed.csv",
        k=3,
        graph_type="real_knn_directed",
    )
    graph_variants = {
        "real_knn_directed": real_directed,
        "real_knn_undirected": symmetrize_edges(real_directed, graph_type="real_knn_undirected"),
        "real_knn_reversed": reverse_edges(real_directed, graph_type="real_knn_reversed"),
        "random_degree_matched": make_degree_matched_random_edges(real_directed, basin_ids, seed=RANDOM_SEED),
    }
    for graph_type, edges in graph_variants.items():
        validate_edges(edges, set(basin_ids), graph_type=graph_type)
        save_edges(edges, REGION_OUTPUTS / f"edges_{graph_type}.csv")
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
    combined.to_csv(REGION_PREDICTIONS_CSV, index=False)
    return combined


def save_metrics(predictions: pd.DataFrame) -> None:
    overall = metrics_overall(predictions).sort_values(["split", "rmse_cm"])
    by_basin = metrics_by_basin(predictions, split="test").sort_values(["basin_name", "rmse_cm"])
    improvement = improvement_by_basin(by_basin)
    diagnostics = prediction_diagnostics(predictions)

    overall.to_csv(REGION_METRICS_OVERALL_CSV, index=False)
    by_basin.to_csv(REGION_METRICS_BY_BASIN_CSV, index=False)
    improvement.to_csv(REGION_IMPROVEMENT_BY_BASIN_CSV, index=False)
    diagnostics.to_csv(REGION_PREDICTION_DIAGNOSTICS_CSV, index=False)
    print("Test metrics:")
    print(overall[overall["split"] == "test"].to_string(index=False))


def main() -> None:
    basin_month = build_basin_month(force=False)
    lagged = build_lagged(basin_month)
    predictions = train_baselines(lagged)
    predictions = train_gnns(lagged, predictions)
    save_metrics(predictions)


if __name__ == "__main__":
    main()
