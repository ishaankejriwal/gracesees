from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

import numpy as np
import pandas as pd

from grace_gnn.config import (
    LAGGED_DATASET_CSV,
    LAGGED_GRACE_ERA5_DATASET_CSV,
    RANDOM_SEED,
    REGION_METRICS_OVERALL_CSV,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VAL_FRACTION,
)
from grace_gnn.evaluate import prediction_frame
from grace_gnn.graph import save_edges
from grace_gnn.metrics import metrics_by_basin, metrics_overall, prediction_diagnostics
from grace_gnn.splits import chronological_fraction_split
from grace_gnn.validation import validate_edges, validate_lagged_dataset, write_json
from scripts.run_africa_l3_extra_architectures import (
    add_neighbor_lag_features,
    correlation_topk_edges,
    train_residual_mlp,
    train_ridge_predictions,
)


OUTPUT_DIR = ROOT / "outputs" / "africa_l3_era5_one_month"
PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"
METRICS_OVERALL_CSV = OUTPUT_DIR / "metrics_overall.csv"
METRICS_BY_REGION_CSV = OUTPUT_DIR / "metrics_by_region.csv"
PREDICTION_DIAGNOSTICS_CSV = OUTPUT_DIR / "prediction_diagnostics.csv"
SUMMARY_CSV = OUTPUT_DIR / "era5_vs_grace_only_summary.csv"
VALIDATION_JSON = OUTPUT_DIR / "run_validation.json"
GRACE_ONLY_BASELINE_FALLBACK_RMSE_CM = 2.3769

EXPECTED_ROWS = 6734
EXPECTED_BASINS = 37
TARGET_MONTH_ERA5_COLUMNS = {"era5_tp_mm", "era5_ro_mm", "era5_evap_mm"}


def era5_feature_columns(df: pd.DataFrame) -> list[str]:
    grace_lags = sorted(
        [col for col in df.columns if col.startswith("lag_")],
        key=lambda col: int(col.rsplit("_", 1)[1]),
    )
    era5_lags = sorted(
        [col for col in df.columns if col.startswith("era5_") and "_lag_" in col],
        key=lambda col: (col.split("_lag_", 1)[0], int(col.rsplit("_", 1)[1])),
    )
    return [*grace_lags, *era5_lags]


def split_signature(splits: dict[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    return {
        split_name: {
            "rows": int(len(frame)),
            "dates": [str(pd.Timestamp(date).date()) for date in sorted(frame["date"].unique())],
            "min_date": str(frame["date"].min().date()),
            "max_date": str(frame["date"].max().date()),
        }
        for split_name, frame in splits.items()
    }


def validate_inputs(lagged: pd.DataFrame, features: list[str]) -> dict[str, object]:
    validate_lagged_dataset(lagged)
    if len(lagged) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} rows, found {len(lagged):,}.")
    basin_count = lagged["basin_id"].astype(str).nunique()
    if basin_count != EXPECTED_BASINS:
        raise ValueError(f"Expected {EXPECTED_BASINS} basins, found {basin_count}.")
    forbidden = sorted(TARGET_MONTH_ERA5_COLUMNS & set(features))
    if forbidden:
        raise ValueError(f"Target-month ERA5 columns leaked into model features: {forbidden}")
    missing = sorted(set(features) - set(lagged.columns))
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")
    if lagged[features].isna().any().any():
        null_counts = lagged[features].isna().sum()
        offenders = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Feature matrix contains missing values: {offenders}")
    unlagged_present = sorted(TARGET_MONTH_ERA5_COLUMNS & set(lagged.columns))

    grace_only = pd.read_csv(LAGGED_DATASET_CSV, parse_dates=["date"])
    validate_lagged_dataset(grace_only)
    era5_splits = chronological_fraction_split(lagged, TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION)
    grace_splits = chronological_fraction_split(grace_only, TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION)
    era5_sig = split_signature(era5_splits)
    grace_sig = split_signature(grace_splits)
    for split_name in ["train", "val", "test"]:
        if era5_sig[split_name]["dates"] != grace_sig[split_name]["dates"]:
            raise ValueError(f"{split_name} dates do not match the GRACE-only split.")

    return {
        "rows": int(len(lagged)),
        "basins": int(basin_count),
        "feature_count": int(len(features)),
        "features": features,
        "target_month_era5_columns_in_dataset": unlagged_present,
        "split_signature": era5_sig,
    }


def frame_predictions(
    splits: dict[str, pd.DataFrame],
    preds: dict[str, np.ndarray],
    model_name: str,
    graph_type: str,
) -> list[pd.DataFrame]:
    return [
        prediction_frame(splits[split_name], preds[split_name], model_name, graph_type, split_name)
        for split_name in ["train", "val", "test"]
    ]


def train_neighbor_residual(
    splits: dict[str, pd.DataFrame],
    edges: pd.DataFrame,
    features: list[str],
    model_name: str,
    graph_type: str,
) -> list[pd.DataFrame]:
    neighbor_splits = {
        split_name: add_neighbor_lag_features(frame, edges, features)
        for split_name, frame in splits.items()
    }
    neighbor_cols = [*features, *[f"neighbor_{col}" for col in features]]
    _, base_preds = train_ridge_predictions(
        neighbor_splits["train"],
        neighbor_splits["val"],
        neighbor_splits["test"],
        neighbor_cols,
    )
    _, residual_preds = train_residual_mlp(
        neighbor_splits["train"],
        neighbor_splits["val"],
        neighbor_splits["test"],
        neighbor_cols,
        base_preds,
        seed=RANDOM_SEED + 1,
    )
    return frame_predictions(neighbor_splits, residual_preds, model_name, graph_type)


def make_source_degree_matched_random_edges(
    edges: pd.DataFrame,
    basin_ids: list[str],
    seed: int = RANDOM_SEED,
    graph_type: str = "random_degree_matched",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    edge_data = edges.copy()
    edge_data["src_basin_id"] = edge_data["src_basin_id"].astype(str)
    source_degrees = edge_data.groupby("src_basin_id").size().to_dict()
    rows = []
    existing: set[tuple[str, str]] = set()
    for src in basin_ids:
        degree = int(source_degrees.get(str(src), 0))
        if degree <= 0:
            continue
        candidates = [str(bid) for bid in basin_ids if str(bid) != str(src)]
        if degree > len(candidates):
            raise ValueError(f"Cannot draw {degree} non-self random edges for node {src}.")
        rng.shuffle(candidates)
        picked = 0
        for dst in candidates:
            pair = (str(src), dst)
            if pair in existing:
                continue
            existing.add(pair)
            rows.append(pair)
            picked += 1
            if picked == degree:
                break
    random_edges = pd.DataFrame(rows, columns=["src_basin_id", "dst_basin_id"])
    random_edges["weight"] = 1.0
    random_edges["graph_type"] = graph_type
    return random_edges


def grace_only_best_rmse() -> float:
    if not REGION_METRICS_OVERALL_CSV.exists():
        return GRACE_ONLY_BASELINE_FALLBACK_RMSE_CM
    metrics = pd.read_csv(REGION_METRICS_OVERALL_CSV)
    test = metrics[metrics["split"].eq("test")]
    if test.empty:
        return GRACE_ONLY_BASELINE_FALLBACK_RMSE_CM
    return float(test.sort_values("rmse_cm").iloc[0]["rmse_cm"])


def save_summary(overall: pd.DataFrame, grace_only_rmse: float) -> pd.DataFrame:
    test = overall[overall["split"].eq("test")].copy()
    rows = []
    for model_name, graph_type in [
        ("ridge_ar_era5", "own_lags"),
        ("ridge_residual_mlp_era5", "own_lags"),
        ("ridge_neighbor_residual_mlp_era5", "corr_top3_directed"),
        ("ridge_neighbor_residual_mlp_era5", "random_degree_matched"),
    ]:
        match = test[test["model_name"].eq(model_name) & test["graph_type"].eq(graph_type)]
        if match.empty:
            continue
        row = match.iloc[0].to_dict()
        row["grace_only_best_rmse_cm"] = grace_only_rmse
        row["rmse_delta_vs_grace_only_cm"] = row["rmse_cm"] - grace_only_rmse
        rows.append(row)
    summary = pd.DataFrame(rows)

    corr = summary[
        summary["model_name"].eq("ridge_neighbor_residual_mlp_era5")
        & summary["graph_type"].eq("corr_top3_directed")
    ]
    random = summary[
        summary["model_name"].eq("ridge_neighbor_residual_mlp_era5")
        & summary["graph_type"].eq("random_degree_matched")
    ]
    if not corr.empty and not random.empty:
        corr_rmse = float(corr.iloc[0]["rmse_cm"])
        random_rmse = float(random.iloc[0]["rmse_cm"])
        useful = corr_rmse < grace_only_rmse and corr_rmse < random_rmse
        summary["era5_acceptance_result"] = (
            "useful" if useful else "not_beneficial_keep_grace_only_recommendation"
        )
        summary["corr_minus_random_control_rmse_cm"] = corr_rmse - random_rmse
    summary.to_csv(SUMMARY_CSV, index=False)
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lagged = pd.read_csv(LAGGED_GRACE_ERA5_DATASET_CSV, parse_dates=["date"])
    lagged["basin_id"] = lagged["basin_id"].astype(str)
    features = era5_feature_columns(lagged)
    validation = validate_inputs(lagged, features)

    splits = chronological_fraction_split(lagged, TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION)
    for frame in splits.values():
        validate_lagged_dataset(frame)

    prediction_parts = []
    _, ridge_preds = train_ridge_predictions(splits["train"], splits["val"], splits["test"], features)
    prediction_parts.extend(frame_predictions(splits, ridge_preds, "ridge_ar_era5", "own_lags"))

    _, residual_preds = train_residual_mlp(
        splits["train"],
        splits["val"],
        splits["test"],
        features,
        ridge_preds,
        seed=RANDOM_SEED,
    )
    prediction_parts.extend(frame_predictions(splits, residual_preds, "ridge_residual_mlp_era5", "own_lags"))

    basin_ids = sorted(lagged["basin_id"].astype(str).unique())
    corr_edges = correlation_topk_edges(splits["train"], top_k=3)
    validate_edges(corr_edges, set(basin_ids), graph_type="corr_top3_directed")
    save_edges(corr_edges, OUTPUT_DIR / "edges_corr_top3_directed.csv")

    random_edges = make_source_degree_matched_random_edges(corr_edges, basin_ids, seed=RANDOM_SEED)
    validate_edges(random_edges, set(basin_ids), graph_type="random_degree_matched")
    save_edges(random_edges, OUTPUT_DIR / "edges_random_degree_matched.csv")

    for graph_type, edges in [
        ("corr_top3_directed", corr_edges),
        ("random_degree_matched", random_edges),
    ]:
        prediction_parts.extend(
            train_neighbor_residual(
                splits,
                edges,
                features,
                "ridge_neighbor_residual_mlp_era5",
                graph_type,
            )
        )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions.to_csv(PREDICTIONS_CSV, index=False)

    overall = metrics_overall(predictions).sort_values(["split", "rmse_cm"])
    by_region = metrics_by_basin(predictions, split="test").sort_values(["basin_name", "rmse_cm"])
    diagnostics = prediction_diagnostics(predictions).sort_values(["split", "model_name", "graph_type"])
    overall.to_csv(METRICS_OVERALL_CSV, index=False)
    by_region.to_csv(METRICS_BY_REGION_CSV, index=False)
    diagnostics.to_csv(PREDICTION_DIAGNOSTICS_CSV, index=False)
    grace_only_rmse = grace_only_best_rmse()
    summary = save_summary(overall, grace_only_rmse)

    validation["output_dir"] = str(OUTPUT_DIR)
    validation["grace_only_metrics_source"] = str(REGION_METRICS_OVERALL_CSV)
    validation["grace_only_best_rmse_cm"] = grace_only_rmse
    validation["corr_edge_count"] = int(len(corr_edges))
    validation["random_edge_count"] = int(len(random_edges))
    write_json(VALIDATION_JSON, validation)

    print("ERA5 one-month test metrics:")
    print(overall[overall["split"].eq("test")].to_string(index=False))
    print("\nERA5 vs GRACE-only summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
