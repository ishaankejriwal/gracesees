from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

import pandas as pd

from grace_gnn.config import (
    LAGGED_DATASET_CSV,
    RANDOM_SEED,
    REGION_IMPROVEMENT_BY_BASIN_CSV,
    REGION_METRICS_BY_BASIN_CSV,
    REGION_METRICS_OVERALL_CSV,
    REGION_OUTPUTS,
    REGION_PREDICTION_DIAGNOSTICS_CSV,
    REGION_PREDICTIONS_CSV,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VAL_FRACTION,
)
from grace_gnn.evaluate import prediction_frame
from grace_gnn.features import feature_columns
from grace_gnn.metrics import improvement_by_basin, metrics_by_basin, metrics_overall, prediction_diagnostics
from grace_gnn.splits import chronological_fraction_split
from grace_gnn.validation import validate_edges, validate_lagged_dataset
from scripts.run_africa_l3_extra_architectures import (
    add_neighbor_lag_features,
    train_residual_mlp,
    train_ridge_predictions,
)


MODEL_NAME = "ridge_neighbor_residual_mlp"
GRAPH_TYPE = "random_degree_matched"


def main() -> None:
    lagged = pd.read_csv(LAGGED_DATASET_CSV, parse_dates=["date"])
    validate_lagged_dataset(lagged)
    splits = chronological_fraction_split(lagged, TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION)
    for frame in splits.values():
        validate_lagged_dataset(frame)

    lag_cols = feature_columns(lagged)
    basin_ids = set(lagged["basin_id"].astype(str).unique())
    edges_path = REGION_OUTPUTS / f"edges_{GRAPH_TYPE}.csv"
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing control graph: {edges_path}")
    edges = pd.read_csv(edges_path)
    validate_edges(edges, basin_ids, graph_type=GRAPH_TYPE)

    neighbor_splits = {
        split_name: add_neighbor_lag_features(frame, edges, lag_cols)
        for split_name, frame in splits.items()
    }
    neighbor_cols = [*lag_cols, *[f"neighbor_{col}" for col in lag_cols]]
    _, neighbor_base_preds = train_ridge_predictions(
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
        neighbor_base_preds,
        seed=RANDOM_SEED + 1,
    )

    new_predictions = pd.concat(
        [
            prediction_frame(neighbor_splits[split_name], residual_preds[split_name], MODEL_NAME, GRAPH_TYPE, split_name)
            for split_name in ["train", "val", "test"]
        ],
        ignore_index=True,
    )

    existing = pd.read_csv(REGION_PREDICTIONS_CSV, parse_dates=["date"]) if REGION_PREDICTIONS_CSV.exists() else pd.DataFrame()
    if not existing.empty:
        keep = ~(
            existing["model_name"].eq(MODEL_NAME)
            & existing["graph_type"].eq(GRAPH_TYPE)
        )
        existing = existing[keep].copy()
    combined = pd.concat([existing, new_predictions], ignore_index=True)
    combined = combined.drop_duplicates(["date", "basin_id", "model_name", "graph_type", "split"], keep="last")
    combined.to_csv(REGION_PREDICTIONS_CSV, index=False)

    overall = metrics_overall(combined).sort_values(["split", "rmse_cm"])
    by_basin = metrics_by_basin(combined, split="test").sort_values(["basin_name", "rmse_cm"])
    improvement = improvement_by_basin(by_basin)
    diagnostics = prediction_diagnostics(combined)
    overall.to_csv(REGION_METRICS_OVERALL_CSV, index=False)
    by_basin.to_csv(REGION_METRICS_BY_BASIN_CSV, index=False)
    improvement.to_csv(REGION_IMPROVEMENT_BY_BASIN_CSV, index=False)
    diagnostics.to_csv(REGION_PREDICTION_DIAGNOSTICS_CSV, index=False)

    compare = overall[
        overall["split"].eq("test")
        & (
            (overall["model_name"].eq(MODEL_NAME) & overall["graph_type"].isin(["corr_top3_directed", GRAPH_TYPE]))
            | (overall["model_name"].eq("ridge_residual_mlp") & overall["graph_type"].eq("own_lags"))
            | (overall["model_name"].eq("ridge_ar") & overall["graph_type"].eq("none"))
        )
    ].sort_values("rmse_cm")
    print(compare.to_string(index=False))


if __name__ == "__main__":
    main()
