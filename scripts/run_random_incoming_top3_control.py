from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

import numpy as np
import pandas as pd

from grace_gnn.config import RANDOM_SEED, TEST_FRACTION, TRAIN_FRACTION, VAL_FRACTION
from grace_gnn.evaluate import prediction_frame
from grace_gnn.features import feature_columns
from grace_gnn.metrics import metrics_by_basin, metrics_overall, prediction_diagnostics
from grace_gnn.splits import chronological_fraction_split
from grace_gnn.validation import validate_edges, validate_lagged_dataset
from scripts.run_africa_l3_era5_one_month import era5_feature_columns
from scripts.run_africa_l3_extra_architectures import (
    add_neighbor_lag_features,
    correlation_topk_edges,
    train_residual_mlp,
    train_ridge_predictions,
)


GRAPH_TYPE = "random_incoming_top3"


def random_incoming_topk_edges(
    corr_edges: pd.DataFrame,
    basin_ids: list[str],
    top_k: int = 3,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    basin_ids = [str(basin_id) for basin_id in basin_ids]
    incoming_degree = corr_edges.groupby(corr_edges["dst_basin_id"].astype(str)).size().to_dict()
    rows = []
    for dst in basin_ids:
        degree = int(incoming_degree.get(dst, top_k))
        candidates = [src for src in basin_ids if src != dst]
        if degree > len(candidates):
            raise ValueError(f"Cannot draw {degree} random incoming edges for basin {dst}.")
        picked = rng.choice(candidates, size=degree, replace=False)
        for src in picked:
            rows.append({"src_basin_id": str(src), "dst_basin_id": dst, "weight": 1.0})
    edges = pd.DataFrame(rows)
    edges["graph_type"] = GRAPH_TYPE
    return edges


def append_predictions(
    existing_predictions: pd.DataFrame,
    new_predictions: pd.DataFrame,
    model_names: set[str],
) -> pd.DataFrame:
    if existing_predictions.empty:
        return new_predictions
    keep = ~(
        existing_predictions["model_name"].isin(model_names)
        & existing_predictions["graph_type"].eq(GRAPH_TYPE)
    )
    combined = pd.concat([existing_predictions[keep].copy(), new_predictions], ignore_index=True)
    return combined.drop_duplicates(["date", "basin_id", "model_name", "graph_type", "split"], keep="last")


def run_control(
    lagged_csv: Path,
    output_dir: Path,
    era5: bool = False,
    seed: int = RANDOM_SEED,
) -> None:
    predictions_csv = output_dir / "predictions.csv"
    metrics_overall_csv = output_dir / "metrics_overall.csv"
    metrics_by_region_csv = output_dir / "metrics_by_region.csv"
    diagnostics_csv = output_dir / "prediction_diagnostics.csv"
    edges_csv = output_dir / f"edges_{GRAPH_TYPE}.csv"

    lagged = pd.read_csv(lagged_csv, parse_dates=["date"])
    lagged["basin_id"] = lagged["basin_id"].astype(str)
    validate_lagged_dataset(lagged)
    splits = chronological_fraction_split(lagged, TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION)
    for frame in splits.values():
        validate_lagged_dataset(frame)

    features = era5_feature_columns(lagged) if era5 else feature_columns(lagged)
    basin_ids = sorted(lagged["basin_id"].astype(str).unique())
    corr_edges = correlation_topk_edges(splits["train"], top_k=3)
    validate_edges(corr_edges, set(basin_ids), graph_type="corr_top3_directed")
    random_edges = random_incoming_topk_edges(corr_edges, basin_ids, top_k=3, seed=seed)
    validate_edges(random_edges, set(basin_ids), graph_type=GRAPH_TYPE)
    output_dir.mkdir(parents=True, exist_ok=True)
    random_edges.to_csv(edges_csv, index=False)

    neighbor_splits = {
        split_name: add_neighbor_lag_features(frame, random_edges, features)
        for split_name, frame in splits.items()
    }
    neighbor_cols = [*features, *[f"neighbor_{col}" for col in features]]
    _, neighbor_preds = train_ridge_predictions(
        neighbor_splits["train"],
        neighbor_splits["val"],
        neighbor_splits["test"],
        neighbor_cols,
    )

    prefix = "ridge_neighbor" if not era5 else "ridge_neighbor_era5"
    residual_name = "ridge_neighbor_residual_mlp" if not era5 else "ridge_neighbor_residual_mlp_era5"
    ar_name = "ridge_neighbor_ar" if not era5 else "ridge_neighbor_ar_era5"

    prediction_parts = []
    for split_name in ["train", "val", "test"]:
        prediction_parts.append(
            prediction_frame(neighbor_splits[split_name], neighbor_preds[split_name], ar_name, GRAPH_TYPE, split_name)
        )

    _, residual_preds = train_residual_mlp(
        neighbor_splits["train"],
        neighbor_splits["val"],
        neighbor_splits["test"],
        neighbor_cols,
        neighbor_preds,
        seed=seed + 1,
    )
    for split_name in ["train", "val", "test"]:
        prediction_parts.append(
            prediction_frame(
                neighbor_splits[split_name],
                residual_preds[split_name],
                residual_name,
                GRAPH_TYPE,
                split_name,
            )
        )

    new_predictions = pd.concat(prediction_parts, ignore_index=True)
    existing = pd.read_csv(predictions_csv, parse_dates=["date"]) if predictions_csv.exists() else pd.DataFrame()
    combined = append_predictions(existing, new_predictions, {ar_name, residual_name})
    combined.to_csv(predictions_csv, index=False)

    overall = metrics_overall(combined).sort_values(["split", "rmse_cm"])
    by_region = metrics_by_basin(combined, split="test").sort_values(["basin_name", "rmse_cm"])
    diagnostics = prediction_diagnostics(combined).sort_values(["split", "model_name", "graph_type"])
    overall.to_csv(metrics_overall_csv, index=False)
    by_region.to_csv(metrics_by_region_csv, index=False)
    diagnostics.to_csv(diagnostics_csv, index=False)

    compare = overall[
        overall["split"].eq("test")
        & overall["model_name"].isin([ar_name, residual_name])
        & overall["graph_type"].isin(["corr_top3_directed", "random_degree_matched", GRAPH_TYPE])
    ].sort_values("rmse_cm")
    print(f"\n{output_dir}")
    print(compare.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run corrected random incoming top-3 control.")
    parser.add_argument("--lagged-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--era5", action="store_true")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    run_control(args.lagged_csv, args.output_dir, era5=args.era5, seed=args.seed)


if __name__ == "__main__":
    main()
