from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

import numpy as np
import pandas as pd

from grace_gnn.config import (
    LAGGED_GRACE_ERA5_DATASET_CSV,
    RANDOM_SEED,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VAL_FRACTION,
)
from grace_gnn.evaluate import graph_prediction_frame, prediction_frame
from grace_gnn.graph import save_edges
from grace_gnn.metrics import metrics_by_basin, metrics_overall, prediction_diagnostics
from grace_gnn.models import train_manual_gcn, train_mlp, train_random_forest, train_xgboost
from grace_gnn.splits import chronological_fraction_split
from grace_gnn.validation import validate_edges, validate_lagged_dataset, write_json
from scripts.run_africa_l3_era5_one_month import (
    PREDICTIONS_CSV as ERA5_BASELINE_PREDICTIONS_CSV,
    era5_feature_columns,
    make_source_degree_matched_random_edges,
    validate_inputs,
)
from scripts.run_africa_l3_extra_architectures import train_ridge_predictions
from scripts.run_africa_l3_gnn_embeddings import (
    add_embeddings,
    correlation_topk_edges,
    train_gnn_embeddings,
    train_residual_tabular_models,
)


OUTPUT_DIR = ROOT / "outputs" / "africa_l3_era5_heavy_architectures"
PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"
METRICS_OVERALL_CSV = OUTPUT_DIR / "metrics_overall.csv"
METRICS_BY_REGION_CSV = OUTPUT_DIR / "metrics_by_region.csv"
PREDICTION_DIAGNOSTICS_CSV = OUTPUT_DIR / "prediction_diagnostics.csv"
SUMMARY_CSV = OUTPUT_DIR / "heavy_architecture_summary.csv"
VALIDATION_JSON = OUTPUT_DIR / "run_validation.json"


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


def rename_prediction_model(frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    out = frame.copy()
    out["model_name"] = model_name
    return out


def train_gnn_embedding_residuals(
    splits: dict[str, pd.DataFrame],
    features: list[str],
    edges: pd.DataFrame,
    basin_ids: list[str],
    base_preds: dict[str, np.ndarray],
) -> list[pd.DataFrame]:
    _, embedding_frames, emb_cols = train_gnn_embeddings(
        splits["train"],
        splits["val"],
        splits["test"],
        features,
        edges,
        basin_ids,
        base_preds,
        embedding_dim=16,
        seed=RANDOM_SEED,
    )
    enhanced_splits = {
        split_name: add_embeddings(frame, embedding_frames[split_name], emb_cols)
        for split_name, frame in splits.items()
    }
    second_stage_features = [*features, *emb_cols]
    model_preds = train_residual_tabular_models(enhanced_splits, second_stage_features, base_preds)

    parts = []
    rename = {
        "ridge_gnn_embedding_residual": "ridge_gnn_embedding_residual_era5",
        "random_forest_gnn_embedding_residual": "random_forest_gnn_embedding_residual_era5",
        "xgboost_gnn_embedding_residual": "xgboost_gnn_embedding_residual_era5",
    }
    for model_name, preds in model_preds.items():
        parts.extend(frame_predictions(enhanced_splits, preds, rename[model_name], "corr_top3_directed"))
    return parts


def save_summary(overall: pd.DataFrame) -> pd.DataFrame:
    test = overall[overall["split"].eq("test")].sort_values("rmse_cm").copy()
    keep_cols = ["model_name", "graph_type", "rmse_cm", "mae_cm", "pearson_r", "nse_optional"]
    summary = test[keep_cols].reset_index(drop=True)
    summary.insert(0, "rank_rmse", np.arange(1, len(summary) + 1))
    baseline = summary[
        summary["model_name"].eq("ridge_neighbor_residual_mlp_era5")
        & summary["graph_type"].eq("corr_top3_directed")
    ]
    if not baseline.empty:
        best_baseline = float(baseline.iloc[0]["rmse_cm"])
        summary["rmse_delta_vs_best_era5_baseline_cm"] = summary["rmse_cm"] - best_baseline
    summary.to_csv(SUMMARY_CSV, index=False)
    return summary


def load_baseline_predictions() -> pd.DataFrame:
    if not ERA5_BASELINE_PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"Missing ERA5 baseline predictions: {ERA5_BASELINE_PREDICTIONS_CSV}. "
            "Run scripts/run_africa_l3_era5_one_month.py first."
        )
    return pd.read_csv(ERA5_BASELINE_PREDICTIONS_CSV, parse_dates=["date"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lagged = pd.read_csv(LAGGED_GRACE_ERA5_DATASET_CSV, parse_dates=["date"])
    lagged["basin_id"] = lagged["basin_id"].astype(str)
    features = era5_feature_columns(lagged)
    validation = validate_inputs(lagged, features)

    splits = chronological_fraction_split(lagged, TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION)
    for frame in splits.values():
        validate_lagged_dataset(frame)

    basin_ids = sorted(lagged["basin_id"].astype(str).unique())
    corr_edges = correlation_topk_edges(splits["train"], top_k=3)
    validate_edges(corr_edges, set(basin_ids), graph_type="corr_top3_directed")
    save_edges(corr_edges, OUTPUT_DIR / "edges_corr_top3_directed.csv")

    random_edges = make_source_degree_matched_random_edges(corr_edges, basin_ids, seed=RANDOM_SEED)
    validate_edges(random_edges, set(basin_ids), graph_type="random_degree_matched")
    save_edges(random_edges, OUTPUT_DIR / "edges_random_degree_matched.csv")

    prediction_parts = [load_baseline_predictions()]

    _, ridge_preds = train_ridge_predictions(splits["train"], splits["val"], splits["test"], features)

    for model_name, trainer in [
        (
            "random_forest_ar_era5",
            lambda: train_random_forest(splits["train"], splits["val"], splits["test"], features, seed=RANDOM_SEED),
        ),
        (
            "xgboost_ar_era5",
            lambda: train_xgboost(splits["train"], splits["val"], splits["test"], features, seed=RANDOM_SEED),
        ),
        (
            "basin_only_mlp_era5",
            lambda: train_mlp(splits["train"], splits["val"], splits["test"], features, seed=RANDOM_SEED),
        ),
    ]:
        _, preds = trainer()
        prediction_parts.extend(frame_predictions(splits, preds, model_name, "own_lags"))

    for model_name, use_residual, model_seed_offset in [
        ("neighbor_gnn_era5", False, 10),
        ("residual_neighbor_gnn_era5", True, 20),
    ]:
        for graph_type, edges, graph_seed_offset in [
            ("corr_top3_directed", corr_edges, 0),
            ("random_degree_matched", random_edges, 1),
        ]:
            _, gcn_preds = train_manual_gcn(
                splits["train"],
                splits["val"],
                splits["test"],
                features,
                edges,
                basin_ids,
                seed=RANDOM_SEED + model_seed_offset + graph_seed_offset,
                epochs=300,
                residual=use_residual,
            )
            for split_name, pred_df in gcn_preds.items():
                prediction_parts.append(graph_prediction_frame(pred_df, model_name, graph_type, split_name))

    prediction_parts.extend(
        train_gnn_embedding_residuals(splits, features, corr_edges, basin_ids, ridge_preds)
    )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions = predictions.drop_duplicates(
        ["date", "basin_id", "model_name", "graph_type", "split"],
        keep="last",
    )
    predictions.to_csv(PREDICTIONS_CSV, index=False)

    overall = metrics_overall(predictions).sort_values(["split", "rmse_cm"])
    by_region = metrics_by_basin(predictions, split="test").sort_values(["basin_name", "rmse_cm"])
    diagnostics = prediction_diagnostics(predictions).sort_values(["split", "model_name", "graph_type"])
    overall.to_csv(METRICS_OVERALL_CSV, index=False)
    by_region.to_csv(METRICS_BY_REGION_CSV, index=False)
    diagnostics.to_csv(PREDICTION_DIAGNOSTICS_CSV, index=False)
    summary = save_summary(overall)

    validation["output_dir"] = str(OUTPUT_DIR)
    validation["corr_edge_count"] = int(len(corr_edges))
    validation["random_edge_count"] = int(len(random_edges))
    validation["included_baseline_predictions"] = str(ERA5_BASELINE_PREDICTIONS_CSV)
    write_json(VALIDATION_JSON, validation)

    print("ERA5 heavy architecture test ranking:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
