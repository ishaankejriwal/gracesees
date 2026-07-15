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
from grace_gnn.evaluate import prediction_frame
from grace_gnn.graph import save_edges
from grace_gnn.metrics import metrics_by_basin, metrics_overall, prediction_diagnostics
from grace_gnn.splits import chronological_fraction_split
from grace_gnn.validation import validate_edges, validate_lagged_dataset, write_json
from scripts.run_africa_l3_era5_one_month import (
    EXPECTED_BASINS,
    EXPECTED_ROWS,
    era5_feature_columns,
    make_source_degree_matched_random_edges,
    split_signature,
    validate_inputs,
)
from scripts.run_africa_l3_extra_architectures import (
    add_neighbor_lag_features,
    correlation_topk_edges,
    train_ridge_predictions,
)


OUTPUT_DIR = ROOT / "outputs" / "africa_l3_era5_svr"
PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"
METRICS_OVERALL_CSV = OUTPUT_DIR / "metrics_overall.csv"
METRICS_BY_REGION_CSV = OUTPUT_DIR / "metrics_by_region.csv"
PREDICTION_DIAGNOSTICS_CSV = OUTPUT_DIR / "prediction_diagnostics.csv"
SUMMARY_CSV = OUTPUT_DIR / "svr_summary.csv"
VALIDATION_JSON = OUTPUT_DIR / "run_validation.json"
BEST_ERA5_BASELINE_RMSE_CM = 1.8493880852710136


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


def make_svr_model(kind: str):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR, LinearSVR

    if kind == "linear":
        regressor = LinearSVR(
            C=1.0,
            epsilon=0.05,
            max_iter=50_000,
            random_state=RANDOM_SEED,
            dual="auto",
        )
    elif kind == "rbf":
        regressor = SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.05)
    else:
        raise ValueError(f"Unsupported SVR kind: {kind}")

    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), regressor),
        transformer=StandardScaler(),
    )


def train_direct_svr(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    kind: str,
) -> tuple[object, dict[str, np.ndarray]]:
    model = make_svr_model(kind)
    model.fit(train_df[feature_cols], train_df["target_twsa_cm"])

    def predict(frame: pd.DataFrame) -> np.ndarray:
        return model.predict(frame[feature_cols]) if len(frame) else np.array([])

    return model, {"train": predict(train_df), "val": predict(val_df), "test": predict(test_df)}


def train_residual_svr(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    base_preds: dict[str, np.ndarray],
    kind: str = "rbf",
) -> tuple[object, dict[str, np.ndarray]]:
    model = make_svr_model(kind)
    train_residual = train_df["target_twsa_cm"].to_numpy() - base_preds["train"]
    model.fit(train_df[feature_cols], train_residual)

    def predict(frame: pd.DataFrame, split_name: str) -> np.ndarray:
        if frame.empty:
            return np.array([])
        return base_preds[split_name] + model.predict(frame[feature_cols])

    return model, {
        "train": predict(train_df, "train"),
        "val": predict(val_df, "val"),
        "test": predict(test_df, "test"),
    }


def save_summary(overall: pd.DataFrame) -> pd.DataFrame:
    test = overall[overall["split"].eq("test")].copy()
    summary = test.sort_values("rmse_cm").reset_index(drop=True)
    summary.insert(0, "rank_rmse", np.arange(1, len(summary) + 1))
    summary["best_era5_baseline_rmse_cm"] = BEST_ERA5_BASELINE_RMSE_CM
    summary["rmse_delta_vs_best_era5_baseline_cm"] = summary["rmse_cm"] - BEST_ERA5_BASELINE_RMSE_CM
    summary.to_csv(SUMMARY_CSV, index=False)
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lagged = pd.read_csv(LAGGED_GRACE_ERA5_DATASET_CSV, parse_dates=["date"])
    lagged["basin_id"] = lagged["basin_id"].astype(str)
    features = era5_feature_columns(lagged)
    validation = validate_inputs(lagged, features)
    if validation["rows"] != EXPECTED_ROWS or validation["basins"] != EXPECTED_BASINS:
        raise ValueError(f"Unexpected dataset shape: {validation['rows']} rows, {validation['basins']} basins.")

    splits = chronological_fraction_split(lagged, TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION)
    for frame in splits.values():
        validate_lagged_dataset(frame)

    prediction_parts: list[pd.DataFrame] = []

    _, linear_preds = train_direct_svr(splits["train"], splits["val"], splits["test"], features, kind="linear")
    prediction_parts.extend(frame_predictions(splits, linear_preds, "linear_svr_era5", "own_lags"))

    _, rbf_preds = train_direct_svr(splits["train"], splits["val"], splits["test"], features, kind="rbf")
    prediction_parts.extend(frame_predictions(splits, rbf_preds, "rbf_svr_era5", "own_lags"))

    basin_ids = sorted(lagged["basin_id"].astype(str).unique())
    corr_edges = correlation_topk_edges(splits["train"], top_k=3)
    validate_edges(corr_edges, set(basin_ids), graph_type="corr_top3_directed")
    save_edges(corr_edges, OUTPUT_DIR / "edges_corr_top3_directed.csv")

    random_edges = make_source_degree_matched_random_edges(corr_edges, basin_ids, seed=RANDOM_SEED)
    validate_edges(random_edges, set(basin_ids), graph_type="random_degree_matched")
    save_edges(random_edges, OUTPUT_DIR / "edges_random_degree_matched.csv")

    neighbor_splits = {
        split_name: add_neighbor_lag_features(frame, corr_edges, features)
        for split_name, frame in splits.items()
    }
    neighbor_features = [*features, *[f"neighbor_{col}" for col in features]]
    _, base_neighbor_preds = train_ridge_predictions(
        neighbor_splits["train"],
        neighbor_splits["val"],
        neighbor_splits["test"],
        neighbor_features,
    )
    _, residual_neighbor_preds = train_residual_svr(
        neighbor_splits["train"],
        neighbor_splits["val"],
        neighbor_splits["test"],
        neighbor_features,
        base_neighbor_preds,
        kind="rbf",
    )
    prediction_parts.extend(
        frame_predictions(
            neighbor_splits,
            residual_neighbor_preds,
            "rbf_svr_neighbor_residual_era5",
            "corr_top3_directed",
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
    summary = save_summary(overall)

    validation["output_dir"] = str(OUTPUT_DIR)
    validation["split_signature"] = split_signature(splits)
    validation["corr_edge_count"] = int(len(corr_edges))
    validation["random_edge_count"] = int(len(random_edges))
    validation["best_era5_baseline_rmse_cm"] = BEST_ERA5_BASELINE_RMSE_CM
    write_json(VALIDATION_JSON, validation)

    print("ERA5 SVR test metrics:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
