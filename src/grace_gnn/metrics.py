from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def pearson_r(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2 or np.std(y_true[mask]) == 0 or np.std(y_pred[mask]) == 0:
        return np.nan
    return float(np.corrcoef(y_true[mask], y_pred[mask])[0, 1])


def nse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return np.nan
    return float(1 - np.sum((y_true - y_pred) ** 2) / denom)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return {
        "rmse_cm": float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))),
        "mae_cm": float(mean_absolute_error(y_true, y_pred)),
        "pearson_r": pearson_r(y_true, y_pred),
        "nse_optional": nse(y_true, y_pred),
    }


def metrics_overall(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, graph_type, split), group in predictions.groupby(["model_name", "graph_type", "split"], dropna=False):
        rows.append({
            "model_name": model_name,
            "graph_type": graph_type,
            "split": split,
            **regression_metrics(group["observed_twsa_cm"], group["predicted_twsa_cm"]),
        })
    return pd.DataFrame(rows)


def metrics_by_basin(predictions: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    rows = []
    data = predictions[predictions["split"] == split]
    for (basin_id, basin_name, model_name, graph_type), group in data.groupby(
        ["basin_id", "basin_name", "model_name", "graph_type"], dropna=False
    ):
        observed_std = float(group["observed_twsa_cm"].std(ddof=0))
        values = regression_metrics(group["observed_twsa_cm"], group["predicted_twsa_cm"])
        rows.append({
            "basin_id": basin_id,
            "basin_name": basin_name,
            "model_name": model_name,
            "graph_type": graph_type,
            "model_graph": f"{model_name}|{graph_type}",
            **values,
            "test_std_cm": observed_std,
            "normalized_rmse": values["rmse_cm"] / observed_std if observed_std else np.nan,
        })
    return pd.DataFrame(rows)


def improvement_by_basin(
    metrics_basin: pd.DataFrame,
    real_key: str = "residual_neighbor_gnn|real_knn_undirected",
    random_key: str = "residual_neighbor_gnn|random_degree_matched",
) -> pd.DataFrame:
    pivot = metrics_basin.pivot_table(
        index=["basin_id", "basin_name"], columns="model_graph", values="rmse_cm", aggfunc="first"
    ).reset_index()
    out = pivot[["basin_id", "basin_name"]].copy()
    basin_only_key = "basin_only_nn|none"
    ridge_key = "ridge_ar|none"
    if {basin_only_key, real_key} <= set(pivot.columns):
        out["rmse_improvement_real_vs_basin_only_cm"] = pivot[basin_only_key] - pivot[real_key]
    if {ridge_key, real_key} <= set(pivot.columns):
        out["rmse_improvement_real_vs_ridge_cm"] = pivot[ridge_key] - pivot[real_key]
    if {random_key, real_key} <= set(pivot.columns):
        out["rmse_improvement_real_vs_random_cm"] = pivot[random_key] - pivot[real_key]
    graph_suffixes = {
        "real_knn_directed": "directed",
        "real_knn_undirected": "undirected",
        "real_knn_reversed": "reversed",
    }
    for graph_type, suffix in graph_suffixes.items():
        key = f"residual_neighbor_gnn|{graph_type}"
        if {basin_only_key, key} <= set(pivot.columns):
            out[f"rmse_improvement_{suffix}_vs_basin_only_cm"] = pivot[basin_only_key] - pivot[key]
        if {ridge_key, key} <= set(pivot.columns):
            out[f"rmse_improvement_{suffix}_vs_ridge_cm"] = pivot[ridge_key] - pivot[key]
        if {random_key, key} <= set(pivot.columns):
            out[f"rmse_improvement_{suffix}_vs_random_cm"] = pivot[random_key] - pivot[key]
    return out


def prediction_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, graph_type, split), group in predictions.groupby(["model_name", "graph_type", "split"], dropna=False):
        rows.append({
            "model_name": model_name,
            "graph_type": graph_type,
            "split": split,
            "n": len(group),
            "observed_min_cm": group["observed_twsa_cm"].min(),
            "observed_max_cm": group["observed_twsa_cm"].max(),
            "predicted_min_cm": group["predicted_twsa_cm"].min(),
            "predicted_max_cm": group["predicted_twsa_cm"].max(),
            "residual_min_cm": group["residual_cm"].min(),
            "residual_max_cm": group["residual_cm"].max(),
        })
    return pd.DataFrame(rows)
