from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

import numpy as np
import pandas as pd

from grace_gnn.config import (
    BASIN_MONTH_CSV,
    DATA_PROCESSED,
    RANDOM_SEED,
    REGION_OUTPUTS,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VAL_FRACTION,
)
from grace_gnn.graph import make_degree_matched_random_edges, save_edges
from grace_gnn.metrics import regression_metrics
from grace_gnn.models import train_random_forest, train_ridge, train_xgboost
from grace_gnn.splits import chronological_fraction_split
from grace_gnn.validation import validate_basin_month, validate_edges, validate_lagged_dataset
from scripts.run_africa_l3_gnn_embeddings import (
    add_embeddings,
    train_gnn_embeddings,
    train_residual_tabular_models,
)
from scripts.run_africa_l3_extra_architectures import (
    correlation_topk_edges,
    train_residual_mlp,
    train_ridge_predictions,
)


HORIZONS = [1, 2, 3, 4, 5, 6]
LAGS = [0, 1, 2, 5, 11]
OUTPUT_DIR = REGION_OUTPUTS / "grace_only_horizons"
HORIZON_DATASET_CSV = DATA_PROCESSED / "grace_horizon_dataset_africa_l3_no_madagascar.csv"
OLD_LAGGED_DATASET_CSV = DATA_PROCESSED / "lagged_grace_dataset_africa_l3_no_madagascar.csv"


def feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("lag_")], key=lambda c: int(c.split("_")[1]))


def make_horizon_dataset(basin_month: pd.DataFrame, output_csv: Path | None = HORIZON_DATASET_CSV) -> pd.DataFrame:
    validate_basin_month(basin_month)
    data = basin_month.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.to_period("M").dt.to_timestamp()
    data["basin_id"] = data["basin_id"].astype(str)
    data = (
        data.groupby(["basin_id", "date"], as_index=False)
        .agg({"basin_name": "first", "twsa_cm": "mean"})
        .sort_values(["basin_id", "date"])
    )

    monthly_parts = []
    for basin_id, group in data.groupby("basin_id", sort=False):
        group = group.set_index("date").sort_index()
        full_index = pd.date_range(group.index.min(), group.index.max(), freq="MS")
        group = group.reindex(full_index)
        group.index.name = "issue_date"
        group["basin_id"] = basin_id
        group["basin_name"] = group["basin_name"].dropna().iloc[0] if group["basin_name"].notna().any() else basin_id
        grouped_twsa = group["twsa_cm"]
        for lag in LAGS:
            group[f"lag_{lag}"] = grouped_twsa.shift(lag)
        for horizon in HORIZONS:
            horizon_frame = group.reset_index()
            horizon_frame["horizon_months"] = horizon
            horizon_frame["target_date"] = horizon_frame["issue_date"] + pd.DateOffset(months=horizon)
            horizon_frame["target_twsa_cm"] = grouped_twsa.shift(-horizon).to_numpy()
            monthly_parts.append(horizon_frame)

    out = pd.concat(monthly_parts, ignore_index=True)
    keep_cols = [
        "issue_date",
        "target_date",
        "horizon_months",
        "basin_id",
        "basin_name",
        "target_twsa_cm",
        *[f"lag_{lag}" for lag in LAGS],
    ]
    out = out[keep_cols].dropna(subset=["target_twsa_cm", *[f"lag_{lag}" for lag in LAGS]]).copy()
    out["date"] = out["target_date"]
    out = out[
        [
            "date",
            "issue_date",
            "target_date",
            "horizon_months",
            "basin_id",
            "basin_name",
            "target_twsa_cm",
            *[f"lag_{lag}" for lag in LAGS],
        ]
    ].sort_values(["target_date", "horizon_months", "basin_id"]).reset_index(drop=True)
    validate_horizon_dataset(out)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
    return out


def validate_horizon_dataset(df: pd.DataFrame) -> None:
    required = {
        "date",
        "issue_date",
        "target_date",
        "horizon_months",
        "basin_id",
        "basin_name",
        "target_twsa_cm",
        *[f"lag_{lag}" for lag in LAGS],
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Horizon data is missing columns: {sorted(missing)}")
    data = df.copy()
    data["issue_date"] = pd.to_datetime(data["issue_date"])
    data["target_date"] = pd.to_datetime(data["target_date"])
    data["date"] = pd.to_datetime(data["date"])
    data["basin_id"] = data["basin_id"].astype(str)

    expected_target_dates = data.apply(
        lambda row: row.issue_date + pd.DateOffset(months=int(row.horizon_months)),
        axis=1,
    )
    if not expected_target_dates.equals(data["target_date"]):
        raise ValueError("Found rows where target_date != issue_date + horizon_months.")
    if not data["date"].equals(data["target_date"]):
        raise ValueError("The compatibility date column must equal target_date.")
    duplicates = data[data.duplicated(["basin_id", "issue_date", "horizon_months"], keep=False)]
    if not duplicates.empty:
        sample = duplicates[["basin_id", "issue_date", "horizon_months"]].head(5).to_dict("records")
        raise ValueError(f"Duplicate horizon rows found, examples: {sample}")
    if (data["target_date"] <= data["issue_date"]).any():
        raise ValueError("Every target_date must be after issue_date.")
    if set(data["horizon_months"].unique()) != set(HORIZONS):
        raise ValueError(f"Expected horizons {HORIZONS}, found {sorted(data['horizon_months'].unique())}.")


def validate_horizon_one_against_old(horizon_df: pd.DataFrame) -> None:
    if not OLD_LAGGED_DATASET_CSV.exists():
        print(f"Skipping horizon-1 sanity check; missing {OLD_LAGGED_DATASET_CSV}")
        return
    old = pd.read_csv(OLD_LAGGED_DATASET_CSV, parse_dates=["date"])
    validate_lagged_dataset(old)
    old["basin_id"] = old["basin_id"].astype(str)
    h1 = horizon_df[horizon_df["horizon_months"].eq(1)].copy()
    h1["basin_id"] = h1["basin_id"].astype(str)
    merged = h1.merge(
        old,
        on=["date", "basin_id"],
        how="inner",
        suffixes=("_horizon", "_old"),
    )
    if merged.empty:
        raise ValueError("Horizon-1 sanity check found no overlap with the old one-month dataset.")
    checks = {
        "target_twsa_cm_horizon": "target_twsa_cm_old",
        "lag_0": "lag_1_old",
        "lag_1_horizon": "lag_2_old",
        "lag_2_horizon": "lag_3",
        "lag_5": "lag_6",
        "lag_11": "lag_12",
    }
    for left, right in checks.items():
        if not np.allclose(merged[left].to_numpy(), merged[right].to_numpy(), equal_nan=False):
            raise ValueError(f"Horizon-1 sanity check failed for {left} vs {right}.")
    coverage = len(merged) / len(h1) if len(h1) else 0.0
    print(f"Horizon-1 sanity check passed on {len(merged):,} overlapping rows ({coverage:.1%} of horizon-1 rows).")


def add_neighbor_horizon_features(df: pd.DataFrame, edges: pd.DataFrame, lag_cols: list[str]) -> pd.DataFrame:
    data = df.copy()
    data["basin_id"] = data["basin_id"].astype(str)
    edges = edges.copy()
    edges["src_basin_id"] = edges["src_basin_id"].astype(str)
    edges["dst_basin_id"] = edges["dst_basin_id"].astype(str)
    if "weight" not in edges.columns:
        edges["weight"] = 1.0
    rows = data[["issue_date", "horizon_months", "basin_id", *lag_cols]].rename(columns={"basin_id": "src_basin_id"})
    joined = edges[["src_basin_id", "dst_basin_id", "weight"]].merge(rows, on="src_basin_id", how="left")
    for col in lag_cols:
        joined[f"{col}_weighted"] = joined[col] * joined["weight"]
    grouped = joined.groupby(["issue_date", "horizon_months", "dst_basin_id"], dropna=False)
    agg = grouped[[f"{col}_weighted" for col in lag_cols]].sum()
    denom = grouped["weight"].sum().replace(0, np.nan)
    neighbor = agg.div(denom, axis=0).reset_index()
    neighbor = neighbor.rename(
        columns={
            "dst_basin_id": "basin_id",
            **{f"{col}_weighted": f"neighbor_{col}" for col in lag_cols},
        }
    )
    out = data.merge(neighbor, on=["issue_date", "horizon_months", "basin_id"], how="left")
    for col in lag_cols:
        out[f"neighbor_{col}"] = out[f"neighbor_{col}"].fillna(out[col])
    return out


def horizon_prediction_frame(
    df: pd.DataFrame,
    preds: np.ndarray,
    model_name: str,
    graph_type: str,
    split: str,
) -> pd.DataFrame:
    out = df[
        ["date", "issue_date", "target_date", "horizon_months", "basin_id", "basin_name", "target_twsa_cm"]
    ].copy()
    out = out.rename(columns={"target_twsa_cm": "observed_twsa_cm"})
    out["model_name"] = model_name
    out["graph_type"] = graph_type
    out["split"] = split
    out["predicted_twsa_cm"] = preds
    out["residual_cm"] = out["observed_twsa_cm"] - out["predicted_twsa_cm"]
    return out[
        [
            "issue_date",
            "target_date",
            "horizon_months",
            "date",
            "basin_id",
            "basin_name",
            "model_name",
            "graph_type",
            "split",
            "observed_twsa_cm",
            "predicted_twsa_cm",
            "residual_cm",
        ]
    ]


def metrics_by_horizon(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["horizon_months", "model_name", "graph_type", "split"]
    for (horizon, model_name, graph_type, split), group in predictions.groupby(keys, dropna=False):
        rows.append(
            {
                "horizon_months": horizon,
                "model_name": model_name,
                "graph_type": graph_type,
                "split": split,
                "n": len(group),
                **regression_metrics(group["observed_twsa_cm"], group["predicted_twsa_cm"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon_months", "split", "rmse_cm"]).reset_index(drop=True)


def metrics_by_basin_horizon(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["horizon_months", "basin_id", "basin_name", "model_name", "graph_type", "split"]
    for (horizon, basin_id, basin_name, model_name, graph_type, split), group in predictions.groupby(keys, dropna=False):
        values = regression_metrics(group["observed_twsa_cm"], group["predicted_twsa_cm"])
        observed_std = float(group["observed_twsa_cm"].std(ddof=0))
        rows.append(
            {
                "horizon_months": horizon,
                "basin_id": basin_id,
                "basin_name": basin_name,
                "model_name": model_name,
                "graph_type": graph_type,
                "split": split,
                "n": len(group),
                **values,
                "test_std_cm": observed_std if split == "test" else np.nan,
                "normalized_rmse": values["rmse_cm"] / observed_std if split == "test" and observed_std else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["horizon_months", "basin_name", "split", "rmse_cm", "model_name", "graph_type"]
    ).reset_index(drop=True)


def train_for_horizon(horizon_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    horizon_data = horizon_df[horizon_df["horizon_months"].eq(horizon)].copy()
    horizon_splits = chronological_fraction_split(horizon_data, TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION)
    print(
        f"horizon {horizon} split:",
        "train",
        len(horizon_splits["train"]),
        horizon_splits["train"]["target_date"].min().date(),
        horizon_splits["train"]["target_date"].max().date(),
        "val",
        len(horizon_splits["val"]),
        horizon_splits["val"]["target_date"].min().date(),
        horizon_splits["val"]["target_date"].max().date(),
        "test",
        len(horizon_splits["test"]),
        horizon_splits["test"]["target_date"].min().date(),
        horizon_splits["test"]["target_date"].max().date(),
    )
    lag_cols = feature_columns(horizon_df)
    prediction_parts = []

    for split_name, frame in horizon_splits.items():
        prediction_parts.append(
            horizon_prediction_frame(frame, frame["lag_0"].to_numpy(), "horizon_persistence", "none", split_name)
        )

    _, ridge_preds = train_ridge(
        horizon_splits["train"],
        horizon_splits["val"],
        horizon_splits["test"],
        lag_cols,
    )
    for split_name in ["train", "val", "test"]:
        prediction_parts.append(
            horizon_prediction_frame(horizon_splits[split_name], ridge_preds[split_name], "ridge_ar", "none", split_name)
        )

    _, rf_preds = train_random_forest(
        horizon_splits["train"],
        horizon_splits["val"],
        horizon_splits["test"],
        lag_cols,
        seed=RANDOM_SEED,
    )
    for split_name in ["train", "val", "test"]:
        prediction_parts.append(
            horizon_prediction_frame(
                horizon_splits[split_name],
                rf_preds[split_name],
                "random_forest_ar",
                "none",
                split_name,
            )
        )

    _, xgb_preds = train_xgboost(
        horizon_splits["train"],
        horizon_splits["val"],
        horizon_splits["test"],
        lag_cols,
        seed=RANDOM_SEED,
    )
    for split_name in ["train", "val", "test"]:
        prediction_parts.append(
            horizon_prediction_frame(
                horizon_splits[split_name],
                xgb_preds[split_name],
                "xgboost_ar",
                "none",
                split_name,
            )
        )

    _, base_preds = train_ridge_predictions(
        horizon_splits["train"],
        horizon_splits["val"],
        horizon_splits["test"],
        lag_cols,
    )
    _, residual_preds = train_residual_mlp(
        horizon_splits["train"],
        horizon_splits["val"],
        horizon_splits["test"],
        lag_cols,
        base_preds,
        seed=RANDOM_SEED,
    )
    for split_name in ["train", "val", "test"]:
        prediction_parts.append(
            horizon_prediction_frame(
                horizon_splits[split_name],
                residual_preds[split_name],
                "ridge_residual_mlp",
                "own_lags",
                split_name,
            )
        )

    corr_edges = correlation_topk_edges(horizon_splits["train"], top_k=3)
    basin_ids = sorted(horizon_df["basin_id"].astype(str).unique())
    edge_variants = {
        "corr_top3_directed": corr_edges,
        "random_degree_matched": make_degree_matched_random_edges(
            corr_edges,
            basin_ids,
            seed=RANDOM_SEED + horizon,
        ),
    }
    for graph_type, edges in edge_variants.items():
        validate_edges(edges, set(basin_ids), graph_type=graph_type)
        save_edges(edges, OUTPUT_DIR / f"edges_h{horizon}_{graph_type}.csv")
        neighbor_splits = {
            split_name: add_neighbor_horizon_features(frame, edges, lag_cols)
            for split_name, frame in horizon_splits.items()
        }
        neighbor_cols = [*lag_cols, *[f"neighbor_{col}" for col in lag_cols]]
        _, neighbor_base_preds = train_ridge_predictions(
            neighbor_splits["train"],
            neighbor_splits["val"],
            neighbor_splits["test"],
            neighbor_cols,
        )
        for split_name in ["train", "val", "test"]:
            prediction_parts.append(
                horizon_prediction_frame(
                    neighbor_splits[split_name],
                    neighbor_base_preds[split_name],
                    "ridge_neighbor_ar",
                    graph_type,
                    split_name,
                )
            )
        _, neighbor_residual_preds = train_residual_mlp(
            neighbor_splits["train"],
            neighbor_splits["val"],
            neighbor_splits["test"],
            neighbor_cols,
            neighbor_base_preds,
            seed=RANDOM_SEED + 1,
        )
        for split_name in ["train", "val", "test"]:
            prediction_parts.append(
                horizon_prediction_frame(
                    neighbor_splits[split_name],
                    neighbor_residual_preds[split_name],
                    "ridge_neighbor_residual_mlp",
                    graph_type,
                    split_name,
                )
            )

    _, embedding_frames, emb_cols = train_gnn_embeddings(
        horizon_splits["train"],
        horizon_splits["val"],
        horizon_splits["test"],
        lag_cols,
        corr_edges,
        basin_ids,
        base_preds,
        embedding_dim=16,
        seed=RANDOM_SEED,
    )
    enhanced_splits = {
        split_name: add_embeddings(frame, embedding_frames[split_name], emb_cols)
        for split_name, frame in horizon_splits.items()
    }
    embedding_features = [*lag_cols, *emb_cols]
    embedding_preds = train_residual_tabular_models(enhanced_splits, embedding_features, base_preds)
    for model_name, preds in embedding_preds.items():
        for split_name in ["train", "val", "test"]:
            prediction_parts.append(
                horizon_prediction_frame(
                    enhanced_splits[split_name],
                    preds[split_name],
                    model_name,
                    "corr_top3_directed",
                    split_name,
                )
            )

    return pd.concat(prediction_parts, ignore_index=True)


def save_outputs(predictions: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = predictions.sort_values(
        ["horizon_months", "split", "model_name", "graph_type", "target_date", "basin_id"]
    ).reset_index(drop=True)
    predictions.to_csv(OUTPUT_DIR / "predictions_by_horizon.csv", index=False)

    metrics = metrics_by_horizon(predictions)
    metrics.to_csv(OUTPUT_DIR / "metrics_by_horizon.csv", index=False)

    basin_metrics = metrics_by_basin_horizon(predictions)
    basin_metrics.to_csv(OUTPUT_DIR / "metrics_by_basin_horizon.csv", index=False)

    test_metrics = metrics[metrics["split"].eq("test")].copy()
    test_metrics["rank"] = test_metrics.groupby("horizon_months")["rmse_cm"].rank(method="first")
    rankings = test_metrics.sort_values(["horizon_months", "rank"])
    rankings.to_csv(OUTPUT_DIR / "rankings_by_horizon.csv", index=False)

    summary = rankings[rankings["rank"].eq(1)].drop(columns=["rank"]).copy()
    summary.to_csv(OUTPUT_DIR / "metrics_summary_by_horizon.csv", index=False)

    print("Best test model by horizon:")
    print(
        summary[
            ["horizon_months", "model_name", "graph_type", "n", "rmse_cm", "mae_cm", "pearson_r", "nse_optional"]
        ].to_string(index=False)
    )


def main() -> None:
    basin_month = pd.read_csv(BASIN_MONTH_CSV, parse_dates=["date"])
    horizon_df = make_horizon_dataset(basin_month)
    validate_horizon_one_against_old(horizon_df)

    parts = []
    for horizon in HORIZONS:
        print(f"Training horizon {horizon} month(s)")
        parts.append(train_for_horizon(horizon_df, horizon))
    predictions = pd.concat(parts, ignore_index=True)
    save_outputs(predictions)


if __name__ == "__main__":
    main()
