from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

import numpy as np
import pandas as pd

from grace_gnn.config import DATA_PROCESSED, OUTPUTS, RANDOM_SEED, TEST_FRACTION, TRAIN_FRACTION, VAL_FRACTION
from grace_gnn.metrics import regression_metrics
from grace_gnn.splits import chronological_fraction_split
from grace_gnn.validation import validate_basin_month, validate_edges
from scripts.run_africa_l3_era5_one_month import TARGET_MONTH_ERA5_COLUMNS, era5_feature_columns
from scripts.run_africa_l3_extra_architectures import train_residual_mlp, train_ridge_predictions
from scripts.run_africa_l3_grace_only_horizons import (
    add_neighbor_horizon_features,
    horizon_prediction_frame,
    metrics_by_basin_horizon,
    metrics_by_horizon,
)
from scripts.run_correlation_topk_sweep import correlation_topk_edges
from scripts.run_geographic_topk_sweep import incoming_knn_edges


CSR_REGION = "africa_l3_no_madagascar_csr"
HORIZONS = [2, 3, 4, 5, 6]
LAGS = [0, 1, 2, 5, 11]
TOP_K = 2
RANDOM_SEEDS = [42, 43, 44, 45, 46]

BASIN_MONTH_CSV = DATA_PROCESSED / f"basin_month_grace_{CSR_REGION}.csv"
ERA5_BASIN_MONTH_CSV = DATA_PROCESSED / f"basin_month_era5_{CSR_REGION}.csv"
OUTPUT_DIR = OUTPUTS / CSR_REGION / "top5_horizons_2_6"
HORIZON_DATASET_CSV = DATA_PROCESSED / f"top5_horizon_dataset_{CSR_REGION}.csv"


def _month_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp()


def _issue_lag_columns() -> list[str]:
    return [f"lag_{lag}" for lag in LAGS]


def _era5_lag_columns() -> list[str]:
    cols = []
    for variable in ["tp", "ro", "evap"]:
        cols.extend([f"era5_{variable}_lag_{lag}" for lag in LAGS])
    return cols


def make_horizon_dataset(
    basin_month: pd.DataFrame,
    era5_basin_month: pd.DataFrame,
    output_csv: Path | None = HORIZON_DATASET_CSV,
) -> pd.DataFrame:
    validate_basin_month(basin_month)
    grace = basin_month.copy()
    grace["date"] = _month_start(grace["date"])
    grace["basin_id"] = grace["basin_id"].astype(str)
    grace = (
        grace.groupby(["basin_id", "date"], as_index=False)
        .agg({"basin_name": "first", "twsa_cm": "mean"})
        .sort_values(["basin_id", "date"])
    )

    era5 = era5_basin_month.copy()
    era5["date"] = _month_start(era5["date"])
    era5["basin_id"] = era5["basin_id"].astype(str)
    required_era5 = {"date", "basin_id", "era5_tp_mm", "era5_ro_mm", "era5_evap_mm"}
    missing_era5 = required_era5 - set(era5.columns)
    if missing_era5:
        raise ValueError(f"ERA5 basin-month data is missing columns: {sorted(missing_era5)}")
    era5 = (
        era5.groupby(["basin_id", "date"], as_index=False)
        .agg({"era5_tp_mm": "mean", "era5_ro_mm": "mean", "era5_evap_mm": "mean"})
        .sort_values(["basin_id", "date"])
    )

    monthly_parts = []
    for basin_id, group in grace.groupby("basin_id", sort=False):
        group = group.set_index("date").sort_index()
        full_index = pd.date_range(group.index.min(), group.index.max(), freq="MS")
        group = group.reindex(full_index)
        group.index.name = "issue_date"
        group["basin_id"] = basin_id
        group["basin_name"] = group["basin_name"].dropna().iloc[0] if group["basin_name"].notna().any() else basin_id

        era5_group = era5[era5["basin_id"].eq(basin_id)].set_index("date").sort_index()
        era5_group = era5_group.reindex(full_index)
        for source_col in ["era5_tp_mm", "era5_ro_mm", "era5_evap_mm"]:
            short_name = source_col.removeprefix("era5_").removesuffix("_mm")
            group[f"era5_{short_name}"] = era5_group[source_col].to_numpy()

        twsa = group["twsa_cm"]
        for lag in LAGS:
            group[f"lag_{lag}"] = twsa.shift(lag)
            for variable in ["tp", "ro", "evap"]:
                group[f"era5_{variable}_lag_{lag}"] = group[f"era5_{variable}"].shift(lag)

        for horizon in HORIZONS:
            horizon_frame = group.reset_index()
            horizon_frame["horizon_months"] = horizon
            horizon_frame["target_date"] = horizon_frame["issue_date"] + pd.DateOffset(months=horizon)
            horizon_frame["target_twsa_cm"] = twsa.shift(-horizon).to_numpy()
            monthly_parts.append(horizon_frame)

    feature_cols = [*_issue_lag_columns(), *_era5_lag_columns()]
    keep_cols = [
        "issue_date",
        "target_date",
        "horizon_months",
        "basin_id",
        "basin_name",
        "target_twsa_cm",
        *feature_cols,
    ]
    out = pd.concat(monthly_parts, ignore_index=True)
    out = out[keep_cols].dropna(subset=["target_twsa_cm", *feature_cols]).copy()
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
            *feature_cols,
        ]
    ].sort_values(["target_date", "horizon_months", "basin_id"])
    out = out.reset_index(drop=True)
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
        *_issue_lag_columns(),
        *_era5_lag_columns(),
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Horizon data is missing columns: {sorted(missing)}")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["issue_date"] = pd.to_datetime(data["issue_date"])
    data["target_date"] = pd.to_datetime(data["target_date"])
    data["basin_id"] = data["basin_id"].astype(str)

    expected_target = data.apply(
        lambda row: row.issue_date + pd.DateOffset(months=int(row.horizon_months)),
        axis=1,
    )
    if not expected_target.equals(data["target_date"]):
        raise ValueError("Found rows where target_date != issue_date + horizon_months.")
    if not data["date"].equals(data["target_date"]):
        raise ValueError("The compatibility date column must equal target_date.")
    if set(data["horizon_months"].unique()) != set(HORIZONS):
        raise ValueError(f"Expected horizons {HORIZONS}, found {sorted(data['horizon_months'].unique())}.")
    duplicates = data[data.duplicated(["basin_id", "issue_date", "horizon_months"], keep=False)]
    if not duplicates.empty:
        sample = duplicates[["basin_id", "issue_date", "horizon_months"]].head(5).to_dict("records")
        raise ValueError(f"Duplicate horizon rows found, examples: {sample}")
    basin_counts = data.groupby("horizon_months")["basin_id"].nunique()
    bad_counts = basin_counts[basin_counts.ne(37)]
    if not bad_counts.empty:
        raise ValueError(f"Expected 37 basins for every horizon, found {bad_counts.to_dict()}.")
    feature_cols = [*_issue_lag_columns(), *_era5_lag_columns()]
    if data[feature_cols].isna().any().any():
        offenders = data[feature_cols].isna().sum()
        raise ValueError(f"Feature matrix contains missing values: {offenders[offenders > 0].to_dict()}")
    forbidden = sorted(TARGET_MONTH_ERA5_COLUMNS & set(feature_cols))
    if forbidden:
        raise ValueError(f"Target-month ERA5 columns leaked into model features: {forbidden}")


def predictive_issue_topk_edges(train_df: pd.DataFrame, top_k: int = TOP_K) -> pd.DataFrame:
    data = train_df.copy()
    data["basin_id"] = data["basin_id"].astype(str)
    target = data.pivot_table(index="target_date", columns="basin_id", values="target_twsa_cm", aggfunc="first")
    source = data.pivot_table(index="target_date", columns="basin_id", values="lag_0", aggfunc="first")
    basin_ids = sorted(set(target.columns) & set(source.columns))
    rows = []
    for dst in basin_ids:
        scores = []
        y = target[dst]
        for src in basin_ids:
            if src == dst:
                continue
            pair = pd.concat([source[src], y], axis=1).dropna()
            if len(pair) < 3:
                continue
            corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            if pd.notna(corr) and corr > 0:
                scores.append((src, float(corr)))
        for src, weight in sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]:
            rows.append({"src_basin_id": str(src), "dst_basin_id": str(dst), "weight": weight})
    edges = pd.DataFrame(rows)
    edges["graph_type"] = f"pred_issue_top{top_k}_directed"
    return edges


def matched_random_incoming_edges(edges: pd.DataFrame, basin_ids: list[str], seed: int, graph_type: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    edge_data = edges.copy()
    edge_data["dst_basin_id"] = edge_data["dst_basin_id"].astype(str)
    degrees = edge_data.groupby("dst_basin_id").size().to_dict()
    rows = []
    for dst in basin_ids:
        degree = int(degrees.get(str(dst), 0))
        candidates = [src for src in basin_ids if src != str(dst)]
        if degree > len(candidates):
            raise ValueError(f"Cannot draw {degree} non-self random incoming edges for basin {dst}.")
        for src in rng.choice(candidates, size=degree, replace=False):
            rows.append({"src_basin_id": str(src), "dst_basin_id": str(dst), "weight": 1.0})
    out = pd.DataFrame(rows)
    out["graph_type"] = graph_type
    return out


def edge_audit_row(edges: pd.DataFrame, horizon: int, dataset: str, family: str, graph_type: str, seed: int | None) -> dict:
    incoming = edges.groupby("dst_basin_id").size()
    return {
        "horizon_months": horizon,
        "dataset": dataset,
        "neighbor_family": family,
        "graph_type": graph_type,
        "seed": seed,
        "n_edges": int(len(edges)),
        "n_target_basins": int(incoming.size),
        "min_incoming_degree": int(incoming.min()) if len(incoming) else 0,
        "max_incoming_degree": int(incoming.max()) if len(incoming) else 0,
        "mean_incoming_degree": float(incoming.mean()) if len(incoming) else 0.0,
    }


def train_own_lag_residual(
    splits: dict[str, pd.DataFrame],
    features: list[str],
    model_name: str,
    graph_type: str,
    seed: int,
) -> list[pd.DataFrame]:
    _, base_preds = train_ridge_predictions(splits["train"], splits["val"], splits["test"], features)
    _, residual_preds = train_residual_mlp(
        splits["train"],
        splits["val"],
        splits["test"],
        features,
        base_preds,
        seed=seed,
    )
    return [
        horizon_prediction_frame(splits[split_name], residual_preds[split_name], model_name, graph_type, split_name)
        for split_name in ["train", "val", "test"]
    ]


def train_neighbor_residual(
    splits: dict[str, pd.DataFrame],
    features: list[str],
    edges: pd.DataFrame,
    model_name: str,
    graph_type: str,
    seed: int,
) -> list[pd.DataFrame]:
    neighbor_splits = {
        split_name: add_neighbor_horizon_features(frame, edges, features)
        for split_name, frame in splits.items()
    }
    neighbor_features = [*features, *[f"neighbor_{col}" for col in features]]
    missing = {
        split_name: sorted(set(neighbor_features) - set(frame.columns))
        for split_name, frame in neighbor_splits.items()
    }
    missing = {split_name: cols for split_name, cols in missing.items() if cols}
    if missing:
        raise ValueError(f"Missing neighbor feature columns: {missing}")

    _, base_preds = train_ridge_predictions(
        neighbor_splits["train"],
        neighbor_splits["val"],
        neighbor_splits["test"],
        neighbor_features,
    )
    _, residual_preds = train_residual_mlp(
        neighbor_splits["train"],
        neighbor_splits["val"],
        neighbor_splits["test"],
        neighbor_features,
        base_preds,
        seed=seed,
    )
    return [
        horizon_prediction_frame(
            neighbor_splits[split_name],
            residual_preds[split_name],
            model_name,
            graph_type,
            split_name,
        )
        for split_name in ["train", "val", "test"]
    ]


def split_signature(splits: dict[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    return {
        split_name: {
            "rows": int(len(frame)),
            "min_target_date": str(frame["target_date"].min().date()),
            "max_target_date": str(frame["target_date"].max().date()),
            "target_dates": [str(pd.Timestamp(date).date()) for date in sorted(frame["target_date"].unique())],
        }
        for split_name, frame in splits.items()
    }


def train_for_horizon(horizon_df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[dict], dict]:
    data = horizon_df[horizon_df["horizon_months"].eq(horizon)].copy()
    splits = chronological_fraction_split(data, TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION)
    grace_features = _issue_lag_columns()
    era5_features = era5_feature_columns(data)
    expected_era5_features = [*grace_features, *_era5_lag_columns()]
    if set(era5_features) != set(expected_era5_features):
        raise ValueError(f"Unexpected ERA5 feature columns: {era5_features}")

    basin_ids = sorted(data["basin_id"].astype(str).unique())
    basin_id_set = set(basin_ids)
    prediction_parts = []
    edge_rows = []

    print(
        f"horizon {horizon} split:",
        "train",
        len(splits["train"]),
        splits["train"]["target_date"].min().date(),
        splits["train"]["target_date"].max().date(),
        "val",
        len(splits["val"]),
        splits["val"]["target_date"].min().date(),
        splits["val"]["target_date"].max().date(),
        "test",
        len(splits["test"]),
        splits["test"]["target_date"].min().date(),
        splits["test"]["target_date"].max().date(),
    )

    prediction_parts.extend(
        train_own_lag_residual(
            splits,
            grace_features,
            "ridge_residual_mlp",
            "own_lags",
            seed=RANDOM_SEED + horizon,
        )
    )
    prediction_parts.extend(
        train_own_lag_residual(
            splits,
            era5_features,
            "ridge_residual_mlp_era5",
            "own_lags",
            seed=RANDOM_SEED + 100 + horizon,
        )
    )

    corr_edges = correlation_topk_edges(splits["train"], top_k=TOP_K)
    corr_edges["graph_type"] = "corr_top2_directed"
    pred_edges = predictive_issue_topk_edges(splits["train"], top_k=TOP_K)
    geo_edges = incoming_knn_edges(data, top_k=TOP_K, output_csv=OUTPUT_DIR / f"edges_h{horizon}_geo_outgoing_top2.csv")
    geo_edges["graph_type"] = "geo_incoming_top2"

    selected_runs = [
        ("grace", "corr", "ridge_neighbor_residual_mlp", "corr_top2_directed", grace_features, corr_edges),
        ("grace", "pred_issue", "ridge_neighbor_residual_mlp", "pred_issue_top2_directed", grace_features, pred_edges),
        ("grace", "geo", "ridge_neighbor_residual_mlp", "geo_incoming_top2", grace_features, geo_edges),
        ("era5", "corr", "ridge_neighbor_residual_mlp_era5", "corr_top2_directed", era5_features, corr_edges),
        (
            "era5",
            "pred_issue",
            "ridge_neighbor_residual_mlp_era5",
            "pred_issue_top2_directed",
            era5_features,
            pred_edges,
        ),
    ]

    for dataset, family, model_name, graph_type, features, edges in selected_runs:
        validate_edges(edges, basin_id_set, graph_type=graph_type)
        edge_path = OUTPUT_DIR / f"edges_h{horizon}_{dataset}_{graph_type}.csv"
        edge_path.parent.mkdir(parents=True, exist_ok=True)
        edges.to_csv(edge_path, index=False)
        edge_rows.append(edge_audit_row(edges, horizon, dataset, family, graph_type, None))
        prediction_parts.extend(
            train_neighbor_residual(
                splits,
                features,
                edges,
                model_name,
                graph_type,
                seed=RANDOM_SEED + horizon * 10 + len(edge_rows),
            )
        )

        for seed in RANDOM_SEEDS:
            random_graph = f"random_incoming_top2_seed{seed}_for_{graph_type}"
            random_edges = matched_random_incoming_edges(edges, basin_ids, seed=seed, graph_type=random_graph)
            validate_edges(random_edges, basin_id_set, graph_type=random_graph)
            random_edges.to_csv(OUTPUT_DIR / f"edges_h{horizon}_{dataset}_{random_graph}.csv", index=False)
            edge_rows.append(edge_audit_row(random_edges, horizon, dataset, family, random_graph, seed))
            prediction_parts.extend(
                train_neighbor_residual(
                    splits,
                    features,
                    random_edges,
                    model_name,
                    random_graph,
                    seed=seed + horizon * 100 + (0 if dataset == "grace" else 1000),
                )
            )

    return pd.concat(prediction_parts, ignore_index=True), edge_rows, split_signature(splits)


def summarize_top5(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["split"].eq("test")].copy()
    rows = []
    selected = test[~test["graph_type"].str.startswith("random_incoming_top2_seed")].copy()
    selected = selected[selected["model_name"].str.startswith("ridge_neighbor_residual_mlp")]
    for _, row in selected.iterrows():
        graph_type = row["graph_type"]
        random = test[
            test["model_name"].eq(row["model_name"])
            & test["horizon_months"].eq(row["horizon_months"])
            & test["graph_type"].str.endswith(f"_for_{graph_type}")
        ]
        out = row.to_dict()
        out["random_mean_rmse_cm"] = float(random["rmse_cm"].mean()) if not random.empty else np.nan
        out["random_min_rmse_cm"] = float(random["rmse_cm"].min()) if not random.empty else np.nan
        out["random_max_rmse_cm"] = float(random["rmse_cm"].max()) if not random.empty else np.nan
        out["selected_minus_random_mean_rmse_cm"] = (
            float(row["rmse_cm"] - out["random_mean_rmse_cm"]) if not random.empty else np.nan
        )
        out["beats_all_random_seeds"] = bool(row["rmse_cm"] < random["rmse_cm"].min()) if not random.empty else False
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["horizon_months", "model_name", "graph_type"]).reset_index(drop=True)


def summarize_random_controls(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["split"].eq("test")].copy()
    random = test[test["graph_type"].str.startswith("random_incoming_top2_seed")].copy()
    random["selected_graph_type"] = random["graph_type"].str.split("_for_", n=1).str[1]
    rows = []
    for (horizon, model_name, selected_graph), group in random.groupby(
        ["horizon_months", "model_name", "selected_graph_type"], dropna=False
    ):
        rows.append(
            {
                "horizon_months": horizon,
                "model_name": model_name,
                "selected_graph_type": selected_graph,
                "n_random_seeds": int(len(group)),
                "random_mean_rmse_cm": float(group["rmse_cm"].mean()),
                "random_min_rmse_cm": float(group["rmse_cm"].min()),
                "random_max_rmse_cm": float(group["rmse_cm"].max()),
                "random_mean_mae_cm": float(group["mae_cm"].mean()),
                "random_mean_pearson_r": float(group["pearson_r"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon_months", "model_name", "selected_graph_type"]).reset_index(drop=True)


def validate_outputs(predictions: pd.DataFrame, edge_audit: pd.DataFrame) -> dict[str, object]:
    expected_splits = {"train", "val", "test"}
    missing_split_rows = []
    for (horizon, model_name, graph_type), group in predictions.groupby(["horizon_months", "model_name", "graph_type"]):
        splits = set(group["split"])
        if splits != expected_splits:
            missing_split_rows.append(
                {
                    "horizon_months": int(horizon),
                    "model_name": model_name,
                    "graph_type": graph_type,
                    "splits": sorted(splits),
                }
            )
    if missing_split_rows:
        raise ValueError(f"Some model/horizon outputs are missing splits: {missing_split_rows[:5]}")

    selected_edges = edge_audit[~edge_audit["graph_type"].str.startswith("random_incoming_top2_seed")]
    bad_selected = selected_edges[
        selected_edges["n_edges"].ne(74)
        | selected_edges["min_incoming_degree"].ne(2)
        | selected_edges["max_incoming_degree"].ne(2)
    ]
    if not bad_selected.empty:
        raise ValueError(f"Selected top-2 graphs do not all have two incoming edges: {bad_selected.to_dict('records')}")

    random_edges = edge_audit[edge_audit["graph_type"].str.startswith("random_incoming_top2_seed")]
    bad_random = random_edges[
        random_edges["n_edges"].ne(74)
        | random_edges["min_incoming_degree"].ne(2)
        | random_edges["max_incoming_degree"].ne(2)
    ]
    if not bad_random.empty:
        raise ValueError(f"Random top-2 controls do not match incoming degree: {bad_random.to_dict('records')[:5]}")

    return {
        "prediction_rows": int(len(predictions)),
        "model_graph_horizon_count": int(
            predictions[["horizon_months", "model_name", "graph_type"]].drop_duplicates().shape[0]
        ),
        "all_models_have_train_val_test": True,
        "selected_graphs_have_two_incoming_edges_per_basin": True,
        "random_controls_have_two_incoming_edges_per_basin": True,
    }


def save_outputs(predictions: pd.DataFrame, edge_rows: list[dict], split_signatures: dict[int, dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = predictions.sort_values(
        ["horizon_months", "split", "model_name", "graph_type", "target_date", "basin_id"]
    ).reset_index(drop=True)
    predictions.to_csv(OUTPUT_DIR / "predictions_by_horizon.csv", index=False)

    metrics = metrics_by_horizon(predictions)
    metrics.to_csv(OUTPUT_DIR / "metrics_by_horizon.csv", index=False)

    basin_metrics = metrics_by_basin_horizon(predictions)
    basin_metrics.to_csv(OUTPUT_DIR / "metrics_by_basin_horizon.csv", index=False)
    test_basin = basin_metrics[basin_metrics["split"].eq("test")]
    if test_basin["normalized_rmse"].isna().any():
        raise ValueError("Basin-level test metrics contain missing normalized RMSE values.")

    top5_summary = summarize_top5(metrics)
    top5_summary.to_csv(OUTPUT_DIR / "top5_summary_by_horizon.csv", index=False)
    random_summary = summarize_random_controls(metrics)
    random_summary.to_csv(OUTPUT_DIR / "random_control_summary_by_horizon.csv", index=False)

    edge_audit = pd.DataFrame(edge_rows).sort_values(["horizon_months", "dataset", "neighbor_family", "graph_type"])
    edge_audit.to_csv(OUTPUT_DIR / "edge_audit.csv", index=False)

    validation = validate_outputs(predictions, edge_audit)
    validation.update(
        {
            "csr_region": CSR_REGION,
            "horizons": HORIZONS,
            "basins_per_horizon": 37,
            "top_k": TOP_K,
            "random_seeds": RANDOM_SEEDS,
            "grace_features": _issue_lag_columns(),
            "era5_features": [
                *_issue_lag_columns(),
                *sorted(_era5_lag_columns(), key=lambda col: (col.split("_lag_", 1)[0], int(col.rsplit("_", 1)[1]))),
            ],
            "target_definition": "target_twsa_cm at issue_date + horizon_months",
            "split_rule": f"chronological_fraction_split by target_date: {TRAIN_FRACTION}/{VAL_FRACTION}/{TEST_FRACTION}",
            "split_signatures": split_signatures,
            "target_month_era5_columns_in_features": [],
            "output_dir": str(OUTPUT_DIR),
        }
    )
    with (OUTPUT_DIR / "run_validation.json").open("w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)

    print("Top-5 horizon summary:")
    cols = [
        "horizon_months",
        "model_name",
        "graph_type",
        "n",
        "rmse_cm",
        "mae_cm",
        "pearson_r",
        "beats_all_random_seeds",
    ]
    print(top5_summary[cols].to_string(index=False))


def main() -> None:
    basin_month = pd.read_csv(BASIN_MONTH_CSV, parse_dates=["date"])
    era5_basin_month = pd.read_csv(ERA5_BASIN_MONTH_CSV, parse_dates=["date"])
    horizon_df = make_horizon_dataset(basin_month, era5_basin_month)

    parts = []
    edge_rows: list[dict] = []
    split_signatures = {}
    for horizon in HORIZONS:
        print(f"Training CSR top-5 horizon {horizon}")
        predictions, horizon_edge_rows, signature = train_for_horizon(horizon_df, horizon)
        parts.append(predictions)
        edge_rows.extend(horizon_edge_rows)
        split_signatures[horizon] = signature

    save_outputs(pd.concat(parts, ignore_index=True), edge_rows, split_signatures)


if __name__ == "__main__":
    main()
