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
from grace_gnn.metrics import metrics_overall
from grace_gnn.splits import chronological_fraction_split
from grace_gnn.validation import validate_edges, validate_lagged_dataset
from scripts.run_africa_l3_era5_one_month import era5_feature_columns
from scripts.run_africa_l3_extra_architectures import (
    add_neighbor_lag_features,
    train_residual_mlp,
    train_ridge_predictions,
)


def predictive_lag_topk_edges(train_df: pd.DataFrame, top_k: int, source_lag: str = "lag_1") -> pd.DataFrame:
    data = train_df.copy()
    data["basin_id"] = data["basin_id"].astype(str)
    target = data.pivot_table(index="date", columns="basin_id", values="target_twsa_cm", aggfunc="first")
    source = data.pivot_table(index="date", columns="basin_id", values=source_lag, aggfunc="first")
    basin_ids = sorted(set(target.columns) & set(source.columns))
    rows = []
    for dst in basin_ids:
        scores = []
        y = target[dst]
        for src in basin_ids:
            if src == dst:
                continue
            x = source[src]
            pair = pd.concat([x, y], axis=1).dropna()
            if len(pair) < 3:
                continue
            corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            if pd.notna(corr) and corr > 0:
                scores.append((src, float(corr)))
        for src, weight in sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]:
            rows.append({"src_basin_id": str(src), "dst_basin_id": str(dst), "weight": weight})
    edges = pd.DataFrame(rows)
    edges["graph_type"] = f"pred_lag1_top{top_k}_directed"
    return edges


def random_incoming_topk_edges(basin_ids: list[str], top_k: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for dst in basin_ids:
        candidates = [src for src in basin_ids if src != dst]
        picked = rng.choice(candidates, size=min(top_k, len(candidates)), replace=False)
        for src in picked:
            rows.append({"src_basin_id": str(src), "dst_basin_id": str(dst), "weight": 1.0})
    edges = pd.DataFrame(rows)
    edges["graph_type"] = f"random_incoming_top{top_k}"
    return edges


def frame_predictions(splits, preds, model_name: str, graph_type: str) -> list[pd.DataFrame]:
    return [
        prediction_frame(splits[split_name], preds[split_name], model_name, graph_type, split_name)
        for split_name in ["train", "val", "test"]
    ]


def train_neighbor_models(
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    edges: pd.DataFrame,
    graph_type: str,
    seed: int,
    model_suffix: str,
) -> list[pd.DataFrame]:
    neighbor_splits = {
        split_name: add_neighbor_lag_features(frame, edges, feature_cols)
        for split_name, frame in splits.items()
    }
    neighbor_cols = [*feature_cols, *[f"neighbor_{col}" for col in feature_cols]]
    _, base_preds = train_ridge_predictions(
        neighbor_splits["train"],
        neighbor_splits["val"],
        neighbor_splits["test"],
        neighbor_cols,
    )
    parts = frame_predictions(neighbor_splits, base_preds, f"ridge_neighbor_ar{model_suffix}", graph_type)
    _, residual_preds = train_residual_mlp(
        neighbor_splits["train"],
        neighbor_splits["val"],
        neighbor_splits["test"],
        neighbor_cols,
        base_preds,
        seed=seed,
    )
    parts.extend(
        frame_predictions(
            neighbor_splits,
            residual_preds,
            f"ridge_neighbor_residual_mlp{model_suffix}",
            graph_type,
        )
    )
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run predictive lag-correlation top-k neighbor sweep.")
    parser.add_argument("--lagged-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--era5", action="store_true")
    parser.add_argument("--ks", default="1,2,3,5,8,12")
    parser.add_argument("--random-seeds", default="42,43,44,45,46")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lagged = pd.read_csv(args.lagged_csv, parse_dates=["date"])
    lagged["basin_id"] = lagged["basin_id"].astype(str)
    validate_lagged_dataset(lagged)
    splits = chronological_fraction_split(lagged, TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION)
    for frame in splits.values():
        validate_lagged_dataset(frame)

    feature_cols = era5_feature_columns(lagged) if args.era5 else feature_columns(lagged)
    model_suffix = "_era5" if args.era5 else ""
    basin_ids = sorted(lagged["basin_id"].astype(str).unique())
    basin_id_set = set(basin_ids)
    ks = [int(value) for value in args.ks.split(",") if value.strip()]
    random_seeds = [int(value) for value in args.random_seeds.split(",") if value.strip()]

    prediction_parts = []
    edge_audit_rows = []
    for k in ks:
        pred_edges = predictive_lag_topk_edges(splits["train"], top_k=k, source_lag="lag_1")
        pred_graph = f"pred_lag1_top{k}_directed"
        validate_edges(pred_edges, basin_id_set, graph_type=pred_graph)
        pred_edges.to_csv(args.output_dir / f"edges_{pred_graph}.csv", index=False)
        edge_audit_rows.append({"graph_type": pred_graph, "k": k, "seed": np.nan, "n_edges": len(pred_edges)})
        prediction_parts.extend(
            train_neighbor_models(
                splits,
                feature_cols,
                pred_edges,
                pred_graph,
                seed=RANDOM_SEED + k,
                model_suffix=model_suffix,
            )
        )

        for seed in random_seeds:
            random_edges = random_incoming_topk_edges(basin_ids, top_k=k, seed=seed)
            random_graph = f"random_incoming_top{k}_seed{seed}"
            random_edges["graph_type"] = random_graph
            validate_edges(random_edges, basin_id_set, graph_type=random_graph)
            random_edges.to_csv(args.output_dir / f"edges_{random_graph}.csv", index=False)
            edge_audit_rows.append({"graph_type": random_graph, "k": k, "seed": seed, "n_edges": len(random_edges)})
            prediction_parts.extend(
                train_neighbor_models(
                    splits,
                    feature_cols,
                    random_edges,
                    random_graph,
                    seed=seed + 1000,
                    model_suffix=model_suffix,
                )
            )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    overall = metrics_overall(predictions).sort_values(["split", "rmse_cm"])
    overall.to_csv(args.output_dir / "metrics_overall.csv", index=False)
    pd.DataFrame(edge_audit_rows).to_csv(args.output_dir / "edge_audit.csv", index=False)

    test = overall[overall["split"].eq("test")].copy()
    test["k"] = test["graph_type"].str.extract(r"top(\d+)").astype(int)
    test["control"] = np.where(test["graph_type"].str.startswith("pred_"), "predictive_lag_corr", "random")
    summary_rows = []
    for (model_name, k), group in test.groupby(["model_name", "k"]):
        pred = group[group["control"].eq("predictive_lag_corr")]
        random = group[group["control"].eq("random")]
        if pred.empty:
            continue
        row = {
            "model_name": model_name,
            "k": int(k),
            "predictive_corr_rmse_cm": float(pred.iloc[0]["rmse_cm"]),
            "predictive_corr_mae_cm": float(pred.iloc[0]["mae_cm"]),
            "predictive_corr_pearson_r": float(pred.iloc[0]["pearson_r"]),
        }
        if not random.empty:
            row["random_mean_rmse_cm"] = float(random["rmse_cm"].mean())
            row["random_min_rmse_cm"] = float(random["rmse_cm"].min())
            row["random_max_rmse_cm"] = float(random["rmse_cm"].max())
            row["predictive_minus_random_mean_rmse_cm"] = row["predictive_corr_rmse_cm"] - row["random_mean_rmse_cm"]
            row["predictive_beats_all_random_seeds"] = bool(row["predictive_corr_rmse_cm"] < row["random_min_rmse_cm"])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["model_name", "predictive_corr_rmse_cm"])
    summary.to_csv(args.output_dir / "topk_summary.csv", index=False)

    print("Best test rows:")
    print(test.sort_values("rmse_cm").head(20).to_string(index=False))
    print("\nPredictive lag-correlation vs random summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
