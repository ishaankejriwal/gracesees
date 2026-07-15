from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

import numpy as np
import pandas as pd

from grace_gnn.config import AFRICA_L3_MASK_ZIP_NAME, RANDOM_SEED, TEST_FRACTION, TRAIN_FRACTION, VAL_FRACTION
from grace_gnn.evaluate import prediction_frame
from grace_gnn.features import feature_columns
from grace_gnn.graph import build_knn_edges_from_mask_zips
from grace_gnn.metrics import metrics_overall
from grace_gnn.splits import chronological_fraction_split
from grace_gnn.validation import validate_edges, validate_lagged_dataset
from scripts.run_africa_l3_era5_one_month import era5_feature_columns
from scripts.run_africa_l3_extra_architectures import (
    add_neighbor_lag_features,
    train_residual_mlp,
    train_ridge_predictions,
)


def incoming_knn_edges(lagged: pd.DataFrame, top_k: int, output_csv: Path) -> pd.DataFrame:
    mask_zip = ROOT / "masks" / AFRICA_L3_MASK_ZIP_NAME
    basin_names = sorted(lagged["basin_name"].dropna().unique())
    outgoing = build_knn_edges_from_mask_zips(
        [mask_zip],
        basin_names,
        output_csv,
        k=top_k,
        graph_type=f"geo_outgoing_top{top_k}",
    )
    incoming = outgoing.rename(columns={"src_basin_id": "dst_basin_id", "dst_basin_id": "src_basin_id"})
    incoming = incoming[["src_basin_id", "dst_basin_id"]].drop_duplicates()
    incoming["weight"] = 1.0
    incoming["graph_type"] = f"geo_incoming_top{top_k}"
    return incoming.reset_index(drop=True)


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
    parser = argparse.ArgumentParser(description="Run geographic centroid top-k neighbor sweep.")
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
        geo_edges = incoming_knn_edges(lagged, top_k=k, output_csv=args.output_dir / f"edges_geo_outgoing_top{k}.csv")
        geo_graph = f"geo_incoming_top{k}"
        validate_edges(geo_edges, basin_id_set, graph_type=geo_graph)
        geo_edges.to_csv(args.output_dir / f"edges_{geo_graph}.csv", index=False)
        edge_audit_rows.append({"graph_type": geo_graph, "k": k, "seed": np.nan, "n_edges": len(geo_edges)})
        prediction_parts.extend(
            train_neighbor_models(
                splits,
                feature_cols,
                geo_edges,
                geo_graph,
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
    test["control"] = np.where(test["graph_type"].str.startswith("geo_"), "geographic", "random")
    summary_rows = []
    for (model_name, k), group in test.groupby(["model_name", "k"]):
        geo = group[group["control"].eq("geographic")]
        random = group[group["control"].eq("random")]
        if geo.empty:
            continue
        row = {
            "model_name": model_name,
            "k": int(k),
            "geo_rmse_cm": float(geo.iloc[0]["rmse_cm"]),
            "geo_mae_cm": float(geo.iloc[0]["mae_cm"]),
            "geo_pearson_r": float(geo.iloc[0]["pearson_r"]),
        }
        if not random.empty:
            row["random_mean_rmse_cm"] = float(random["rmse_cm"].mean())
            row["random_min_rmse_cm"] = float(random["rmse_cm"].min())
            row["random_max_rmse_cm"] = float(random["rmse_cm"].max())
            row["geo_minus_random_mean_rmse_cm"] = row["geo_rmse_cm"] - row["random_mean_rmse_cm"]
            row["geo_beats_all_random_seeds"] = bool(row["geo_rmse_cm"] < row["random_min_rmse_cm"])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["model_name", "geo_rmse_cm"])
    summary.to_csv(args.output_dir / "topk_summary.csv", index=False)

    print("Best test rows:")
    print(test.sort_values("rmse_cm").head(20).to_string(index=False))
    print("\nGeographic vs random summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
