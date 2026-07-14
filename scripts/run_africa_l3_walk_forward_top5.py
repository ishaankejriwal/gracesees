from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

import numpy as np
import pandas as pd

from grace_gnn.config import AFRICA_L3_MASK_ZIP_NAME, LAGGED_DATASET_CSV, RANDOM_SEED, REGION_OUTPUTS
from grace_gnn.evaluate import prediction_frame
from grace_gnn.features import feature_columns
from grace_gnn.graph import build_knn_edges_from_mask_zips, make_degree_matched_random_edges
from grace_gnn.metrics import regression_metrics
from grace_gnn.validation import validate_edges, validate_lagged_dataset
from scripts.run_africa_l3_extra_architectures import (
    add_neighbor_lag_features,
    correlation_topk_edges,
    train_residual_mlp,
    train_ridge_predictions,
)
from scripts.run_africa_l3_gnn_embeddings import (
    add_embeddings,
    train_gnn_embeddings,
    train_residual_tabular_models,
)


OUTPUT_DIR = REGION_OUTPUTS / "walk_forward_top5"
N_SPLITS = 5
VAL_MONTHS = 12
TEST_MONTHS = 24

TOP_MODELS = [
    ("ridge_neighbor_residual_mlp", "corr_top3_directed"),
    ("ridge_neighbor_residual_mlp", "random_degree_matched"),
    ("ridge_residual_mlp", "own_lags"),
    ("xgboost_gnn_embedding_residual", "corr_top3_directed"),
    ("random_forest_gnn_embedding_residual", "corr_top3_directed"),
]


def walk_forward_splits(df: pd.DataFrame) -> list[dict[str, object]]:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    dates = pd.Index(sorted(data["date"].dropna().unique()))
    required_dates = N_SPLITS * TEST_MONTHS + VAL_MONTHS + 1
    if len(dates) < required_dates:
        raise ValueError(
            f"Need at least {required_dates} unique dates for {N_SPLITS} folds, "
            f"{VAL_MONTHS} validation months, and {TEST_MONTHS} test months; found {len(dates)}."
        )

    first_test_idx = len(dates) - (N_SPLITS * TEST_MONTHS)
    folds = []
    for fold_idx in range(N_SPLITS):
        test_start = first_test_idx + fold_idx * TEST_MONTHS
        test_end = test_start + TEST_MONTHS
        val_start = test_start - VAL_MONTHS
        if val_start <= 0:
            raise ValueError("Fold configuration leaves no training dates before validation.")
        split_dates = {
            "train": dates[:val_start],
            "val": dates[val_start:test_start],
            "test": dates[test_start:test_end],
        }
        splits = {
            split_name: data[data["date"].isin(date_index)].copy()
            for split_name, date_index in split_dates.items()
        }
        validate_fold(fold_idx + 1, splits)
        folds.append({"fold": fold_idx + 1, "splits": splits, "dates": split_dates})
    return folds


def validate_fold(fold: int, splits: dict[str, pd.DataFrame]) -> None:
    for split_name, frame in splits.items():
        if frame.empty:
            raise ValueError(f"Fold {fold} has an empty {split_name} split.")
        validate_lagged_dataset(frame)

    train = splits["train"]
    val = splits["val"]
    test = splits["test"]
    if not (train["date"].max() < val["date"].min() <= val["date"].max() < test["date"].min()):
        raise ValueError(
            f"Fold {fold} is not chronological: "
            f"train max={train['date'].max()}, val={val['date'].min()}..{val['date'].max()}, "
            f"test min={test['date'].min()}."
        )

    row_sets = {
        split_name: set(zip(frame["basin_id"].astype(str), pd.to_datetime(frame["date"])))
        for split_name, frame in splits.items()
    }
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = row_sets[left] & row_sets[right]
        if overlap:
            raise ValueError(f"Fold {fold} has {len(overlap)} basin-months in both {left} and {right}.")


def build_real_knn_edges(lagged: pd.DataFrame) -> pd.DataFrame:
    mask_zip = ROOT / "masks" / AFRICA_L3_MASK_ZIP_NAME
    if not mask_zip.exists():
        raise FileNotFoundError(f"Missing L3 mask zip: {mask_zip}")
    basin_names = sorted(lagged["basin_name"].dropna().unique())
    return build_knn_edges_from_mask_zips(
        [mask_zip],
        basin_names,
        OUTPUT_DIR / "edges_real_knn_directed.csv",
        k=3,
        graph_type="real_knn_directed",
    )


def fold_prediction_parts(
    fold: int,
    splits: dict[str, pd.DataFrame],
    lag_cols: list[str],
    corr_edges: pd.DataFrame,
    random_edges: pd.DataFrame,
    basin_ids: list[str],
) -> list[pd.DataFrame]:
    parts = []

    _, base_preds = train_ridge_predictions(splits["train"], splits["val"], splits["test"], lag_cols)
    _, residual_preds = train_residual_mlp(
        splits["train"],
        splits["val"],
        splits["test"],
        lag_cols,
        base_preds,
        seed=RANDOM_SEED + 1000 * fold,
    )
    parts.append(prediction_frame(splits["test"], residual_preds["test"], "ridge_residual_mlp", "own_lags", "test"))

    for graph_type, edges in [
        ("corr_top3_directed", corr_edges),
        ("random_degree_matched", random_edges),
    ]:
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
        _, neighbor_residual_preds = train_residual_mlp(
            neighbor_splits["train"],
            neighbor_splits["val"],
            neighbor_splits["test"],
            neighbor_cols,
            neighbor_base_preds,
            seed=RANDOM_SEED + 1000 * fold + (1 if graph_type == "corr_top3_directed" else 2),
        )
        parts.append(
            prediction_frame(
                neighbor_splits["test"],
                neighbor_residual_preds["test"],
                "ridge_neighbor_residual_mlp",
                graph_type,
                "test",
            )
        )

    _, embedding_frames, emb_cols = train_gnn_embeddings(
        splits["train"],
        splits["val"],
        splits["test"],
        lag_cols,
        corr_edges,
        basin_ids,
        base_preds,
        embedding_dim=16,
        seed=RANDOM_SEED + 1000 * fold + 3,
    )
    enhanced_splits = {
        split_name: add_embeddings(frame, embedding_frames[split_name], emb_cols)
        for split_name, frame in splits.items()
    }
    embedding_preds = train_residual_tabular_models(enhanced_splits, [*lag_cols, *emb_cols], base_preds)
    for model_name in ["xgboost_gnn_embedding_residual", "random_forest_gnn_embedding_residual"]:
        parts.append(
            prediction_frame(
                enhanced_splits["test"],
                embedding_preds[model_name]["test"],
                model_name,
                "corr_top3_directed",
                "test",
            )
        )

    out = []
    wanted = set(TOP_MODELS)
    for frame in parts:
        key = (frame["model_name"].iloc[0], frame["graph_type"].iloc[0])
        if key in wanted:
            frame = frame.copy()
            frame.insert(0, "fold", fold)
            out.append(frame)
    return out


def metrics_by_fold(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, model_name, graph_type), group in predictions.groupby(["fold", "model_name", "graph_type"]):
        rows.append(
            {
                "fold": int(fold),
                "model_name": model_name,
                "graph_type": graph_type,
                "n": int(len(group)),
                "test_start": group["date"].min(),
                "test_end": group["date"].max(),
                **regression_metrics(group["observed_twsa_cm"], group["predicted_twsa_cm"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["fold", "rmse_cm", "model_name", "graph_type"]).reset_index(drop=True)


def rankings_by_fold(metrics: pd.DataFrame) -> pd.DataFrame:
    ranked = metrics.copy()
    ranked["rank_rmse"] = ranked.groupby("fold")["rmse_cm"].rank(method="min", ascending=True).astype(int)
    return ranked.sort_values(["fold", "rank_rmse", "model_name", "graph_type"]).reset_index(drop=True)


def metrics_summary(rankings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, graph_type), group in rankings.groupby(["model_name", "graph_type"]):
        rows.append(
            {
                "model_name": model_name,
                "graph_type": graph_type,
                "folds": int(group["fold"].nunique()),
                "mean_rmse_cm": float(group["rmse_cm"].mean()),
                "median_rmse_cm": float(group["rmse_cm"].median()),
                "std_rmse_cm": float(group["rmse_cm"].std(ddof=0)),
                "best_fold_rmse_cm": float(group["rmse_cm"].min()),
                "worst_fold_rmse_cm": float(group["rmse_cm"].max()),
                "best_rank": int(group["rank_rmse"].min()),
                "worst_rank": int(group["rank_rmse"].max()),
                "mean_rank": float(group["rank_rmse"].mean()),
                "rank1_folds": int((group["rank_rmse"] == 1).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_rmse_cm", "mean_rank", "model_name"]).reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lagged = pd.read_csv(LAGGED_DATASET_CSV, parse_dates=["date"])
    validate_lagged_dataset(lagged)

    lag_cols = feature_columns(lagged)
    basin_ids = sorted(lagged["basin_id"].astype(str).unique())
    basin_id_set = set(basin_ids)
    real_edges = build_real_knn_edges(lagged)
    validate_edges(real_edges, basin_id_set, graph_type="real_knn_directed")

    prediction_parts = []
    graph_audit_rows = []
    for fold_info in walk_forward_splits(lagged):
        fold = int(fold_info["fold"])
        splits = fold_info["splits"]
        print(
            f"Fold {fold}: train {splits['train']['date'].min().date()}..{splits['train']['date'].max().date()} "
            f"({splits['train']['date'].nunique()} months), "
            f"val {splits['val']['date'].min().date()}..{splits['val']['date'].max().date()}, "
            f"test {splits['test']['date'].min().date()}..{splits['test']['date'].max().date()}"
        )

        corr_edges = correlation_topk_edges(splits["train"], top_k=3)
        validate_edges(corr_edges, basin_id_set, graph_type="corr_top3_directed")
        corr_edges.to_csv(OUTPUT_DIR / f"fold_{fold:02d}_edges_corr_top3_directed.csv", index=False)

        random_edges = make_degree_matched_random_edges(real_edges, basin_ids, seed=RANDOM_SEED + fold)
        validate_edges(random_edges, basin_id_set, graph_type="random_degree_matched")
        random_edges.to_csv(OUTPUT_DIR / f"fold_{fold:02d}_edges_random_degree_matched.csv", index=False)

        graph_audit_rows.extend(
            [
                {
                    "fold": fold,
                    "graph_type": "corr_top3_directed",
                    "source": "fold_train_target_twsa_cm",
                    "train_start": splits["train"]["date"].min(),
                    "train_end": splits["train"]["date"].max(),
                    "n_edges": len(corr_edges),
                },
                {
                    "fold": fold,
                    "graph_type": "random_degree_matched",
                    "source": "mask_knn_out_degree_randomized",
                    "train_start": pd.NaT,
                    "train_end": pd.NaT,
                    "n_edges": len(random_edges),
                },
            ]
        )

        prediction_parts.extend(
            fold_prediction_parts(fold, splits, lag_cols, corr_edges, random_edges, basin_ids)
        )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions = predictions.sort_values(["fold", "model_name", "graph_type", "date", "basin_id"]).reset_index(drop=True)
    metrics = metrics_by_fold(predictions)
    rankings = rankings_by_fold(metrics)
    summary = metrics_summary(rankings)

    predictions.to_csv(OUTPUT_DIR / "predictions_walk_forward.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "metrics_by_fold.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    rankings.to_csv(OUTPUT_DIR / "rankings_by_fold.csv", index=False)
    pd.DataFrame(graph_audit_rows).to_csv(OUTPUT_DIR / "graph_audit.csv", index=False)

    print("\nWalk-forward RMSE summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
