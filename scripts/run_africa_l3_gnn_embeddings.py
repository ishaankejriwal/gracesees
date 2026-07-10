from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import numpy as np
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
from grace_gnn.graph import normalized_adjacency, save_edges
from grace_gnn.metrics import (
    improvement_by_basin,
    metrics_by_basin,
    metrics_overall,
    prediction_diagnostics,
)
from grace_gnn.models import set_seeds
from grace_gnn.splits import chronological_fraction_split
from grace_gnn.validation import validate_edges, validate_lagged_dataset


EMBEDDING_MODEL_NAMES = {
    "ridge_gnn_embedding_residual",
    "random_forest_gnn_embedding_residual",
    "xgboost_gnn_embedding_residual",
}


def train_ridge_baseline(train_df, val_df, test_df, feature_cols):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train_df[feature_cols], train_df["target_twsa_cm"])

    def predict(frame):
        return model.predict(frame[feature_cols]) if len(frame) else np.array([])

    return model, {"train": predict(train_df), "val": predict(val_df), "test": predict(test_df)}


def correlation_topk_edges(train_df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    train = train_df.copy()
    train["basin_id"] = train["basin_id"].astype(str)
    pivot = train.pivot_table(index="date", columns="basin_id", values="target_twsa_cm", aggfunc="first")
    corr = pivot.corr()
    rows = []
    for dst in corr.columns:
        neighbors = corr[dst].drop(labels=[dst], errors="ignore").dropna().sort_values(ascending=False)
        neighbors = neighbors[neighbors > 0].head(top_k)
        for src, weight in neighbors.items():
            rows.append({"src_basin_id": str(src), "dst_basin_id": str(dst), "weight": float(weight)})
    edges = pd.DataFrame(rows)
    edges["graph_type"] = f"corr_top{top_k}_directed"
    return edges


def _graph_snapshots(df: pd.DataFrame, basin_ids: list[str], feature_cols: list[str], residual_col: str):
    snapshots = []
    data = df.copy()
    data["basin_id"] = data["basin_id"].astype(str)
    for date, group in data.groupby("date", sort=True):
        group = group.set_index("basin_id").reindex(basin_ids)
        feature_mask = group[feature_cols].notna().all(axis=1).to_numpy()
        target_mask = group[residual_col].notna().to_numpy()
        mask = feature_mask & target_mask
        if not mask.any():
            continue
        x = group[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        y = group[residual_col].fillna(0.0).to_numpy(dtype=np.float32)
        meta = group.reset_index()[["date", "basin_id"]].iloc[np.where(mask)[0]].copy()
        snapshots.append({"date": date, "x": x, "y": y, "mask": mask, "meta": meta})
    return snapshots


def train_gnn_embeddings(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    edges: pd.DataFrame,
    basin_ids: list[str],
    base_preds: dict[str, np.ndarray],
    embedding_dim: int = 16,
    seed: int = RANDOM_SEED,
):
    import torch
    from torch import nn
    from sklearn.preprocessing import StandardScaler

    set_seeds(seed)
    split_frames = {
        "train": train_df.copy(),
        "val": val_df.copy(),
        "test": test_df.copy(),
    }
    for split_name, frame in split_frames.items():
        frame["ridge_residual_cm"] = frame["target_twsa_cm"].to_numpy() - base_preds[split_name]

    x_scaler = StandardScaler().fit(split_frames["train"][feature_cols])
    y_scaler = StandardScaler().fit(split_frames["train"][["ridge_residual_cm"]])

    def scaled(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out[feature_cols] = x_scaler.transform(out[feature_cols])
        out["ridge_residual_scaled"] = y_scaler.transform(out[["ridge_residual_cm"]]).ravel()
        return out

    train_snaps = _graph_snapshots(scaled(split_frames["train"]), basin_ids, feature_cols, "ridge_residual_scaled")
    val_snaps = _graph_snapshots(scaled(split_frames["val"]), basin_ids, feature_cols, "ridge_residual_scaled")
    test_snaps = _graph_snapshots(scaled(split_frames["test"]), basin_ids, feature_cols, "ridge_residual_scaled")
    if not train_snaps:
        raise ValueError("No train graph snapshots available for GNN embedding training.")

    a_norm = normalized_adjacency(edges, basin_ids)

    class GNNEmbeddingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layer = nn.Linear(len(feature_cols), 32)
            self.embedding_layer = nn.Linear(32, embedding_dim)
            self.head = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(embedding_dim, 1),
            )

        def encode(self, x):
            h = torch.relu(self.input_layer(a_norm @ x))
            return torch.relu(self.embedding_layer(a_norm @ h))

        def forward(self, x):
            return self.head(self.encode(x)).squeeze(-1)

    model = GNNEmbeddingModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    best_state = None
    best_val = float("inf")
    stale = 0
    for _ in range(300):
        model.train()
        for snap in train_snaps:
            x = torch.tensor(snap["x"], dtype=torch.float32)
            y = torch.tensor(snap["y"], dtype=torch.float32)
            mask = torch.tensor(snap["mask"], dtype=torch.bool)
            opt.zero_grad()
            loss = loss_fn(model(x)[mask], y[mask])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            eval_snaps = val_snaps or train_snaps
            losses = []
            for snap in eval_snaps:
                x = torch.tensor(snap["x"], dtype=torch.float32)
                y = torch.tensor(snap["y"], dtype=torch.float32)
                mask = torch.tensor(snap["mask"], dtype=torch.bool)
                losses.append(loss_fn(model(x)[mask], y[mask]).item())
            val_loss = float(np.mean(losses))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 40:
            break
    if best_state:
        model.load_state_dict(best_state)

    emb_cols = [f"gnn_emb_{i}" for i in range(embedding_dim)]

    def embeddings(snaps) -> pd.DataFrame:
        rows = []
        model.eval()
        with torch.no_grad():
            for snap in snaps:
                x = torch.tensor(snap["x"], dtype=torch.float32)
                emb = model.encode(x).numpy()
                meta = snap["meta"].copy()
                meta[emb_cols] = emb[snap["mask"]]
                rows.append(meta)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "basin_id", *emb_cols])

    return model, {
        "train": embeddings(train_snaps),
        "val": embeddings(val_snaps),
        "test": embeddings(test_snaps),
    }, emb_cols


def add_embeddings(frame: pd.DataFrame, embeddings: pd.DataFrame, emb_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    out["basin_id"] = out["basin_id"].astype(str)
    out["__row_order"] = np.arange(len(out))
    emb = embeddings.copy()
    emb["date"] = pd.to_datetime(emb["date"])
    emb["basin_id"] = emb["basin_id"].astype(str)
    merged = out.merge(emb, on=["date", "basin_id"], how="left", validate="one_to_one")
    if merged[emb_cols].isna().any().any():
        missing = int(merged[emb_cols].isna().any(axis=1).sum())
        raise ValueError(f"Missing GNN embeddings for {missing} rows.")
    return merged.sort_values("__row_order").drop(columns="__row_order").reset_index(drop=True)


def train_residual_tabular_models(splits, feature_cols, base_preds):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    train_residual = splits["train"]["target_twsa_cm"].to_numpy() - base_preds["train"]
    models = {
        "ridge_gnn_embedding_residual": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest_gnn_embedding_residual": RandomForestRegressor(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "xgboost_gnn_embedding_residual": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=250,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            n_jobs=1,
        ),
    }
    predictions = {}
    for model_name, model in models.items():
        model.fit(splits["train"][feature_cols], train_residual)
        split_preds = {}
        for split_name, frame in splits.items():
            residual_pred = model.predict(frame[feature_cols]) if len(frame) else np.array([])
            split_preds[split_name] = base_preds[split_name] + residual_pred
        predictions[model_name] = split_preds
    return predictions


def frame_predictions(splits, preds, model_name: str, graph_type: str) -> list[pd.DataFrame]:
    return [
        prediction_frame(splits[split_name], preds[split_name], model_name, graph_type, split_name)
        for split_name in ["train", "val", "test"]
    ]


def write_csv_or_warn(df: pd.DataFrame, path: Path, **kwargs) -> None:
    try:
        df.to_csv(path, **kwargs)
    except PermissionError:
        print(f"Warning: could not write {path}; close the file if it is open and rerun.")


def main() -> None:
    lagged = pd.read_csv(LAGGED_DATASET_CSV, parse_dates=["date"])
    validate_lagged_dataset(lagged)
    splits = chronological_fraction_split(lagged, TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION)
    for frame in splits.values():
        validate_lagged_dataset(frame)

    lag_cols = feature_columns(lagged)
    basin_ids = sorted(lagged["basin_id"].astype(str).unique())
    _, base_preds = train_ridge_baseline(splits["train"], splits["val"], splits["test"], lag_cols)

    edges = correlation_topk_edges(splits["train"], top_k=3)
    validate_edges(edges, set(basin_ids), graph_type="corr_top3_directed")
    save_edges(edges, REGION_OUTPUTS / "edges_corr_top3_directed.csv")

    _, embedding_frames, emb_cols = train_gnn_embeddings(
        splits["train"],
        splits["val"],
        splits["test"],
        lag_cols,
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
    second_stage_features = [*lag_cols, *emb_cols]
    model_preds = train_residual_tabular_models(enhanced_splits, second_stage_features, base_preds)

    prediction_parts = []
    for model_name, preds in model_preds.items():
        prediction_parts.extend(frame_predictions(enhanced_splits, preds, model_name, "corr_top3_directed"))
    new_predictions = pd.concat(prediction_parts, ignore_index=True)

    existing = (
        pd.read_csv(REGION_PREDICTIONS_CSV, parse_dates=["date"])
        if REGION_PREDICTIONS_CSV.exists()
        else pd.DataFrame()
    )
    if not existing.empty:
        existing = existing[~existing["model_name"].isin(EMBEDDING_MODEL_NAMES)].copy()
    combined = pd.concat([existing, new_predictions], ignore_index=True)
    combined = combined.drop_duplicates(
        ["date", "basin_id", "model_name", "graph_type", "split"],
        keep="last",
    )

    REGION_PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(REGION_PREDICTIONS_CSV, index=False)

    overall = metrics_overall(combined).sort_values(["split", "rmse_cm"])
    by_basin = metrics_by_basin(combined, split="test").sort_values(["basin_name", "rmse_cm"])
    diagnostics = prediction_diagnostics(combined)
    improvement = improvement_by_basin(by_basin)

    overall.to_csv(REGION_METRICS_OVERALL_CSV, index=False)
    write_csv_or_warn(by_basin, REGION_METRICS_BY_BASIN_CSV, index=False)
    write_csv_or_warn(improvement, REGION_IMPROVEMENT_BY_BASIN_CSV, index=False)
    write_csv_or_warn(diagnostics, REGION_PREDICTION_DIAGNOSTICS_CSV, index=False)

    print("GNN embedding residual models:")
    keep = overall["split"].eq("test") & overall["model_name"].isin(EMBEDDING_MODEL_NAMES)
    print(overall[keep].sort_values("rmse_cm").to_string(index=False))
    print("\nTop test models:")
    print(overall[overall["split"].eq("test")].sort_values("rmse_cm").head(12).to_string(index=False))


if __name__ == "__main__":
    main()
