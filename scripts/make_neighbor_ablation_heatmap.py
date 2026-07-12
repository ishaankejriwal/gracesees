from __future__ import annotations

import sys
from pathlib import Path
from textwrap import shorten

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grace_gnn.config import (
    LAGGED_DATASET_CSV,
    RANDOM_SEED,
    REGION_OUTPUTS,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VAL_FRACTION,
)
from grace_gnn.features import feature_columns
from grace_gnn.models import set_seeds
from grace_gnn.splits import chronological_fraction_split
from scripts.run_africa_l3_extra_architectures import add_neighbor_lag_features


MODEL_NAME = "ridge_neighbor_residual_mlp"
GRAPH_TYPE = "corr_top3_directed"
FIGURE_DIR = REGION_OUTPUTS / "figures" / "neighbor_ablation_heatmap"


def _train_ridge(train_df: pd.DataFrame, feature_cols: list[str]):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train_df[feature_cols], train_df["target_twsa_cm"])
    return model


def _train_residual_mlp(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    base_train: np.ndarray,
    base_val: np.ndarray,
):
    import torch
    from sklearn.preprocessing import StandardScaler
    from torch import nn

    set_seeds(RANDOM_SEED + 1)
    x_scaler = StandardScaler().fit(train_df[feature_cols])
    residual_scaler = StandardScaler().fit((train_df["target_twsa_cm"].to_numpy() - base_train).reshape(-1, 1))
    basin_ids = sorted(set(train_df["basin_id"].astype(str)) | set(val_df["basin_id"].astype(str)))
    basin_to_idx = {basin_id: i for i, basin_id in enumerate(basin_ids)}
    embedding_dim = min(8, max(2, int(np.ceil(np.sqrt(len(basin_ids))))))

    def tensors(frame: pd.DataFrame, base_pred: np.ndarray):
        x = torch.tensor(x_scaler.transform(frame[feature_cols]), dtype=torch.float32)
        basin = torch.tensor(frame["basin_id"].astype(str).map(basin_to_idx).to_numpy(), dtype=torch.long)
        residual = frame["target_twsa_cm"].to_numpy() - base_pred
        y = torch.tensor(residual_scaler.transform(residual.reshape(-1, 1)), dtype=torch.float32)
        return x, basin, y

    class ResidualMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.basin_embedding = nn.Embedding(len(basin_ids), embedding_dim)
            self.net = nn.Sequential(
                nn.Linear(len(feature_cols) + embedding_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, x, basin):
            return self.net(torch.cat([x, self.basin_embedding(basin)], dim=1))

    x_train, basin_train, y_train = tensors(train_df, base_train)
    x_val, basin_val, y_val = tensors(val_df, base_val) if len(val_df) else (None, None, None)
    model = ResidualMLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    stale = 0
    for _ in range(300):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x_train, basin_train), y_train)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val, basin_val), y_val).item() if x_val is not None else loss.item()
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

    def predict_residual(frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.array([])
        x = torch.tensor(x_scaler.transform(frame[feature_cols]), dtype=torch.float32)
        basin = torch.tensor(frame["basin_id"].astype(str).map(basin_to_idx).to_numpy(), dtype=torch.long)
        model.eval()
        with torch.no_grad():
            residual_scaled = model(x, basin).numpy()
        return residual_scaler.inverse_transform(residual_scaled).ravel()

    return predict_residual


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _short_label(name: str) -> str:
    return shorten(str(name).replace(" Coast", "").replace(" Basin", ""), width=24, placeholder="...")


def _plot_delta_heatmap(impact: pd.DataFrame, basin_names: pd.Series, output_path: Path) -> None:
    basin_ids = sorted(set(impact["src_basin_id"]) | set(impact["dst_basin_id"]), key=lambda bid: basin_names.get(bid, bid))
    matrix = pd.DataFrame(np.nan, index=basin_ids, columns=basin_ids, dtype=float)
    for row in impact.itertuples(index=False):
        matrix.loc[str(row.dst_basin_id), str(row.src_basin_id)] = float(row.delta_rmse_cm)

    fig, ax = plt.subplots(figsize=(18, 14))
    masked = np.ma.masked_invalid(matrix.to_numpy())
    vmax = np.nanmax(np.abs(matrix.to_numpy()))
    vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0
    image = ax.imshow(masked, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("Leave-one-neighbor-out impact on test RMSE\nridge_neighbor_residual_mlp | corr_top3_directed")
    ax.set_xlabel("Removed source neighbor basin")
    ax.set_ylabel("Destination basin being predicted")
    ax.set_xticks(np.arange(len(basin_ids)))
    ax.set_yticks(np.arange(len(basin_ids)))
    ax.set_xticklabels([_short_label(basin_names.get(bid, bid)) for bid in basin_ids], rotation=90, fontsize=7)
    ax.set_yticklabels([_short_label(basin_names.get(bid, bid)) for bid in basin_ids], fontsize=7)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Delta RMSE cm after removing edge\npositive = neighbor helped; negative = neighbor hurt")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_weight_heatmap(edges: pd.DataFrame, basin_names: pd.Series, output_path: Path) -> None:
    basin_ids = sorted(set(edges["src_basin_id"]) | set(edges["dst_basin_id"]), key=lambda bid: basin_names.get(bid, bid))
    matrix = pd.DataFrame(np.nan, index=basin_ids, columns=basin_ids, dtype=float)
    for row in edges.itertuples(index=False):
        matrix.loc[str(row.dst_basin_id), str(row.src_basin_id)] = float(row.weight)

    fig, ax = plt.subplots(figsize=(18, 14))
    image = ax.imshow(np.ma.masked_invalid(matrix.to_numpy()), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_title("Train-period correlation weights\ncorr_top3_directed")
    ax.set_xlabel("Source neighbor basin")
    ax.set_ylabel("Destination basin")
    ax.set_xticks(np.arange(len(basin_ids)))
    ax.set_yticks(np.arange(len(basin_ids)))
    ax.set_xticklabels([_short_label(basin_names.get(bid, bid)) for bid in basin_ids], rotation=90, fontsize=7)
    ax.set_yticklabels([_short_label(basin_names.get(bid, bid)) for bid in basin_ids], fontsize=7)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Correlation edge weight")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    lagged = pd.read_csv(LAGGED_DATASET_CSV, parse_dates=["date"])
    edges = pd.read_csv(REGION_OUTPUTS / f"edges_{GRAPH_TYPE}.csv")
    edges["src_basin_id"] = edges["src_basin_id"].astype(str)
    edges["dst_basin_id"] = edges["dst_basin_id"].astype(str)
    if "weight" not in edges.columns:
        edges["weight"] = 1.0

    splits = chronological_fraction_split(lagged, TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION)
    lag_cols = feature_columns(lagged)
    neighbor_splits = {name: add_neighbor_lag_features(frame, edges, lag_cols) for name, frame in splits.items()}
    feature_cols = [*lag_cols, *[f"neighbor_{col}" for col in lag_cols]]

    ridge = _train_ridge(neighbor_splits["train"], feature_cols)
    base_train = ridge.predict(neighbor_splits["train"][feature_cols])
    base_val = ridge.predict(neighbor_splits["val"][feature_cols])
    predict_residual = _train_residual_mlp(neighbor_splits["train"], neighbor_splits["val"], feature_cols, base_train, base_val)

    full_test = neighbor_splits["test"].copy()
    full_base_test = ridge.predict(full_test[feature_cols])
    full_pred = full_base_test + predict_residual(full_test)
    observed = full_test["target_twsa_cm"].to_numpy()
    full_rmse = _rmse(observed, full_pred)

    rows = []
    for idx, edge in edges.reset_index(drop=True).iterrows():
        ablated_edges = edges.drop(index=idx).reset_index(drop=True)
        ablated_test = add_neighbor_lag_features(splits["test"], ablated_edges, lag_cols)
        ablated_base = ridge.predict(ablated_test[feature_cols])
        ablated_pred = ablated_base + predict_residual(ablated_test)

        dst = str(edge["dst_basin_id"])
        dst_mask = full_test["basin_id"].astype(str).eq(dst).to_numpy()
        src = str(edge["src_basin_id"])
        src_name = lagged.loc[lagged["basin_id"].astype(str).eq(src), "basin_name"].dropna().iloc[0]
        dst_name = lagged.loc[lagged["basin_id"].astype(str).eq(dst), "basin_name"].dropna().iloc[0]
        full_dst_rmse = _rmse(observed[dst_mask], full_pred[dst_mask])
        ablated_dst_rmse = _rmse(observed[dst_mask], ablated_pred[dst_mask])
        rows.append(
            {
                "model_name": MODEL_NAME,
                "graph_type": GRAPH_TYPE,
                "src_basin_id": src,
                "src_basin_name": src_name,
                "dst_basin_id": dst,
                "dst_basin_name": dst_name,
                "edge_weight": float(edge["weight"]),
                "full_test_rmse_cm": full_rmse,
                "full_dst_rmse_cm": full_dst_rmse,
                "ablated_dst_rmse_cm": ablated_dst_rmse,
                "delta_rmse_cm": ablated_dst_rmse - full_dst_rmse,
            }
        )

    impact = pd.DataFrame(rows).sort_values("delta_rmse_cm", ascending=False)
    impact.to_csv(FIGURE_DIR / "neighbor_ablation_edge_impacts.csv", index=False)
    impact.head(20).to_csv(FIGURE_DIR / "top_helpful_neighbor_edges.csv", index=False)
    impact.tail(20).sort_values("delta_rmse_cm").to_csv(FIGURE_DIR / "top_harmful_or_unhelpful_neighbor_edges.csv", index=False)

    basin_names = lagged.drop_duplicates("basin_id").assign(basin_id=lambda df: df["basin_id"].astype(str))
    basin_name_map = basin_names.set_index("basin_id")["basin_name"]
    _plot_delta_heatmap(impact, basin_name_map, FIGURE_DIR / "neighbor_ablation_rmse_delta_heatmap.png")
    _plot_weight_heatmap(edges, basin_name_map, FIGURE_DIR / "corr_top3_edge_weight_heatmap.png")

    print(f"Saved figures and CSVs to {FIGURE_DIR}")
    print("Top helpful edges by destination RMSE increase when removed:")
    print(impact.head(10)[["src_basin_name", "dst_basin_name", "edge_weight", "delta_rmse_cm"]].to_string(index=False))


if __name__ == "__main__":
    main()
