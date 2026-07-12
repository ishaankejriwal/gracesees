from __future__ import annotations

import sys
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import numpy as np
import pandas as pd

from grace_gnn.config import (
    RANDOM_SEED,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VAL_FRACTION,
)
from grace_gnn.evaluate import prediction_frame
from grace_gnn.experiment import ExperimentPaths, MaskExperiment
from grace_gnn.features import feature_columns
from grace_gnn.graph import (
    build_knn_edges_from_mask_zips,
    make_degree_matched_random_edges,
    reverse_edges,
    save_edges,
    symmetrize_edges,
)
from grace_gnn.metrics import (
    improvement_by_basin,
    metrics_by_basin,
    metrics_overall,
    prediction_diagnostics,
)
from grace_gnn.models import set_seeds
from grace_gnn.splits import chronological_fraction_split
from grace_gnn.validation import validate_edges, validate_lagged_dataset


EXTRA_MODEL_NAMES = {
    "ridge_residual_mlp",
    "ridge_neighbor_ar",
    "ridge_neighbor_residual_mlp",
    "ridge_neighbor_residual_tcn",
    "ridge_neighbor_residual_lstm",
}


def _resolve_mask_zip(mask_zip: str | Path) -> Path:
    path = Path(mask_zip)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists() and not str(mask_zip).startswith("masks"):
        candidate = ROOT / "masks" / Path(mask_zip).name
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(f"Missing mask zip: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extra GRACE mask-region model architectures.")
    parser.add_argument("--experiment", default="africa_l3_no_madagascar", help="Experiment/output name.")
    parser.add_argument("--mask-zip", default=None, help="Path or filename under masks/ for the mask zip.")
    parser.add_argument("--basin-name-filter", default=None, help="Keep only mask names containing this text.")
    parser.add_argument("--basin-name-exclude", default=None, help="Exclude mask names containing this text.")
    parser.add_argument("--strict-mask-names", action="store_true", help="Require HydroBASINS-style mask filenames.")
    return parser.parse_args()


def experiment_from_args(args: argparse.Namespace) -> MaskExperiment:
    if args.mask_zip is None and args.experiment == "africa_l3_no_madagascar":
        return MaskExperiment.africa_l3_default()
    if args.mask_zip is None:
        raise ValueError("--mask-zip is required for custom experiments.")
    return MaskExperiment(
        paths=ExperimentPaths.from_name(args.experiment),
        mask_zip=_resolve_mask_zip(args.mask_zip),
        basin_name_filter=args.basin_name_filter,
        basin_name_exclude=args.basin_name_exclude,
        strict_mask_names=args.strict_mask_names,
    )


def train_ridge_predictions(train_df, val_df, test_df, feature_cols):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train_df[feature_cols], train_df["target_twsa_cm"])

    def predict(frame):
        return model.predict(frame[feature_cols]) if len(frame) else np.array([])

    return model, {"train": predict(train_df), "val": predict(val_df), "test": predict(test_df)}


def train_residual_mlp(train_df, val_df, test_df, feature_cols, base_preds, seed: int = RANDOM_SEED):
    import torch
    from torch import nn
    from sklearn.preprocessing import StandardScaler

    set_seeds(seed)
    x_scaler = StandardScaler().fit(train_df[feature_cols])
    residual_scaler = StandardScaler().fit(
        (train_df["target_twsa_cm"].to_numpy() - base_preds["train"]).reshape(-1, 1)
    )
    basin_ids = sorted(
        set(train_df["basin_id"].astype(str))
        | set(val_df["basin_id"].astype(str))
        | set(test_df["basin_id"].astype(str))
    )
    basin_to_idx = {basin_id: i for i, basin_id in enumerate(basin_ids)}
    embedding_dim = min(8, max(2, int(np.ceil(np.sqrt(len(basin_ids))))))

    def tensors(frame, split_name):
        x = torch.tensor(x_scaler.transform(frame[feature_cols]), dtype=torch.float32)
        basin = torch.tensor(frame["basin_id"].astype(str).map(basin_to_idx).to_numpy(), dtype=torch.long)
        residual = frame["target_twsa_cm"].to_numpy() - base_preds[split_name]
        y = torch.tensor(residual_scaler.transform(residual.reshape(-1, 1)), dtype=torch.float32)
        return x, basin, y

    x_train, basin_train, y_train = tensors(train_df, "train")
    x_val, basin_val, y_val = tensors(val_df, "val") if len(val_df) else (None, None, None)

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

    def predict(frame, split_name):
        if frame.empty:
            return np.array([])
        x = torch.tensor(x_scaler.transform(frame[feature_cols]), dtype=torch.float32)
        basin = torch.tensor(frame["basin_id"].astype(str).map(basin_to_idx).to_numpy(), dtype=torch.long)
        model.eval()
        with torch.no_grad():
            residual_scaled = model(x, basin).numpy()
        residual = residual_scaler.inverse_transform(residual_scaled).ravel()
        return base_preds[split_name] + residual

    return model, {
        "train": predict(train_df, "train"),
        "val": predict(val_df, "val"),
        "test": predict(test_df, "test"),
    }


def train_residual_sequence_model(
    train_df,
    val_df,
    test_df,
    feature_cols,
    base_preds,
    architecture: str,
    seed: int = RANDOM_SEED,
):
    import torch
    from torch import nn
    from sklearn.preprocessing import StandardScaler

    if architecture not in {"tcn", "lstm"}:
        raise ValueError(f"Unsupported residual sequence architecture: {architecture}")

    set_seeds(seed)
    lag_numbers = sorted(
        {int(col.split("_")[-1]) for col in feature_cols if col.startswith("lag_") or col.startswith("neighbor_lag_")},
        reverse=True,
    )
    sequence_cols = []
    for lag in lag_numbers:
        step_cols = [f"lag_{lag}"]
        neighbor_col = f"neighbor_lag_{lag}"
        if neighbor_col in feature_cols:
            step_cols.append(neighbor_col)
        sequence_cols.extend(step_cols)
    missing = [col for col in sequence_cols if col not in feature_cols]
    if missing:
        raise ValueError(f"Missing sequence feature columns: {missing}")

    x_scaler = StandardScaler().fit(train_df[sequence_cols])
    residual_scaler = StandardScaler().fit(
        (train_df["target_twsa_cm"].to_numpy() - base_preds["train"]).reshape(-1, 1)
    )
    basin_ids = sorted(
        set(train_df["basin_id"].astype(str))
        | set(val_df["basin_id"].astype(str))
        | set(test_df["basin_id"].astype(str))
    )
    basin_to_idx = {basin_id: i for i, basin_id in enumerate(basin_ids)}
    embedding_dim = min(8, max(2, int(np.ceil(np.sqrt(len(basin_ids))))))
    seq_len = len(lag_numbers)
    channels = len(sequence_cols) // seq_len

    def tensors(frame, split_name):
        scaled = x_scaler.transform(frame[sequence_cols])
        x = torch.tensor(scaled.reshape(len(frame), seq_len, channels), dtype=torch.float32)
        basin = torch.tensor(frame["basin_id"].astype(str).map(basin_to_idx).to_numpy(), dtype=torch.long)
        residual = frame["target_twsa_cm"].to_numpy() - base_preds[split_name]
        y = torch.tensor(residual_scaler.transform(residual.reshape(-1, 1)), dtype=torch.float32)
        return x, basin, y

    x_train, basin_train, y_train = tensors(train_df, "train")
    has_val = len(val_df) > 0
    x_val, basin_val, y_val = tensors(val_df, "val") if has_val else (None, None, None)

    class ResidualSequenceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.basin_embedding = nn.Embedding(len(basin_ids), embedding_dim)
            hidden = 24
            if architecture == "lstm":
                self.encoder = nn.LSTM(input_size=channels, hidden_size=hidden, batch_first=True)
                encoded_dim = hidden
            else:
                self.encoder = nn.Sequential(
                    nn.Conv1d(channels, hidden, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(0.10),
                    nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
                encoded_dim = hidden
            self.head = nn.Sequential(
                nn.Linear(encoded_dim + embedding_dim, 24),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(24, 1),
            )

        def forward(self, x, basin):
            if architecture == "lstm":
                _, (h, _) = self.encoder(x)
                seq_state = h[-1]
            else:
                h = self.encoder(x.transpose(1, 2))
                seq_state = h.mean(dim=2)
            basin_state = self.basin_embedding(basin)
            return self.head(torch.cat([seq_state, basin_state], dim=1))

    model = ResidualSequenceModel()
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
            val_loss = loss_fn(model(x_val, basin_val), y_val).item() if has_val else loss.item()
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

    def predict(frame, split_name):
        if frame.empty:
            return np.array([])
        x, basin, _ = tensors(frame, split_name)
        model.eval()
        with torch.no_grad():
            residual_scaled = model(x, basin).numpy()
        residual = residual_scaler.inverse_transform(residual_scaled).ravel()
        return base_preds[split_name] + residual

    return model, {
        "train": predict(train_df, "train"),
        "val": predict(val_df, "val"),
        "test": predict(test_df, "test"),
    }


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


def add_neighbor_lag_features(df: pd.DataFrame, edges: pd.DataFrame, lag_cols: list[str]) -> pd.DataFrame:
    data = df.copy()
    data["basin_id"] = data["basin_id"].astype(str)
    edges = edges.copy()
    edges["src_basin_id"] = edges["src_basin_id"].astype(str)
    edges["dst_basin_id"] = edges["dst_basin_id"].astype(str)
    if "weight" not in edges.columns:
        edges["weight"] = 1.0
    rows = data[["date", "basin_id", *lag_cols]].rename(columns={"basin_id": "src_basin_id"})
    joined = edges[["src_basin_id", "dst_basin_id", "weight"]].merge(rows, on="src_basin_id", how="left")
    for col in lag_cols:
        joined[f"{col}_weighted"] = joined[col] * joined["weight"]
    grouped = joined.groupby(["date", "dst_basin_id"], dropna=False)
    agg = grouped[[f"{col}_weighted" for col in lag_cols]].sum()
    denom = grouped["weight"].sum().replace(0, np.nan)
    neighbor = agg.div(denom, axis=0).reset_index()
    neighbor = neighbor.rename(
        columns={
            "dst_basin_id": "basin_id",
            **{f"{col}_weighted": f"neighbor_{col}" for col in lag_cols},
        }
    )
    out = data.merge(neighbor, on=["date", "basin_id"], how="left")
    for col in lag_cols:
        out[f"neighbor_{col}"] = out[f"neighbor_{col}"].fillna(out[col])
    return out


def load_edge_variants(
    lagged: pd.DataFrame,
    train_df: pd.DataFrame,
    experiment: MaskExperiment,
) -> dict[str, pd.DataFrame]:
    basin_ids = sorted(lagged["basin_id"].astype(str).unique())
    basin_names = sorted(lagged["basin_name"].dropna().unique())
    real_directed = build_knn_edges_from_mask_zips(
        [experiment.mask_zip],
        basin_names,
        experiment.paths.output_dir / "edges_real_knn_directed.csv",
        k=3,
        graph_type="real_knn_directed",
        strict_mask_names=experiment.strict_mask_names,
    )
    variants = {
        "real_knn_directed": real_directed,
        "real_knn_undirected": symmetrize_edges(real_directed, graph_type="real_knn_undirected"),
        "real_knn_reversed": reverse_edges(real_directed, graph_type="real_knn_reversed"),
        "random_degree_matched": make_degree_matched_random_edges(real_directed, basin_ids, seed=RANDOM_SEED),
    }
    variants["corr_top3_directed"] = correlation_topk_edges(train_df, top_k=3)
    basin_id_set = set(basin_ids)
    for graph_type, edges in variants.items():
        validate_edges(edges, basin_id_set, graph_type=graph_type)
        save_edges(edges, experiment.paths.output_dir / f"edges_{graph_type}.csv")
    return variants


def validate_existing_predictions(existing: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> None:
    required = {"date", "basin_id", "model_name", "graph_type", "split", "observed_twsa_cm", "predicted_twsa_cm"}
    missing = required - set(existing.columns)
    if missing:
        raise ValueError(f"Existing predictions are missing columns: {sorted(missing)}")
    data = existing.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["basin_id"] = data["basin_id"].astype(str)
    expected = {
        split_name: {
            "n": len(frame),
            "basin_ids": set(frame["basin_id"].astype(str).unique()),
            "dates": set(pd.to_datetime(frame["date"]).unique()),
        }
        for split_name, frame in splits.items()
    }
    for (model_name, graph_type, split_name), group in data.groupby(["model_name", "graph_type", "split"], dropna=False):
        if split_name not in expected:
            raise ValueError(f"Existing predictions contain unknown split {split_name!r}.")
        details = expected[split_name]
        if len(group) != details["n"]:
            raise ValueError(
                f"Existing predictions for {model_name}|{graph_type}|{split_name} have {len(group)} rows; "
                f"expected {details['n']} for the current lagged dataset."
            )
        if set(group["basin_id"].unique()) != details["basin_ids"]:
            raise ValueError(f"Existing predictions for {model_name}|{graph_type}|{split_name} use stale basin IDs.")
        if set(group["date"].unique()) != details["dates"]:
            raise ValueError(f"Existing predictions for {model_name}|{graph_type}|{split_name} use stale dates.")


def frame_predictions(splits, preds, model_name: str, graph_type: str) -> list[pd.DataFrame]:
    return [
        prediction_frame(splits[split_name], preds[split_name], model_name, graph_type, split_name)
        for split_name in ["train", "val", "test"]
    ]


def main() -> None:
    experiment = experiment_from_args(parse_args())
    paths = experiment.paths
    lagged = pd.read_csv(paths.lagged_dataset_csv, parse_dates=["date"])
    validate_lagged_dataset(lagged)
    unique_dates = pd.to_datetime(lagged["date"]).nunique()
    if unique_dates < 3:
        raise ValueError(
            f"{paths.lagged_dataset_csv} has fewer than three unique dates. "
            "Rerun scripts/run_africa_l3.py after checking that the custom masks overlap "
            "GRACE cells with enough valid monthly data."
        )
    splits = chronological_fraction_split(lagged, TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION)
    for frame in splits.values():
        validate_lagged_dataset(frame)
    lag_cols = feature_columns(lagged)
    prediction_parts = []

    _, base_preds = train_ridge_predictions(splits["train"], splits["val"], splits["test"], lag_cols)
    _, residual_preds = train_residual_mlp(splits["train"], splits["val"], splits["test"], lag_cols, base_preds)
    prediction_parts.extend(frame_predictions(splits, residual_preds, "ridge_residual_mlp", "own_lags"))

    edge_variants = load_edge_variants(lagged, splits["train"], experiment)
    neighbor_val_scores = {}
    neighbor_frames = {}
    for graph_type, edges in edge_variants.items():
        neighbor_splits = {
            split_name: add_neighbor_lag_features(frame, edges, lag_cols)
            for split_name, frame in splits.items()
        }
        neighbor_cols = [*lag_cols, *[f"neighbor_{col}" for col in lag_cols]]
        _, neighbor_preds = train_ridge_predictions(
            neighbor_splits["train"],
            neighbor_splits["val"],
            neighbor_splits["test"],
            neighbor_cols,
        )
        prediction_parts.extend(
            frame_predictions(neighbor_splits, neighbor_preds, "ridge_neighbor_ar", graph_type)
        )
        val_rmse = np.sqrt(
            np.mean((neighbor_splits["val"]["target_twsa_cm"].to_numpy() - neighbor_preds["val"]) ** 2)
        )
        neighbor_val_scores[graph_type] = float(val_rmse)
        neighbor_frames[graph_type] = (neighbor_splits, neighbor_cols, neighbor_preds)

    best_graph = min(neighbor_val_scores, key=neighbor_val_scores.get)
    best_neighbor_splits, best_neighbor_cols, best_neighbor_base_preds = neighbor_frames[best_graph]
    _, neighbor_residual_preds = train_residual_mlp(
        best_neighbor_splits["train"],
        best_neighbor_splits["val"],
        best_neighbor_splits["test"],
        best_neighbor_cols,
        best_neighbor_base_preds,
        seed=RANDOM_SEED + 1,
    )
    prediction_parts.extend(
        frame_predictions(
            best_neighbor_splits,
            neighbor_residual_preds,
            "ridge_neighbor_residual_mlp",
            best_graph,
        )
    )
    for architecture, model_name, seed in [
        ("tcn", "ridge_neighbor_residual_tcn", RANDOM_SEED + 2),
        ("lstm", "ridge_neighbor_residual_lstm", RANDOM_SEED + 3),
    ]:
        _, sequence_preds = train_residual_sequence_model(
            best_neighbor_splits["train"],
            best_neighbor_splits["val"],
            best_neighbor_splits["test"],
            best_neighbor_cols,
            best_neighbor_base_preds,
            architecture=architecture,
            seed=seed,
        )
        prediction_parts.extend(frame_predictions(best_neighbor_splits, sequence_preds, model_name, best_graph))

    new_predictions = pd.concat(prediction_parts, ignore_index=True)
    existing = (
        pd.read_csv(paths.predictions_csv, parse_dates=["date"])
        if paths.predictions_csv.exists()
        else pd.DataFrame()
    )
    if not existing.empty:
        validate_existing_predictions(existing, splits)
        existing = existing[~existing["model_name"].isin(EXTRA_MODEL_NAMES)].copy()
    combined = pd.concat([existing, new_predictions], ignore_index=True)
    combined = combined.drop_duplicates(
        ["date", "basin_id", "model_name", "graph_type", "split"],
        keep="last",
    )
    paths.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(paths.predictions_csv, index=False)

    overall = metrics_overall(combined).sort_values(["split", "rmse_cm"])
    by_basin = metrics_by_basin(combined, split="test").sort_values(["basin_name", "rmse_cm"])
    improvement = improvement_by_basin(by_basin)
    diagnostics = prediction_diagnostics(combined)
    overall.to_csv(paths.metrics_overall_csv, index=False)
    by_basin.to_csv(paths.metrics_by_basin_csv, index=False)
    improvement.to_csv(paths.improvement_by_basin_csv, index=False)
    diagnostics.to_csv(paths.prediction_diagnostics_csv, index=False)

    print("Neighbor ridge validation RMSE:")
    for graph_type, rmse in sorted(neighbor_val_scores.items(), key=lambda item: item[1]):
        print(f"  {graph_type}: {rmse:.4f}")
    print(f"Selected for residual follow-up: {best_graph}")
    print(overall[overall["split"].eq("test")].to_string(index=False))


if __name__ == "__main__":
    main()
