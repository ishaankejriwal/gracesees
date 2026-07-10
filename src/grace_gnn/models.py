from __future__ import annotations

import numpy as np
import pandas as pd


def set_seeds(seed: int = 42) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def predict_persistence(df: pd.DataFrame) -> np.ndarray:
    if "lag_1" not in df.columns:
        raise ValueError("Persistence baseline requires lag_1.")
    return df["lag_1"].to_numpy()


def train_ridge(train_df, val_df, test_df, feature_cols, alpha: float = 1.0):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(train_df[feature_cols], train_df["target_twsa_cm"])

    def predict(frame):
        if frame.empty:
            return np.array([])
        return model.predict(frame[feature_cols])

    return model, {"train": predict(train_df), "val": predict(val_df), "test": predict(test_df)}


def train_random_forest(
    train_df,
    val_df,
    test_df,
    feature_cols,
    seed: int = 42,
    n_estimators: int = 500,
    max_depth: int | None = 8,
):
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train_df[feature_cols], train_df["target_twsa_cm"])

    def predict(frame):
        if frame.empty:
            return np.array([])
        return model.predict(frame[feature_cols])

    return model, {"train": predict(train_df), "val": predict(val_df), "test": predict(test_df)}


def train_xgboost(
    train_df,
    val_df,
    test_df,
    feature_cols,
    seed: int = 42,
    n_estimators: int = 300,
):
    from xgboost import XGBRegressor

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(train_df[feature_cols], train_df["target_twsa_cm"])

    def predict(frame):
        if frame.empty:
            return np.array([])
        return model.predict(frame[feature_cols])

    return model, {"train": predict(train_df), "val": predict(val_df), "test": predict(test_df)}


def train_correlation_neighbor(train_df, val_df, test_df):
    train = train_df.copy()
    train["date"] = pd.to_datetime(train["date"])
    train["basin_id"] = train["basin_id"].astype(str)
    pivot = train.pivot_table(index="date", columns="basin_id", values="target_twsa_cm", aggfunc="first")
    corr = pivot.corr()

    def predict(frame):
        if frame.empty:
            return np.array([])
        data = frame.copy()
        data["date"] = pd.to_datetime(data["date"])
        data["basin_id"] = data["basin_id"].astype(str)
        lag_lookup = data.pivot_table(index="date", columns="basin_id", values="lag_1", aggfunc="first")
        preds = []
        for row in data.itertuples(index=False):
            basin_id = str(row.basin_id)
            weights = corr[basin_id].drop(labels=[basin_id], errors="ignore").dropna() if basin_id in corr else pd.Series(dtype=float)
            values = lag_lookup.loc[row.date, weights.index].dropna() if len(weights) and row.date in lag_lookup.index else pd.Series(dtype=float)
            weights = weights.reindex(values.index).clip(lower=0.0)
            if len(values) and float(weights.sum()) > 0:
                preds.append(float(np.average(values.to_numpy(), weights=weights.to_numpy())))
            else:
                preds.append(float(row.lag_1))
        return np.asarray(preds)

    return corr, {"train": predict(train_df), "val": predict(val_df), "test": predict(test_df)}


def train_mlp(train_df, val_df, test_df, feature_cols, seed: int = 42, epochs: int = 300, lr: float = 1e-3):
    import torch
    from torch import nn
    from sklearn.preprocessing import StandardScaler

    set_seeds(seed)
    x_scaler = StandardScaler().fit(train_df[feature_cols])
    y_scaler = StandardScaler().fit(train_df[["target_twsa_cm"]])

    def xy(frame):
        x = torch.tensor(x_scaler.transform(frame[feature_cols]), dtype=torch.float32)
        y = torch.tensor(y_scaler.transform(frame[["target_twsa_cm"]]), dtype=torch.float32)
        return x, y

    x_train, y_train = xy(train_df)
    x_val, y_val = xy(val_df) if len(val_df) else (None, None)
    model = nn.Sequential(
        nn.Linear(len(feature_cols), 32),
        nn.ReLU(),
        nn.Dropout(0.10),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    patience = 40
    stale = 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val).item() if x_val is not None else loss.item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)

    def predict(frame):
        if frame.empty:
            return np.array([])
        x = torch.tensor(x_scaler.transform(frame[feature_cols]), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            pred_scaled = model(x).numpy()
        return y_scaler.inverse_transform(pred_scaled).ravel()

    return model, {"train": predict(train_df), "val": predict(val_df), "test": predict(test_df)}


def train_recurrent_lag_model(
    train_df,
    val_df,
    test_df,
    feature_cols,
    seed: int = 42,
    cell_type: str = "gru",
    hidden: int = 24,
    epochs: int = 300,
    lr: float = 1e-3,
):
    import torch
    from torch import nn
    from sklearn.preprocessing import StandardScaler

    if cell_type not in {"gru", "rnn"}:
        raise ValueError(f"Unsupported recurrent cell_type: {cell_type}")

    set_seeds(seed)
    ordered_features = sorted(feature_cols, key=lambda x: int(x.split("_")[1]), reverse=True)
    x_scaler = StandardScaler().fit(train_df[ordered_features])
    y_scaler = StandardScaler().fit(train_df[["target_twsa_cm"]])
    basin_ids = sorted(
        set(train_df["basin_id"].astype(str))
        | set(val_df["basin_id"].astype(str))
        | set(test_df["basin_id"].astype(str))
    )
    basin_to_idx = {basin_id: i for i, basin_id in enumerate(basin_ids)}
    embedding_dim = min(8, max(2, int(np.ceil(np.sqrt(len(basin_ids))))))

    def tensors(frame):
        scaled = x_scaler.transform(frame[ordered_features])
        x = torch.tensor(scaled.reshape(len(frame), len(ordered_features), 1), dtype=torch.float32)
        y = torch.tensor(y_scaler.transform(frame[["target_twsa_cm"]]), dtype=torch.float32)
        basin = torch.tensor(
            frame["basin_id"].astype(str).map(basin_to_idx).to_numpy(),
            dtype=torch.long,
        )
        return x, basin, y

    x_train, basin_train, y_train = tensors(train_df)
    has_val = len(val_df) > 0
    x_val, basin_val, y_val = tensors(val_df) if has_val else (None, None, None)

    class _RecurrentLagModel(nn.Module):
        def __init__(self):
            super().__init__()
            if cell_type == "gru":
                self.recurrent = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
            else:
                self.recurrent = nn.RNN(input_size=1, hidden_size=hidden, batch_first=True, nonlinearity="tanh")
            self.basin_embedding = nn.Embedding(len(basin_ids), embedding_dim)
            self.head = nn.Sequential(
                nn.Linear(hidden + embedding_dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(hidden, 1),
            )

        def forward(self, x, basin):
            _, h = self.recurrent(x)
            seq_state = h[-1]
            basin_state = self.basin_embedding(basin)
            return self.head(torch.cat([seq_state, basin_state], dim=1))

    model = _RecurrentLagModel()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    patience = 40
    stale = 0
    for _ in range(epochs):
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
        if stale >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)

    def predict(frame):
        if frame.empty:
            return np.array([])
        x, basin, _ = tensors(frame)
        model.eval()
        with torch.no_grad():
            pred_scaled = model(x, basin).numpy()
        return y_scaler.inverse_transform(pred_scaled).ravel()

    return model, {"train": predict(train_df), "val": predict(val_df), "test": predict(test_df)}


class ManualGCN:
    """Small two-layer graph convolution model using A_norm @ X @ W."""

    def __init__(self, in_features: int, hidden: int = 32):
        import torch
        from torch import nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = nn.Linear(in_features, hidden)
                self.lin2 = nn.Linear(hidden, 1)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.10)

            def forward(self, x, a_norm):
                h = a_norm @ x
                h = self.relu(self.lin1(h))
                h = self.dropout(h)
                h = a_norm @ h
                return self.lin2(h).squeeze(-1)

        self.model = _Model()


class ResidualManualGCN:
    """Local MLP plus a small graph-message correction."""

    def __init__(self, in_features: int, hidden: int = 32):
        from torch import nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.local = nn.Sequential(
                    nn.Linear(in_features, hidden),
                    nn.ReLU(),
                    nn.Dropout(0.10),
                    nn.Linear(hidden, 1),
                )
                self.message = nn.Sequential(
                    nn.Linear(in_features, hidden),
                    nn.ReLU(),
                    nn.Dropout(0.10),
                    nn.Linear(hidden, 1),
                )
                self.gate = nn.Sequential(
                    nn.Linear(in_features * 2, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                    nn.Sigmoid(),
                )
                nn.init.zeros_(self.message[-1].weight)
                nn.init.zeros_(self.message[-1].bias)

            def forward(self, x, a_norm):
                import torch

                neighbor_x = a_norm @ x
                local_pred = self.local(x).squeeze(-1)
                neighbor_delta = self.message(neighbor_x).squeeze(-1)
                gate = self.gate(torch.cat([x, neighbor_x], dim=1)).squeeze(-1)
                return local_pred + gate * neighbor_delta

        self.model = _Model()


def make_graph_snapshots(df: pd.DataFrame, basin_ids: list[str], feature_cols: list[str]):
    snapshots = []
    data = df.copy()
    data["basin_id"] = data["basin_id"].astype(str)
    for date, group in data.groupby("date"):
        group = group.set_index("basin_id").reindex(basin_ids)
        target_mask = group["target_twsa_cm"].notna().to_numpy()
        feature_mask = group[feature_cols].notna().all(axis=1).to_numpy()
        mask = target_mask & feature_mask
        if mask.sum() == 0:
            continue
        x = group[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        y = group["target_twsa_cm"].fillna(0.0).to_numpy(dtype=np.float32)
        meta = group.reset_index()[["date", "basin_id", "basin_name", "target_twsa_cm"]]
        meta = meta[mask].copy()
        snapshots.append({
            "date": date,
            "x": x,
            "y": y,
            "mask": mask,
            "meta": meta,
        })
    return snapshots


def train_manual_gcn(
    train_df,
    val_df,
    test_df,
    feature_cols,
    edges,
    basin_ids,
    seed: int = 42,
    epochs: int = 300,
    lr: float = 1e-3,
    residual: bool = False,
):
    import torch
    from sklearn.preprocessing import StandardScaler

    from .graph import normalized_adjacency

    set_seeds(seed)
    x_scaler = StandardScaler().fit(train_df[feature_cols])
    y_scaler = StandardScaler().fit(train_df[["target_twsa_cm"]])

    def scaled(frame):
        out = frame.copy()
        out[feature_cols] = x_scaler.transform(out[feature_cols])
        out["target_twsa_cm"] = y_scaler.transform(out[["target_twsa_cm"]]).ravel()
        return out

    train_snaps = make_graph_snapshots(scaled(train_df), basin_ids, feature_cols)
    val_snaps = make_graph_snapshots(scaled(val_df), basin_ids, feature_cols)
    test_snaps = make_graph_snapshots(scaled(test_df), basin_ids, feature_cols)
    if not train_snaps:
        raise ValueError("No graph snapshots with valid target nodes available for GNN training.")

    a_norm = normalized_adjacency(edges, basin_ids)
    wrapper = ResidualManualGCN(len(feature_cols)) if residual else ManualGCN(len(feature_cols))
    model = wrapper.model
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    best_state = None
    best_val = float("inf")
    patience = 40
    stale = 0

    for _ in range(epochs):
        model.train()
        total = 0.0
        for snap in train_snaps:
            x = torch.tensor(snap["x"], dtype=torch.float32)
            y = torch.tensor(snap["y"], dtype=torch.float32)
            mask = torch.tensor(snap["mask"], dtype=torch.bool)
            opt.zero_grad()
            loss = loss_fn(model(x, a_norm)[mask], y[mask])
            loss.backward()
            opt.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            eval_snaps = val_snaps or train_snaps
            losses = []
            for s in eval_snaps:
                pred = model(torch.tensor(s["x"], dtype=torch.float32), a_norm)
                y = torch.tensor(s["y"], dtype=torch.float32)
                mask = torch.tensor(s["mask"], dtype=torch.bool)
                losses.append(loss_fn(pred[mask], y[mask]).item())
            val_loss = np.mean(losses)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)

    def predict_snaps(snaps):
        rows = []
        model.eval()
        with torch.no_grad():
            for snap in snaps:
                pred_scaled = model(torch.tensor(snap["x"], dtype=torch.float32), a_norm).numpy().reshape(-1, 1)
                pred = y_scaler.inverse_transform(pred_scaled).ravel()
                meta = snap["meta"].copy()
                obs_scaled = meta["target_twsa_cm"].to_numpy().reshape(-1, 1)
                meta["target_twsa_cm"] = y_scaler.inverse_transform(obs_scaled).ravel()
                meta["predicted_twsa_cm"] = pred[snap["mask"]]
                rows.append(meta)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    return model, {"train": predict_snaps(train_snaps), "val": predict_snaps(val_snaps), "test": predict_snaps(test_snaps)}
