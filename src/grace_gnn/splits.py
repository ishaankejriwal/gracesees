from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


def chronological_split(df: pd.DataFrame, train_end: str, val_end: str) -> dict[str, pd.DataFrame]:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    return {
        "train": data[data["date"] <= train_end_ts].copy(),
        "val": data[(data["date"] > train_end_ts) & (data["date"] <= val_end_ts)].copy(),
        "test": data[data["date"] > val_end_ts].copy(),
    }


def chronological_fraction_split(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    val_fraction: float = 0.10,
    test_fraction: float = 0.20,
) -> dict[str, pd.DataFrame]:
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}.")
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    dates = pd.Index(sorted(data["date"].dropna().unique()))
    if len(dates) < 3:
        raise ValueError("Need at least three unique dates for train/val/test splitting.")
    train_count = max(1, int(len(dates) * train_fraction))
    val_count = max(1, int(len(dates) * val_fraction))
    if train_count + val_count >= len(dates):
        val_count = max(1, len(dates) - train_count - 1)
    train_dates = set(dates[:train_count])
    val_dates = set(dates[train_count : train_count + val_count])
    test_dates = set(dates[train_count + val_count :])
    return {
        "train": data[data["date"].isin(train_dates)].copy(),
        "val": data[data["date"].isin(val_dates)].copy(),
        "test": data[data["date"].isin(test_dates)].copy(),
    }


def fit_feature_scaler(train_df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def transform_features(df: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler):
    return scaler.transform(df[feature_cols])
