from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import AFRICA_L2_NO_MADAGASCAR_BASIN_NAMES


def make_lagged_dataset(df: pd.DataFrame, lags: list[int], output_csv: Path | None = None) -> pd.DataFrame:
    required = {"date", "basin_id", "twsa_cm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input data is missing columns: {sorted(missing)}")
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["basin_id"] = data["basin_id"].astype(str)
    if "basin_name" not in data.columns:
        data["basin_name"] = data["basin_id"]
    data["date"] = data["date"].dt.to_period("M").dt.to_timestamp()
    duplicate_months = data[data.duplicated(["basin_id", "date"], keep=False)]
    if not duplicate_months.empty:
        n_pairs = duplicate_months[["basin_id", "date"]].drop_duplicates().shape[0]
        print(f"Averaging {n_pairs} duplicate basin-month GRACE entries before lag creation.")
        data = (
            data.groupby(["basin_id", "date"], as_index=False)
            .agg({"basin_name": "first", "twsa_cm": "mean"})
        )
    data = data.sort_values(["basin_id", "date"])
    monthly_parts = []
    for basin_id, group in data.groupby("basin_id", sort=False):
        group = group.set_index("date").sort_index()
        full_index = pd.date_range(group.index.min(), group.index.max(), freq="MS")
        group = group.reindex(full_index)
        group.index.name = "date"
        group["basin_id"] = basin_id
        group["basin_name"] = group["basin_name"].dropna().iloc[0] if group["basin_name"].notna().any() else basin_id
        monthly_parts.append(group.reset_index())
    data = pd.concat(monthly_parts, ignore_index=True).sort_values(["basin_id", "date"])
    grouped = data.groupby("basin_id", sort=False)["twsa_cm"]
    for lag in lags:
        data[f"lag_{lag}"] = grouped.shift(lag)
    data["target_twsa_cm"] = data["twsa_cm"]
    feature_cols = [f"lag_{lag}" for lag in lags]
    keep_cols = ["date", "basin_id", "basin_name", "target_twsa_cm", *feature_cols]
    out = data[keep_cols].dropna(subset=["target_twsa_cm", *feature_cols]).reset_index(drop=True)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("lag_")], key=lambda x: int(x.split("_")[1]))


def filter_region(df: pd.DataFrame, region: str = "africa_l2_no_madagascar") -> pd.DataFrame:
    if region not in {"africa_l2_no_madagascar", "africa_l3_no_madagascar"}:
        raise ValueError(f"Unknown region: {region}")
    if "basin_name" not in df.columns:
        raise ValueError("Region filtering requires a basin_name column.")
    if region == "africa_l3_no_madagascar":
        out = df[~df["basin_name"].str.contains("madagascar", case=False, na=False)].copy()
        return out
    keep = set(AFRICA_L2_NO_MADAGASCAR_BASIN_NAMES)
    out = df[df["basin_name"].isin(keep)].copy()
    found = set(out["basin_name"].dropna().unique())
    missing = sorted(keep - found)
    if missing:
        print(f"Warning: missing expected Africa Level 2 regions: {missing}")
    return out
