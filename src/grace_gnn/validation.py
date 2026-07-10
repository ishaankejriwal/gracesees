from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "name": path.name,
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def require_matching_provenance(path: Path, expected: dict[str, Any]) -> bool:
    actual = read_json(path)
    return actual == expected


def validate_unique_mask_members(members: pd.DataFrame) -> None:
    if members.empty:
        raise ValueError("No mask members were found.")
    for column in ["basin_id", "basin_name", "member_name"]:
        duplicates = members[members[column].astype(str).duplicated(keep=False)]
        if not duplicates.empty:
            examples = duplicates[column].astype(str).drop_duplicates().head(5).tolist()
            raise ValueError(f"Duplicate mask {column} values found: {examples}")


def validate_basin_month(df: pd.DataFrame, expected_basin_ids: set[str] | None = None) -> None:
    required = {"date", "basin_id", "basin_name", "twsa_cm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Basin-month data is missing columns: {sorted(missing)}")
    data = df.copy()
    data["basin_id"] = data["basin_id"].astype(str)
    data["date"] = pd.to_datetime(data["date"])
    duplicates = data[data.duplicated(["basin_id", "date"], keep=False)]
    if not duplicates.empty:
        sample = duplicates[["basin_id", "date"]].head(5).to_dict("records")
        raise ValueError(f"Duplicate basin-month rows found, examples: {sample}")
    if expected_basin_ids is not None:
        actual = set(data["basin_id"].unique())
        missing_ids = sorted(expected_basin_ids - actual)
        extra_ids = sorted(actual - expected_basin_ids)
        if missing_ids or extra_ids:
            raise ValueError(
                "Basin-month basin IDs do not match selected masks: "
                f"missing={missing_ids[:5]}, extra={extra_ids[:5]}"
            )


def validate_lagged_dataset(df: pd.DataFrame, expected_basin_ids: set[str] | None = None) -> None:
    required = {"date", "basin_id", "basin_name", "target_twsa_cm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Lagged data is missing columns: {sorted(missing)}")
    data = df.copy()
    data["basin_id"] = data["basin_id"].astype(str)
    data["date"] = pd.to_datetime(data["date"])
    duplicates = data[data.duplicated(["basin_id", "date"], keep=False)]
    if not duplicates.empty:
        sample = duplicates[["basin_id", "date"]].head(5).to_dict("records")
        raise ValueError(f"Duplicate lagged rows found, examples: {sample}")
    if expected_basin_ids is not None:
        actual = set(data["basin_id"].unique())
        missing_ids = sorted(expected_basin_ids - actual)
        extra_ids = sorted(actual - expected_basin_ids)
        if missing_ids or extra_ids:
            raise ValueError(
                "Lagged basin IDs do not match selected masks: "
                f"missing={missing_ids[:5]}, extra={extra_ids[:5]}"
            )


def validate_edges(edges: pd.DataFrame, basin_ids: set[str], graph_type: str | None = None) -> None:
    required = {"src_basin_id", "dst_basin_id"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"Edges are missing columns: {sorted(missing)}")
    data = edges.copy()
    data["src_basin_id"] = data["src_basin_id"].astype(str)
    data["dst_basin_id"] = data["dst_basin_id"].astype(str)
    unknown = (set(data["src_basin_id"]) | set(data["dst_basin_id"])) - set(basin_ids)
    if unknown:
        raise ValueError(f"Edges contain basin IDs absent from lagged data: {sorted(unknown)[:10]}")
    self_edges = data[data["src_basin_id"] == data["dst_basin_id"]]
    if not self_edges.empty:
        raise ValueError("Graph edge file contains non-explicit self edges.")
    duplicates = data[data.duplicated(["src_basin_id", "dst_basin_id"], keep=False)]
    if not duplicates.empty:
        sample = duplicates[["src_basin_id", "dst_basin_id"]].head(5).to_dict("records")
        raise ValueError(f"Duplicate graph edges found, examples: {sample}")
    if graph_type is not None and "graph_type" in data.columns:
        values = set(data["graph_type"].astype(str).unique())
        if values != {graph_type}:
            raise ValueError(f"Expected graph_type={graph_type}, found {sorted(values)}")
