from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import pandas as pd

from grace_gnn.config import (
    AFRICA_L3_MASK_ZIP_NAME,
    BASIN_MONTH_PROVENANCE_JSON,
    DATA_RAW,
    ERA5_BASIN_MONTH_CSV,
    ERA5_BASIN_MONTH_PROVENANCE_JSON,
    ERA5_NETCDF_NAME,
    EXPERIMENT_REGION,
    LAGGED_DATASET_CSV,
    LAGGED_DATASET_PROVENANCE_JSON,
    LAGGED_GRACE_ERA5_DATASET_CSV,
    LAGGED_GRACE_ERA5_DATASET_PROVENANCE_JSON,
    LAGS,
    ensure_dirs,
)
from grace_gnn.data import aggregate_era5_netcdf_to_mask_zips, list_mask_members
from grace_gnn.features import filter_region, make_lagged_grace_era5_dataset
from grace_gnn.validation import (
    file_fingerprint,
    read_json,
    validate_lagged_dataset,
    validate_unique_mask_members,
    write_json,
)


def _l3_mask_zip() -> Path:
    path = ROOT / "masks" / AFRICA_L3_MASK_ZIP_NAME
    if not path.exists():
        raise FileNotFoundError(f"Missing L3 mask zip: {path}")
    return path


def _era5_nc() -> Path:
    path = DATA_RAW / ERA5_NETCDF_NAME
    if not path.exists():
        raise FileNotFoundError(f"Missing ERA5 NetCDF: {path}")
    return path


def _selected_l3_mask_members() -> pd.DataFrame:
    members = list_mask_members([_l3_mask_zip()], strict=True)
    members = members[~members["basin_name"].str.contains("madagascar", case=False, na=False)].copy()
    validate_unique_mask_members(members)
    return members


def _era5_basin_month_provenance(era5_nc: Path, mask_zip: Path, members: pd.DataFrame) -> dict:
    return {
        "experiment_region": EXPERIMENT_REGION,
        "era5_netcdf": file_fingerprint(era5_nc),
        "mask_zips": [file_fingerprint(mask_zip)],
        "mask_format": "HydroBASINS .mask.csv/.mask.xyz members",
        "basin_name_exclude": "madagascar",
        "basin_count": int(members["basin_id"].nunique()),
        "basin_ids": sorted(members["basin_id"].astype(str).unique()),
        "variables": {
            "era5_tp_mm": {
                "source": "tp",
                "source_units": "m",
                "output_units": "mm",
                "conversion": "tp * 1000",
            },
            "era5_ro_mm": {
                "source": "ro",
                "source_units": "m",
                "output_units": "mm",
                "conversion": "ro * 1000",
            },
            "era5_evap_mm": {
                "source": "e",
                "source_units": "m",
                "output_units": "mm",
                "sign_convention": "positive loss magnitude",
                "conversion": "-e * 1000",
            },
        },
        "aggregation": "mask weight times cos(latitude), nearest ERA5 grid cell per mask cell",
    }


def _joined_provenance(era5_provenance: dict) -> dict:
    return {
        "experiment_region": EXPERIMENT_REGION,
        "source_lagged_grace_dataset": file_fingerprint(LAGGED_DATASET_CSV),
        "source_lagged_grace_provenance": read_json(LAGGED_DATASET_PROVENANCE_JSON),
        "source_grace_basin_month_provenance": read_json(BASIN_MONTH_PROVENANCE_JSON),
        "source_era5_basin_month": file_fingerprint(ERA5_BASIN_MONTH_CSV),
        "source_era5_basin_month_provenance": era5_provenance,
        "join_keys": ["basin_id", "date"],
        "target_month_era5_included": False,
        "era5_lags": LAGS,
        "era5_predictor_columns": [
            f"era5_{variable}_lag_{lag}"
            for variable in ["tp", "ro", "evap"]
            for lag in LAGS
        ],
        "missing_joined_rows_allowed": False,
    }


def build_era5_basin_month() -> pd.DataFrame:
    ensure_dirs()
    era5_nc = _era5_nc()
    mask_zip = _l3_mask_zip()
    members = _selected_l3_mask_members()
    expected_basin_ids = set(members["basin_id"].astype(str))
    basin_month = aggregate_era5_netcdf_to_mask_zips(
        era5_nc,
        [mask_zip],
        ERA5_BASIN_MONTH_CSV,
        basin_name_exclude="madagascar",
    )
    basin_month = filter_region(basin_month, EXPERIMENT_REGION)
    basin_month["date"] = pd.to_datetime(basin_month["date"]).dt.to_period("M").dt.to_timestamp()
    duplicates = basin_month[basin_month.duplicated(["basin_id", "date"], keep=False)]
    if not duplicates.empty:
        sample = duplicates[["basin_id", "date"]].head(5).to_dict("records")
        raise ValueError(f"Duplicate ERA5 basin-month rows found, examples: {sample}")
    actual_basin_ids = set(basin_month["basin_id"].astype(str).unique())
    if actual_basin_ids != expected_basin_ids:
        raise ValueError(
            "ERA5 basin IDs do not match selected masks: "
            f"missing={sorted(expected_basin_ids - actual_basin_ids)[:5]}, "
            f"extra={sorted(actual_basin_ids - expected_basin_ids)[:5]}"
        )
    basin_month.to_csv(ERA5_BASIN_MONTH_CSV, index=False)
    provenance = _era5_basin_month_provenance(era5_nc, mask_zip, members)
    write_json(ERA5_BASIN_MONTH_PROVENANCE_JSON, provenance)
    print(f"Saved {len(basin_month):,} ERA5 basin-month rows to {ERA5_BASIN_MONTH_CSV}")
    return basin_month


def build_lagged_grace_era5(basin_month_era5: pd.DataFrame) -> pd.DataFrame:
    if not LAGGED_DATASET_CSV.exists():
        raise FileNotFoundError(f"Missing existing GRACE lagged dataset: {LAGGED_DATASET_CSV}")
    lagged_grace = pd.read_csv(LAGGED_DATASET_CSV, parse_dates=["date"])
    validate_lagged_dataset(lagged_grace)
    joined = make_lagged_grace_era5_dataset(
        lagged_grace,
        basin_month_era5,
        LAGS,
        LAGGED_GRACE_ERA5_DATASET_CSV,
    )
    validate_lagged_dataset(
        joined,
        expected_basin_ids=set(lagged_grace["basin_id"].astype(str).unique()),
    )
    if len(joined) != len(lagged_grace):
        raise ValueError(f"Joined row count changed from {len(lagged_grace)} to {len(joined)}")
    era5_provenance = read_json(ERA5_BASIN_MONTH_PROVENANCE_JSON) or {}
    write_json(LAGGED_GRACE_ERA5_DATASET_PROVENANCE_JSON, _joined_provenance(era5_provenance))
    print(f"Saved {len(joined):,} joined GRACE+ERA5 lagged rows to {LAGGED_GRACE_ERA5_DATASET_CSV}")
    return joined


def main() -> None:
    basin_month_era5 = build_era5_basin_month()
    build_lagged_grace_era5(basin_month_era5)


if __name__ == "__main__":
    main()
