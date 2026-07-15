from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import pandas as pd

from grace_gnn.config import (
    DATA_PROCESSED,
    DATA_RAW,
    OUTPUTS,
)
from grace_gnn.validation import file_fingerprint

import run_africa_l3
import run_africa_l3_extra_architectures
import run_africa_l3_gnn_embeddings
import run_africa_l3_grace_only_horizons
import run_africa_l3_walk_forward_top5
import run_ridge_neighbor_residual_mlp_random_control
import make_africa_l3_figures
import make_africa_l3_basin_timeseries_selected_models


CSR_SOURCE = "csr"
CSR_GRACE_NETCDF_NAME = "grace_data.nc"
CSR_EXPERIMENT_REGION = "africa_l3_no_madagascar_csr"
CSR_OUTPUTS = OUTPUTS / CSR_EXPERIMENT_REGION
CSR_BASIN_MONTH_CSV = DATA_PROCESSED / f"basin_month_grace_{CSR_EXPERIMENT_REGION}.csv"
CSR_LAGGED_DATASET_CSV = DATA_PROCESSED / f"lagged_grace_dataset_{CSR_EXPERIMENT_REGION}.csv"
CSR_BASIN_MONTH_PROVENANCE_JSON = DATA_PROCESSED / f"basin_month_grace_{CSR_EXPERIMENT_REGION}.provenance.json"
CSR_LAGGED_DATASET_PROVENANCE_JSON = DATA_PROCESSED / f"lagged_grace_dataset_{CSR_EXPERIMENT_REGION}.provenance.json"


def _csr_region_paths() -> dict[str, Path]:
    return {
        "REGION_OUTPUTS": CSR_OUTPUTS,
        "REGION_PREDICTIONS_CSV": CSR_OUTPUTS / "predictions.csv",
        "REGION_METRICS_OVERALL_CSV": CSR_OUTPUTS / "metrics_overall.csv",
        "REGION_METRICS_BY_BASIN_CSV": CSR_OUTPUTS / "metrics_by_region.csv",
        "REGION_IMPROVEMENT_BY_BASIN_CSV": CSR_OUTPUTS / "improvement_by_region.csv",
        "REGION_PREDICTION_DIAGNOSTICS_CSV": CSR_OUTPUTS / "prediction_diagnostics.csv",
        "REGION_CORRELATION_MATRIX_CSV": CSR_OUTPUTS / "train_region_correlation_matrix.csv",
        "REGION_CORRELATION_PAIRS_CSV": CSR_OUTPUTS / "train_region_correlation_pairs.csv",
    }


def _csr_netcdf_metadata(path: Path) -> dict:
    import xarray as xr

    ds = xr.open_dataset(path, decode_times=False)
    var_name = "lwe_thickness" if "lwe_thickness" in ds.data_vars else next(iter(ds.data_vars))
    time = ds["time"]
    lat = ds["lat"].values
    lon = ds["lon"].values
    return {
        "grace_source": CSR_SOURCE,
        "selected_variable": var_name,
        "selected_variable_units": ds[var_name].attrs.get("units") or ds[var_name].attrs.get("Units"),
        "time_units": time.attrs.get("units") or time.attrs.get("Units"),
        "time_decode_policy": "manual CF-style numeric days since origin; lag builder normalizes to month start",
        "grid": {
            "lat_count": int(ds.sizes["lat"]),
            "lon_count": int(ds.sizes["lon"]),
            "time_count": int(ds.sizes["time"]),
            "lat_step_degrees": float(abs(lat[1] - lat[0])),
            "lon_step_degrees": float(abs(lon[1] - lon[0])),
        },
    }


def _csr_basin_month_provenance(grace_nc: Path, mask_zip: Path, members: pd.DataFrame) -> dict:
    return {
        "experiment_region": CSR_EXPERIMENT_REGION,
        "grace_netcdf": file_fingerprint(grace_nc),
        "grace_metadata": _csr_netcdf_metadata(grace_nc),
        "mask_zips": [file_fingerprint(mask_zip)],
        "mask_format": "HydroBASINS .mask.csv/.mask.xyz members",
        "aggregation": "mask weight times cos(latitude), nearest CSR grid cell per mask cell",
        "basin_name_exclude": "madagascar",
        "basin_count": int(members["basin_id"].nunique()),
        "basin_ids": sorted(members["basin_id"].astype(str).unique()),
    }


def _patch_common(module) -> None:
    module.EXPERIMENT_REGION = CSR_EXPERIMENT_REGION
    module.LAGGED_DATASET_CSV = CSR_LAGGED_DATASET_CSV
    for name, path in _csr_region_paths().items():
        if hasattr(module, name):
            setattr(module, name, path)


def _patch_main_runner() -> None:
    _patch_common(run_africa_l3)
    run_africa_l3.GRACE_NETCDF_NAME = CSR_GRACE_NETCDF_NAME
    run_africa_l3.BASIN_MONTH_CSV = CSR_BASIN_MONTH_CSV
    run_africa_l3.BASIN_MONTH_PROVENANCE_JSON = CSR_BASIN_MONTH_PROVENANCE_JSON
    run_africa_l3.LAGGED_DATASET_PROVENANCE_JSON = CSR_LAGGED_DATASET_PROVENANCE_JSON
    run_africa_l3._basin_month_provenance = _csr_basin_month_provenance


def _patch_extra_runner() -> None:
    _patch_common(run_africa_l3_extra_architectures)


def _patch_embedding_runner() -> None:
    _patch_common(run_africa_l3_gnn_embeddings)


def _patch_random_control_runner() -> None:
    _patch_common(run_ridge_neighbor_residual_mlp_random_control)


def _patch_figure_runners() -> None:
    _patch_common(make_africa_l3_figures)
    make_africa_l3_figures.REGION_FIGURES = CSR_OUTPUTS / "figures"
    _patch_common(make_africa_l3_basin_timeseries_selected_models)


def _patch_horizon_runner() -> None:
    _patch_common(run_africa_l3_grace_only_horizons)
    run_africa_l3_grace_only_horizons.BASIN_MONTH_CSV = CSR_BASIN_MONTH_CSV
    run_africa_l3_grace_only_horizons.OUTPUT_DIR = CSR_OUTPUTS / "grace_only_horizons"
    run_africa_l3_grace_only_horizons.HORIZON_DATASET_CSV = (
        DATA_PROCESSED / f"grace_horizon_dataset_{CSR_EXPERIMENT_REGION}.csv"
    )
    run_africa_l3_grace_only_horizons.OLD_LAGGED_DATASET_CSV = CSR_LAGGED_DATASET_CSV


def _patch_walk_forward_runner() -> None:
    _patch_common(run_africa_l3_walk_forward_top5)
    run_africa_l3_walk_forward_top5.OUTPUT_DIR = CSR_OUTPUTS / "walk_forward_top5"


def _run_main() -> None:
    basin_month = run_africa_l3.build_basin_month(force=True)
    lagged = run_africa_l3.build_lagged(basin_month)
    predictions = run_africa_l3.train_baselines(lagged)
    predictions = run_africa_l3.train_gnns(lagged, predictions)
    run_africa_l3.save_metrics(predictions)


def main(
    run_extra: bool = True,
    run_random_control: bool = True,
    run_embeddings: bool = True,
    run_horizons: bool = True,
    run_walk_forward: bool = True,
    run_figures: bool = True,
) -> None:
    grace_nc = DATA_RAW / CSR_GRACE_NETCDF_NAME
    if not grace_nc.exists():
        raise FileNotFoundError(f"Missing CSR GRACE NetCDF: {grace_nc}")
    CSR_OUTPUTS.mkdir(parents=True, exist_ok=True)

    _patch_main_runner()
    _run_main()

    if run_extra:
        _patch_extra_runner()
        run_africa_l3_extra_architectures.main()
    if run_random_control:
        _patch_random_control_runner()
        run_ridge_neighbor_residual_mlp_random_control.main()
    if run_embeddings:
        _patch_embedding_runner()
        run_africa_l3_gnn_embeddings.main()
    if run_horizons:
        _patch_horizon_runner()
        run_africa_l3_grace_only_horizons.main()
    if run_walk_forward:
        _patch_walk_forward_runner()
        run_africa_l3_walk_forward_top5.main()
    if run_figures:
        _patch_figure_runners()
        make_africa_l3_figures.main()
        make_africa_l3_basin_timeseries_selected_models.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Africa L3 no-Madagascar forecast benchmark on CSR GRACE data.")
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="Only run CSR preprocessing, baselines, and baseline GNNs.",
    )
    parser.add_argument("--skip-horizons", action="store_true", help="Skip the 1-6 month CSR horizon benchmark.")
    parser.add_argument("--skip-walk-forward", action="store_true", help="Skip the CSR walk-forward robustness benchmark.")
    parser.add_argument("--skip-figures", action="store_true", help="Skip CSR figure generation.")
    args = parser.parse_args()
    main(
        run_extra=not args.main_only,
        run_random_control=not args.main_only,
        run_embeddings=not args.main_only,
        run_horizons=not args.main_only and not args.skip_horizons,
        run_walk_forward=not args.main_only and not args.skip_walk_forward,
        run_figures=not args.main_only and not args.skip_figures,
    )
