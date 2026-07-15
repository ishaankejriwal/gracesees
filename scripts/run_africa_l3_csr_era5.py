from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from grace_gnn.config import DATA_PROCESSED, OUTPUTS

import preprocess_africa_l3_era5
import run_africa_l3_era5_one_month
import run_africa_l3_era5_heavy_architectures
import run_africa_l3_era5_svr


CSR_EXPERIMENT_REGION = "africa_l3_no_madagascar_csr"
CSR_OUTPUTS = OUTPUTS / CSR_EXPERIMENT_REGION

CSR_BASIN_MONTH_GRACE_CSV = DATA_PROCESSED / f"basin_month_grace_{CSR_EXPERIMENT_REGION}.csv"
CSR_BASIN_MONTH_GRACE_PROVENANCE_JSON = DATA_PROCESSED / f"basin_month_grace_{CSR_EXPERIMENT_REGION}.provenance.json"
CSR_LAGGED_GRACE_CSV = DATA_PROCESSED / f"lagged_grace_dataset_{CSR_EXPERIMENT_REGION}.csv"
CSR_LAGGED_GRACE_PROVENANCE_JSON = DATA_PROCESSED / f"lagged_grace_dataset_{CSR_EXPERIMENT_REGION}.provenance.json"

CSR_ERA5_BASIN_MONTH_CSV = DATA_PROCESSED / f"basin_month_era5_{CSR_EXPERIMENT_REGION}.csv"
CSR_ERA5_BASIN_MONTH_PROVENANCE_JSON = DATA_PROCESSED / f"basin_month_era5_{CSR_EXPERIMENT_REGION}.provenance.json"
CSR_LAGGED_GRACE_ERA5_CSV = DATA_PROCESSED / f"lagged_grace_era5_dataset_{CSR_EXPERIMENT_REGION}.csv"
CSR_LAGGED_GRACE_ERA5_PROVENANCE_JSON = (
    DATA_PROCESSED / f"lagged_grace_era5_dataset_{CSR_EXPERIMENT_REGION}.provenance.json"
)

ONE_MONTH_OUTPUT_DIR = CSR_OUTPUTS / "era5_one_month"
HEAVY_OUTPUT_DIR = CSR_OUTPUTS / "era5_heavy_architectures"
SVR_OUTPUT_DIR = CSR_OUTPUTS / "era5_svr"

EXPECTED_CSR_ROWS = 6771
EXPECTED_CSR_BASINS = 37


def _patch_preprocess() -> None:
    preprocess_africa_l3_era5.EXPERIMENT_REGION = CSR_EXPERIMENT_REGION
    preprocess_africa_l3_era5.BASIN_MONTH_PROVENANCE_JSON = CSR_BASIN_MONTH_GRACE_PROVENANCE_JSON
    preprocess_africa_l3_era5.LAGGED_DATASET_CSV = CSR_LAGGED_GRACE_CSV
    preprocess_africa_l3_era5.LAGGED_DATASET_PROVENANCE_JSON = CSR_LAGGED_GRACE_PROVENANCE_JSON
    preprocess_africa_l3_era5.ERA5_BASIN_MONTH_CSV = CSR_ERA5_BASIN_MONTH_CSV
    preprocess_africa_l3_era5.ERA5_BASIN_MONTH_PROVENANCE_JSON = CSR_ERA5_BASIN_MONTH_PROVENANCE_JSON
    preprocess_africa_l3_era5.LAGGED_GRACE_ERA5_DATASET_CSV = CSR_LAGGED_GRACE_ERA5_CSV
    preprocess_africa_l3_era5.LAGGED_GRACE_ERA5_DATASET_PROVENANCE_JSON = CSR_LAGGED_GRACE_ERA5_PROVENANCE_JSON


def _patch_one_month() -> None:
    run_africa_l3_era5_one_month.LAGGED_DATASET_CSV = CSR_LAGGED_GRACE_CSV
    run_africa_l3_era5_one_month.LAGGED_GRACE_ERA5_DATASET_CSV = CSR_LAGGED_GRACE_ERA5_CSV
    run_africa_l3_era5_one_month.REGION_METRICS_OVERALL_CSV = CSR_OUTPUTS / "metrics_overall.csv"
    run_africa_l3_era5_one_month.OUTPUT_DIR = ONE_MONTH_OUTPUT_DIR
    run_africa_l3_era5_one_month.PREDICTIONS_CSV = ONE_MONTH_OUTPUT_DIR / "predictions.csv"
    run_africa_l3_era5_one_month.METRICS_OVERALL_CSV = ONE_MONTH_OUTPUT_DIR / "metrics_overall.csv"
    run_africa_l3_era5_one_month.METRICS_BY_REGION_CSV = ONE_MONTH_OUTPUT_DIR / "metrics_by_region.csv"
    run_africa_l3_era5_one_month.PREDICTION_DIAGNOSTICS_CSV = ONE_MONTH_OUTPUT_DIR / "prediction_diagnostics.csv"
    run_africa_l3_era5_one_month.SUMMARY_CSV = ONE_MONTH_OUTPUT_DIR / "era5_vs_grace_only_summary.csv"
    run_africa_l3_era5_one_month.VALIDATION_JSON = ONE_MONTH_OUTPUT_DIR / "run_validation.json"
    run_africa_l3_era5_one_month.EXPECTED_ROWS = EXPECTED_CSR_ROWS
    run_africa_l3_era5_one_month.EXPECTED_BASINS = EXPECTED_CSR_BASINS
    run_africa_l3_era5_one_month.GRACE_ONLY_BASELINE_FALLBACK_RMSE_CM = 3.35325865215903


def _patch_heavy() -> None:
    run_africa_l3_era5_heavy_architectures.LAGGED_GRACE_ERA5_DATASET_CSV = CSR_LAGGED_GRACE_ERA5_CSV
    run_africa_l3_era5_heavy_architectures.ERA5_BASELINE_PREDICTIONS_CSV = ONE_MONTH_OUTPUT_DIR / "predictions.csv"
    run_africa_l3_era5_heavy_architectures.era5_feature_columns = run_africa_l3_era5_one_month.era5_feature_columns
    run_africa_l3_era5_heavy_architectures.make_source_degree_matched_random_edges = (
        run_africa_l3_era5_one_month.make_source_degree_matched_random_edges
    )
    run_africa_l3_era5_heavy_architectures.validate_inputs = run_africa_l3_era5_one_month.validate_inputs
    run_africa_l3_era5_heavy_architectures.OUTPUT_DIR = HEAVY_OUTPUT_DIR
    run_africa_l3_era5_heavy_architectures.PREDICTIONS_CSV = HEAVY_OUTPUT_DIR / "predictions.csv"
    run_africa_l3_era5_heavy_architectures.METRICS_OVERALL_CSV = HEAVY_OUTPUT_DIR / "metrics_overall.csv"
    run_africa_l3_era5_heavy_architectures.METRICS_BY_REGION_CSV = HEAVY_OUTPUT_DIR / "metrics_by_region.csv"
    run_africa_l3_era5_heavy_architectures.PREDICTION_DIAGNOSTICS_CSV = HEAVY_OUTPUT_DIR / "prediction_diagnostics.csv"
    run_africa_l3_era5_heavy_architectures.SUMMARY_CSV = HEAVY_OUTPUT_DIR / "heavy_architecture_summary.csv"
    run_africa_l3_era5_heavy_architectures.VALIDATION_JSON = HEAVY_OUTPUT_DIR / "run_validation.json"


def _patch_svr() -> None:
    run_africa_l3_era5_svr.LAGGED_GRACE_ERA5_DATASET_CSV = CSR_LAGGED_GRACE_ERA5_CSV
    run_africa_l3_era5_svr.EXPECTED_ROWS = EXPECTED_CSR_ROWS
    run_africa_l3_era5_svr.EXPECTED_BASINS = EXPECTED_CSR_BASINS
    run_africa_l3_era5_svr.era5_feature_columns = run_africa_l3_era5_one_month.era5_feature_columns
    run_africa_l3_era5_svr.make_source_degree_matched_random_edges = (
        run_africa_l3_era5_one_month.make_source_degree_matched_random_edges
    )
    run_africa_l3_era5_svr.split_signature = run_africa_l3_era5_one_month.split_signature
    run_africa_l3_era5_svr.validate_inputs = run_africa_l3_era5_one_month.validate_inputs
    run_africa_l3_era5_svr.OUTPUT_DIR = SVR_OUTPUT_DIR
    run_africa_l3_era5_svr.PREDICTIONS_CSV = SVR_OUTPUT_DIR / "predictions.csv"
    run_africa_l3_era5_svr.METRICS_OVERALL_CSV = SVR_OUTPUT_DIR / "metrics_overall.csv"
    run_africa_l3_era5_svr.METRICS_BY_REGION_CSV = SVR_OUTPUT_DIR / "metrics_by_region.csv"
    run_africa_l3_era5_svr.PREDICTION_DIAGNOSTICS_CSV = SVR_OUTPUT_DIR / "prediction_diagnostics.csv"
    run_africa_l3_era5_svr.SUMMARY_CSV = SVR_OUTPUT_DIR / "svr_summary.csv"
    run_africa_l3_era5_svr.VALIDATION_JSON = SVR_OUTPUT_DIR / "run_validation.json"
    run_africa_l3_era5_svr.BEST_ERA5_BASELINE_RMSE_CM = 0.0


def patch_all() -> None:
    _patch_preprocess()
    _patch_one_month()
    _patch_heavy()
    _patch_svr()


def main(run_heavy: bool = True, run_svr: bool = True) -> None:
    if not CSR_LAGGED_GRACE_CSV.exists():
        raise FileNotFoundError(f"Missing CSR lagged GRACE dataset: {CSR_LAGGED_GRACE_CSV}")
    if not (CSR_OUTPUTS / "metrics_overall.csv").exists():
        raise FileNotFoundError(f"Missing CSR GRACE-only metrics: {CSR_OUTPUTS / 'metrics_overall.csv'}")

    patch_all()
    preprocess_africa_l3_era5.main()
    run_africa_l3_era5_one_month.main()

    best_era5 = run_africa_l3_era5_one_month.pd.read_csv(
        ONE_MONTH_OUTPUT_DIR / "metrics_overall.csv"
    )
    best_test_rmse = float(best_era5[best_era5["split"].eq("test")]["rmse_cm"].min())
    run_africa_l3_era5_svr.BEST_ERA5_BASELINE_RMSE_CM = best_test_rmse

    if run_heavy:
        run_africa_l3_era5_heavy_architectures.main()
    if run_svr:
        run_africa_l3_era5_svr.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CSR Africa L3 ERA5 add-on workflow.")
    parser.add_argument("--skip-heavy", action="store_true", help="Skip heavier ERA5 neural/tree architectures.")
    parser.add_argument("--skip-svr", action="store_true", help="Skip ERA5 SVR follow-up models.")
    args = parser.parse_args()
    main(run_heavy=not args.skip_heavy, run_svr=not args.skip_svr)
