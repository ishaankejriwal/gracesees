from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
EXPERIMENT_REGION = "africa_l3_no_madagascar"
REGION_OUTPUTS = OUTPUTS / EXPERIMENT_REGION
REGION_FIGURES = REGION_OUTPUTS / "figures"

BASIN_MONTH_CSV = DATA_PROCESSED / f"basin_month_grace_{EXPERIMENT_REGION}.csv"
LAGGED_DATASET_CSV = DATA_PROCESSED / f"lagged_grace_dataset_{EXPERIMENT_REGION}.csv"
BASIN_MONTH_PROVENANCE_JSON = DATA_PROCESSED / f"basin_month_grace_{EXPERIMENT_REGION}.provenance.json"
LAGGED_DATASET_PROVENANCE_JSON = DATA_PROCESSED / f"lagged_grace_dataset_{EXPERIMENT_REGION}.provenance.json"
REAL_EDGES_CSV = DATA_PROCESSED / "edges_real.csv"
RANDOM_EDGES_CSV = DATA_PROCESSED / "edges_random.csv"

PREDICTIONS_CSV = OUTPUTS / "predictions.csv"
METRICS_OVERALL_CSV = OUTPUTS / "metrics_overall.csv"
METRICS_BY_BASIN_CSV = OUTPUTS / "metrics_by_basin.csv"
IMPROVEMENT_BY_BASIN_CSV = OUTPUTS / "improvement_by_basin.csv"

REGION_PREDICTIONS_CSV = REGION_OUTPUTS / "predictions.csv"
REGION_METRICS_OVERALL_CSV = REGION_OUTPUTS / "metrics_overall.csv"
REGION_METRICS_BY_BASIN_CSV = REGION_OUTPUTS / "metrics_by_region.csv"
REGION_IMPROVEMENT_BY_BASIN_CSV = REGION_OUTPUTS / "improvement_by_region.csv"
REGION_PREDICTION_DIAGNOSTICS_CSV = REGION_OUTPUTS / "prediction_diagnostics.csv"
REGION_CORRELATION_MATRIX_CSV = REGION_OUTPUTS / "train_region_correlation_matrix.csv"
REGION_CORRELATION_PAIRS_CSV = REGION_OUTPUTS / "train_region_correlation_pairs.csv"

LAGS = [1, 2, 3, 6, 12]
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.10
TEST_FRACTION = 0.20
TRAIN_END = "2016-12-31"
VAL_END = "2019-12-31"
RANDOM_SEED = 42
PODAAC_GRACE_SHORT_NAME = "TELLUS_GRAC-GRFO_MASCON_GRID_RL06.3_V4"

AFRICA_L2_NO_MADAGASCAR_BASIN_NAMES = [
    "Greater Nile Coastal",
    "Southern Africa Coastal",
    "West Africa Coastal",
    "South Central Africa Coastal",
    "North Africa Coastal",
    "East Africa Coastal",
    "Chad Endorheic",
]

AFRICA_L3_MASK_ZIP_NAME = "L3-20260709T200427Z-2-001.zip"

def ensure_dirs() -> None:
    for path in [DATA_RAW, DATA_PROCESSED, OUTPUTS, FIGURES, REGION_OUTPUTS, REGION_FIGURES]:
        path.mkdir(parents=True, exist_ok=True)
