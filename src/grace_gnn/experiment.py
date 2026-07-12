from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .config import (
    AFRICA_L3_MASK_ZIP_NAME,
    DATA_PROCESSED,
    OUTPUTS,
    ROOT,
)


def slugify_experiment(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    if not slug:
        raise ValueError("Experiment name must contain at least one letter or number.")
    return slug


@dataclass(frozen=True)
class ExperimentPaths:
    name: str
    output_dir: Path
    figures_dir: Path
    basin_month_csv: Path
    basin_month_provenance_json: Path
    lagged_dataset_csv: Path
    lagged_dataset_provenance_json: Path
    predictions_csv: Path
    metrics_overall_csv: Path
    metrics_by_basin_csv: Path
    improvement_by_basin_csv: Path
    prediction_diagnostics_csv: Path
    correlation_matrix_csv: Path
    correlation_pairs_csv: Path

    @classmethod
    def from_name(cls, name: str) -> "ExperimentPaths":
        name = slugify_experiment(name)
        output_dir = OUTPUTS / name
        return cls(
            name=name,
            output_dir=output_dir,
            figures_dir=output_dir / "figures",
            basin_month_csv=DATA_PROCESSED / f"basin_month_grace_{name}.csv",
            basin_month_provenance_json=DATA_PROCESSED / f"basin_month_grace_{name}.provenance.json",
            lagged_dataset_csv=DATA_PROCESSED / f"lagged_grace_dataset_{name}.csv",
            lagged_dataset_provenance_json=DATA_PROCESSED / f"lagged_grace_dataset_{name}.provenance.json",
            predictions_csv=output_dir / "predictions.csv",
            metrics_overall_csv=output_dir / "metrics_overall.csv",
            metrics_by_basin_csv=output_dir / "metrics_by_region.csv",
            improvement_by_basin_csv=output_dir / "improvement_by_region.csv",
            prediction_diagnostics_csv=output_dir / "prediction_diagnostics.csv",
            correlation_matrix_csv=output_dir / "train_region_correlation_matrix.csv",
            correlation_pairs_csv=output_dir / "train_region_correlation_pairs.csv",
        )

    def ensure_dirs(self) -> None:
        for path in [DATA_PROCESSED, OUTPUTS, self.output_dir, self.figures_dir]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class MaskExperiment:
    paths: ExperimentPaths
    mask_zip: Path
    basin_name_filter: str | None = None
    basin_name_exclude: str | None = None
    strict_mask_names: bool = False

    @classmethod
    def africa_l3_default(cls) -> "MaskExperiment":
        return cls(
            paths=ExperimentPaths.from_name("africa_l3_no_madagascar"),
            mask_zip=ROOT / "masks" / AFRICA_L3_MASK_ZIP_NAME,
            basin_name_exclude="madagascar",
            strict_mask_names=True,
        )
