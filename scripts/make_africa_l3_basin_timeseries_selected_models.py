from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from grace_gnn.config import REGION_OUTPUTS


SELECTED_MODELS = [
    ("ridge_neighbor_residual_mlp", "corr_top3_directed", "neighbor residual MLP"),
    ("ridge_residual_mlp", "own_lags", "own-lag residual MLP"),
    ("xgboost_gnn_embedding_residual", "corr_top3_directed", "GNN emb + XGBoost"),
    ("random_forest_ar", "none", "random forest"),
    ("xgboost_ar", "none", "XGBoost"),
    ("persistence", "none", "persistence"),
]


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "basin"


def selected_model_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = pd.DataFrame(
        [(model_name, graph_type) for model_name, graph_type, _ in SELECTED_MODELS],
        columns=["model_name", "graph_type"],
    )
    return predictions.merge(keys, on=["model_name", "graph_type"], how="inner")


def model_label(model_name: str, graph_type: str) -> str:
    for selected_model, selected_graph, label in SELECTED_MODELS:
        if model_name == selected_model and graph_type == selected_graph:
            return label
    return model_name if graph_type in {"none", "own_lags"} else f"{model_name} ({graph_type})"


def main() -> None:
    predictions = pd.read_csv(REGION_OUTPUTS / "predictions.csv", parse_dates=["date"])
    data = selected_model_frame(predictions[predictions["split"].eq("test")].copy())
    if data.empty:
        raise ValueError("No selected model predictions found in the test split.")

    output_dir = REGION_OUTPUTS / "figures" / "basin_timeseries_selected_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    for basin_id, group in data.groupby("basin_id", sort=True):
        group = group.sort_values("date")
        observed = group.drop_duplicates("date")
        basin_name = str(observed["basin_name"].iloc[0])

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            observed["date"],
            observed["observed_twsa_cm"],
            color="black",
            linewidth=2.2,
            label="observed",
        )
        for model_name, graph_type, _ in SELECTED_MODELS:
            model_rows = group[group["model_name"].eq(model_name) & group["graph_type"].eq(graph_type)]
            if model_rows.empty:
                continue
            ax.plot(
                model_rows["date"],
                model_rows["predicted_twsa_cm"],
                linewidth=1.3,
                alpha=0.85,
                label=model_label(model_name, graph_type),
            )

        ax.set_title(f"{basin_name} test time series")
        ax.set_ylabel("TWSA (cm)")
        ax.set_xlabel("Date")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        filename = f"{safe_name(str(basin_id))}_{safe_name(basin_name)}.png"
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    print(f"Saved {data['basin_id'].nunique()} basin time-series figures to {output_dir}")


if __name__ == "__main__":
    main()
