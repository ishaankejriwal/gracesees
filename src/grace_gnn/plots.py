from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def model_label(model_name: str, graph_type: str) -> str:
    return model_name if graph_type == "none" else f"{model_name}\n{graph_type}"


def choose_main_basin(predictions: pd.DataFrame) -> str:
    names = predictions[["basin_id", "basin_name"]].drop_duplicates()
    amazon = names[names["basin_name"].astype(str).str.contains("amazon", case=False, na=False)]
    if not amazon.empty:
        return str(amazon.iloc[0]["basin_id"])
    counts = predictions.groupby("basin_id").size().sort_values(ascending=False)
    return str(counts.index[0])


def plot_time_series(predictions: pd.DataFrame, output_path: Path, basin_id: str | None = None, split: str = "test") -> None:
    data = predictions[predictions["split"] == split].copy()
    if data.empty:
        return
    basin_id = basin_id or choose_main_basin(data)
    data = data[data["basin_id"].astype(str) == str(basin_id)]
    fig, ax = plt.subplots(figsize=(11, 5))
    obs = data.drop_duplicates("date").sort_values("date")
    ax.plot(obs["date"], obs["observed_twsa_cm"], color="black", linewidth=2, label="Observed")
    for (model_name, graph_type), group in data.groupby(["model_name", "graph_type"], dropna=False):
        group = group.sort_values("date")
        ax.plot(group["date"], group["predicted_twsa_cm"], label=model_label(model_name, graph_type), alpha=0.85)
    name = data["basin_name"].dropna().iloc[0] if data["basin_name"].notna().any() else basin_id
    ax.set_title(f"Observed vs predicted GRACE TWSA, {name}")
    ax.set_ylabel("TWSA (cm)")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_rmse_bar(metrics_overall: pd.DataFrame, output_path: Path, split: str = "test") -> None:
    data = metrics_overall[metrics_overall["split"] == split].sort_values("rmse_cm")
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [model_label(r.model_name, r.graph_type) for r in data.itertuples(index=False)]
    ax.bar(labels, data["rmse_cm"], color="#4477AA")
    ax.set_ylabel("RMSE (cm)")
    ax.set_title("Test RMSE by model")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_observed_vs_predicted(predictions: pd.DataFrame, output_path: Path, split: str = "test") -> None:
    data = predictions[predictions["split"] == split]
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    for (model_name, graph_type), group in data.groupby(["model_name", "graph_type"], dropna=False):
        ax.scatter(
            group["observed_twsa_cm"],
            group["predicted_twsa_cm"],
            s=12,
            alpha=0.45,
            label=model_label(model_name, graph_type),
        )
    mn = min(data["observed_twsa_cm"].min(), data["predicted_twsa_cm"].min())
    mx = max(data["observed_twsa_cm"].max(), data["predicted_twsa_cm"].max())
    ax.plot([mn, mx], [mn, mx], color="black", linewidth=1)
    ax.set_xlabel("Observed TWSA (cm)")
    ax.set_ylabel("Predicted TWSA (cm)")
    ax.set_title("Observed vs predicted")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_graph_edges(edges: pd.DataFrame, basin_names: pd.DataFrame, output_path: Path) -> None:
    if edges.empty:
        return
    import numpy as np

    nodes = sorted(set(edges["src_basin_id"].astype(str)) | set(edges["dst_basin_id"].astype(str)))
    names = basin_names.copy()
    names["basin_id"] = names["basin_id"].astype(str)
    name_map = dict(zip(names["basin_id"], names["basin_name"]))
    theta = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    pos = {node: (np.cos(t), np.sin(t)) for node, t in zip(nodes, theta)}
    fig, ax = plt.subplots(figsize=(7, 7))
    for row in edges.itertuples(index=False):
        src = str(row.src_basin_id)
        dst = str(row.dst_basin_id)
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "->", "color": "#777777", "lw": 1, "alpha": 0.65},
        )
    for node, (x, y) in pos.items():
        ax.scatter([x], [y], s=180, color="#4477AA", zorder=3)
        label = str(name_map.get(node, node))
        ax.text(x * 1.12, y * 1.12, label, ha="center", va="center", fontsize=8)
    graph_type = edges["graph_type"].iloc[0] if "graph_type" in edges.columns and len(edges) else "graph"
    ax.set_title(f"Level 2 region graph: {graph_type}")
    ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_improvement_by_basin(improvement: pd.DataFrame, output_path: Path) -> None:
    col = "rmse_improvement_real_vs_basin_only_cm"
    if improvement.empty or col not in improvement.columns:
        return
    data = improvement.sort_values(col, ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = data["basin_name"].astype(str).where(data["basin_name"].notna(), data["basin_id"].astype(str))
    ax.bar(labels, data[col], color="#228833")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("RMSE improvement (cm)")
    ax.set_title("Real-neighbor GNN improvement over basin-only NN")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
