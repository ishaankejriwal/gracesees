from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grace_gnn.config import LAGGED_DATASET_CSV, REGION_FIGURES, REGION_OUTPUTS
from grace_gnn.splits import chronological_fraction_split


TOP_MODEL_KEYS = [
    ("ridge_neighbor_residual_mlp", "corr_top3_directed"),
    ("ridge_residual_mlp", "own_lags"),
    ("ridge_neighbor_ar", "corr_top3_directed"),
    ("ridge_ar", "none"),
    ("basin_only_nn", "none"),
    ("residual_neighbor_gnn", "random_degree_matched"),
    ("residual_neighbor_gnn", "real_knn_undirected"),
    ("persistence", "none"),
]


def model_label(model_name: str, graph_type: str) -> str:
    if graph_type in {"none", "own_lags"}:
        return model_name
    return f"{model_name}\n{graph_type}"


def model_key_frame(df: pd.DataFrame, keys: list[tuple[str, str]]) -> pd.DataFrame:
    keep = pd.DataFrame(keys, columns=["model_name", "graph_type"])
    return df.merge(keep, on=["model_name", "graph_type"], how="inner")


def plot_rmse_bar(metrics: pd.DataFrame) -> None:
    data = metrics[metrics["split"].eq("test")].sort_values("rmse_cm").head(14)
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [model_label(r.model_name, r.graph_type) for r in data.itertuples(index=False)]
    ax.bar(labels, data["rmse_cm"], color="#4477AA")
    ax.set_ylabel("Test RMSE (cm)")
    ax.set_title("Africa L3 test RMSE by model")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    fig.tight_layout()
    fig.savefig(REGION_FIGURES / "rmse_by_model.png", dpi=180)
    plt.close(fig)


def plot_observed_vs_predicted(predictions: pd.DataFrame) -> None:
    data = model_key_frame(predictions[predictions["split"].eq("test")], TOP_MODEL_KEYS)
    fig, ax = plt.subplots(figsize=(7, 7))
    for (model_name, graph_type), group in data.groupby(["model_name", "graph_type"], sort=False):
        ax.scatter(
            group["observed_twsa_cm"],
            group["predicted_twsa_cm"],
            s=10,
            alpha=0.35,
            label=model_label(model_name, graph_type),
        )
    mn = min(data["observed_twsa_cm"].min(), data["predicted_twsa_cm"].min())
    mx = max(data["observed_twsa_cm"].max(), data["predicted_twsa_cm"].max())
    ax.plot([mn, mx], [mn, mx], color="black", linewidth=1)
    ax.set_xlabel("Observed TWSA (cm)")
    ax.set_ylabel("Predicted TWSA (cm)")
    ax.set_title("Africa L3 observed vs predicted, test split")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(REGION_FIGURES / "observed_vs_predicted.png", dpi=180)
    plt.close(fig)


def plot_time_series(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    test = predictions[predictions["split"].eq("test")].copy()
    ridge = test[(test["model_name"].eq("ridge_ar")) & (test["graph_type"].eq("none"))]
    best = test[
        (test["model_name"].eq("ridge_neighbor_residual_mlp"))
        & (test["graph_type"].eq("corr_top3_directed"))
    ]
    comp = ridge[["date", "basin_id", "observed_twsa_cm", "predicted_twsa_cm"]].merge(
        best[["date", "basin_id", "predicted_twsa_cm"]],
        on=["date", "basin_id"],
        suffixes=("_ridge", "_best"),
    )
    comp["ridge_abs_error"] = (comp["observed_twsa_cm"] - comp["predicted_twsa_cm_ridge"]).abs()
    comp["best_abs_error"] = (comp["observed_twsa_cm"] - comp["predicted_twsa_cm_best"]).abs()
    basin_id = comp.groupby("basin_id").apply(
        lambda g: (g["ridge_abs_error"] - g["best_abs_error"]).mean(),
        include_groups=False,
    ).sort_values(ascending=False).index[0]

    model_keys = TOP_MODEL_KEYS[:5].copy()
    worst_test = metrics[metrics["split"].eq("test")].sort_values("rmse_cm", ascending=False).head(1)
    if not worst_test.empty:
        worst_key = (worst_test["model_name"].iloc[0], worst_test["graph_type"].iloc[0])
        if worst_key not in model_keys:
            model_keys.append(worst_key)

    data = model_key_frame(test[test["basin_id"].astype(str).eq(str(basin_id))], model_keys)
    obs = data.drop_duplicates("date").sort_values("date")
    basin_name = obs["basin_name"].iloc[0]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(obs["date"], obs["observed_twsa_cm"], color="black", linewidth=2, label="Observed")
    for (model_name, graph_type), group in data.groupby(["model_name", "graph_type"], sort=False):
        group = group.sort_values("date")
        ax.plot(group["date"], group["predicted_twsa_cm"], alpha=0.85, label=model_label(model_name, graph_type))
    ax.set_ylabel("TWSA (cm)")
    ax.set_title(f"Africa L3 test time series, {basin_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REGION_FIGURES / "timeseries_observed_vs_predicted.png", dpi=180)
    plt.close(fig)


def plot_worst_basin_time_series(predictions: pd.DataFrame, metrics: pd.DataFrame, metrics_by_region: pd.DataFrame) -> None:
    best_key = ("ridge_neighbor_residual_mlp", "corr_top3_directed")
    best_region_metrics = metrics_by_region[
        metrics_by_region["model_name"].eq(best_key[0])
        & metrics_by_region["graph_type"].eq(best_key[1])
    ].sort_values("rmse_cm", ascending=False)
    if best_region_metrics.empty:
        return
    basin_id = str(best_region_metrics.iloc[0]["basin_id"])

    model_keys = [
        best_key,
        ("ridge_ar", "none"),
        ("ridge_residual_mlp", "own_lags"),
        ("ridge_neighbor_ar", "corr_top3_directed"),
    ]
    worst_test = metrics[metrics["split"].eq("test")].sort_values("rmse_cm", ascending=False).head(1)
    if not worst_test.empty:
        worst_key = (worst_test["model_name"].iloc[0], worst_test["graph_type"].iloc[0])
        if worst_key not in model_keys:
            model_keys.append(worst_key)

    test = predictions[predictions["split"].eq("test")].copy()
    data = model_key_frame(test[test["basin_id"].astype(str).eq(basin_id)], model_keys)
    if data.empty:
        return
    obs = data.drop_duplicates("date").sort_values("date")
    basin_name = obs["basin_name"].iloc[0]
    worst_rmse = float(best_region_metrics.iloc[0]["rmse_cm"])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(obs["date"], obs["observed_twsa_cm"], color="black", linewidth=2, label="Observed")
    for (model_name, graph_type), group in data.groupby(["model_name", "graph_type"], sort=False):
        group = group.sort_values("date")
        ax.plot(group["date"], group["predicted_twsa_cm"], alpha=0.85, label=model_label(model_name, graph_type))
    ax.set_ylabel("TWSA (cm)")
    ax.set_title(f"Worst best-model basin time series: {basin_name} (RMSE {worst_rmse:.2f} cm)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REGION_FIGURES / "worst_basin_timeseries_observed_vs_predicted.png", dpi=180)
    plt.close(fig)


def plot_best_improvement_by_region(metrics_by_region: pd.DataFrame) -> None:
    ridge = metrics_by_region[
        metrics_by_region["model_name"].eq("ridge_ar") & metrics_by_region["graph_type"].eq("none")
    ][["basin_id", "basin_name", "rmse_cm"]].rename(columns={"rmse_cm": "ridge_rmse"})
    best = metrics_by_region[
        metrics_by_region["model_name"].eq("ridge_neighbor_residual_mlp")
        & metrics_by_region["graph_type"].eq("corr_top3_directed")
    ][["basin_id", "rmse_cm"]].rename(columns={"rmse_cm": "best_rmse"})
    data = ridge.merge(best, on="basin_id")
    data["rmse_improvement_cm"] = data["ridge_rmse"] - data["best_rmse"]
    data = data.sort_values("rmse_improvement_cm", ascending=False)
    colors = np.where(data["rmse_improvement_cm"] >= 0, "#228833", "#CC6677")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(data["basin_name"], data["rmse_improvement_cm"], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("RMSE improvement vs ridge (cm)")
    ax.set_title("Africa L3 best model improvement by region")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(REGION_FIGURES / "improvement_by_region.png", dpi=180)
    plt.close(fig)


def circular_graph_plot(edges: pd.DataFrame, basin_names: pd.DataFrame, title: str, output_name: str) -> None:
    nodes = sorted(set(edges["src_basin_id"].astype(str)) | set(edges["dst_basin_id"].astype(str)))
    names = basin_names.copy()
    names["basin_id"] = names["basin_id"].astype(str)
    name_map = dict(zip(names["basin_id"], names["basin_name"]))
    theta = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    pos = {node: (np.cos(t), np.sin(t)) for node, t in zip(nodes, theta)}
    fig, ax = plt.subplots(figsize=(8, 8))
    for row in edges.itertuples(index=False):
        src = str(row.src_basin_id)
        dst = str(row.dst_basin_id)
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        alpha = min(0.85, max(0.25, float(getattr(row, "weight", 1.0))))
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "->", "color": "#777777", "lw": 0.8, "alpha": alpha},
        )
    for node, (x, y) in pos.items():
        ax.scatter([x], [y], s=110, color="#4477AA", zorder=3)
        ax.text(x * 1.15, y * 1.15, name_map.get(node, node), ha="center", va="center", fontsize=6)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(REGION_FIGURES / output_name, dpi=180)
    plt.close(fig)


def correlation_topk_edges(train_df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    data = train_df.copy()
    data["basin_id"] = data["basin_id"].astype(str)
    pivot = data.pivot_table(index="date", columns="basin_id", values="target_twsa_cm", aggfunc="first")
    corr = pivot.corr()
    rows = []
    for dst in corr.columns:
        neighbors = corr[dst].drop(labels=[dst], errors="ignore").dropna().sort_values(ascending=False)
        for src, weight in neighbors[neighbors > 0].head(top_k).items():
            rows.append({"src_basin_id": str(src), "dst_basin_id": str(dst), "weight": float(weight)})
    edges = pd.DataFrame(rows)
    edges["graph_type"] = f"corr_top{top_k}_directed"
    return edges


def main() -> None:
    REGION_FIGURES.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(REGION_OUTPUTS / "predictions.csv", parse_dates=["date"])
    metrics = pd.read_csv(REGION_OUTPUTS / "metrics_overall.csv")
    metrics_by_region = pd.read_csv(REGION_OUTPUTS / "metrics_by_region.csv")
    lagged = pd.read_csv(LAGGED_DATASET_CSV, parse_dates=["date"])
    splits = chronological_fraction_split(lagged)
    basin_names = lagged[["basin_id", "basin_name"]].drop_duplicates()

    plot_rmse_bar(metrics)
    plot_observed_vs_predicted(predictions)
    plot_time_series(predictions, metrics)
    plot_worst_basin_time_series(predictions, metrics, metrics_by_region)
    plot_best_improvement_by_region(metrics_by_region)

    real_edges = pd.read_csv(REGION_OUTPUTS / "edges_real_knn_undirected.csv")
    circular_graph_plot(real_edges, basin_names, "Africa L3 centroid-kNN graph", "africa_graph.png")
    corr_edges = correlation_topk_edges(splits["train"], top_k=3)
    corr_edges.to_csv(REGION_OUTPUTS / "edges_corr_top3_directed.csv", index=False)
    circular_graph_plot(corr_edges, basin_names, "Africa L3 train-correlation top-3 graph", "africa_corr_graph.png")
    print(f"Saved Africa L3 figures to {REGION_FIGURES}")


if __name__ == "__main__":
    main()
