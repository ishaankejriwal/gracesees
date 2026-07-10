from __future__ import annotations

import pandas as pd


def prediction_frame(df: pd.DataFrame, preds, model_name: str, graph_type: str, split: str) -> pd.DataFrame:
    out = df[["date", "basin_id", "basin_name", "target_twsa_cm"]].copy()
    out = out.rename(columns={"target_twsa_cm": "observed_twsa_cm"})
    out["model_name"] = model_name
    out["graph_type"] = graph_type
    out["split"] = split
    out["predicted_twsa_cm"] = preds
    out["residual_cm"] = out["observed_twsa_cm"] - out["predicted_twsa_cm"]
    return out[
        [
            "date",
            "basin_id",
            "basin_name",
            "model_name",
            "graph_type",
            "split",
            "observed_twsa_cm",
            "predicted_twsa_cm",
            "residual_cm",
        ]
    ]


def graph_prediction_frame(graph_df: pd.DataFrame, model_name: str, graph_type: str, split: str) -> pd.DataFrame:
    if graph_df.empty:
        return graph_df
    out = graph_df.rename(columns={"target_twsa_cm": "observed_twsa_cm"}).copy()
    out["model_name"] = model_name
    out["graph_type"] = graph_type
    out["split"] = split
    out["residual_cm"] = out["observed_twsa_cm"] - out["predicted_twsa_cm"]
    return out[
        [
            "date",
            "basin_id",
            "basin_name",
            "model_name",
            "graph_type",
            "split",
            "observed_twsa_cm",
            "predicted_twsa_cm",
            "residual_cm",
        ]
    ]

