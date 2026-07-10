from __future__ import annotations

import pandas as pd


def region_correlation_matrix(train_df: pd.DataFrame) -> pd.DataFrame:
    data = train_df.copy()
    data["date"] = pd.to_datetime(data["date"])
    return data.pivot_table(index="date", columns="basin_name", values="target_twsa_cm", aggfunc="first").corr()


def region_correlation_pairs(corr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    names = list(corr.columns)
    for i, source in enumerate(names):
        for target in names[i + 1 :]:
            rows.append({
                "region_a": source,
                "region_b": target,
                "pearson_r": float(corr.loc[source, target]),
                "abs_pearson_r": abs(float(corr.loc[source, target])),
            })
    return pd.DataFrame(rows).sort_values("abs_pearson_r", ascending=False).reset_index(drop=True)
