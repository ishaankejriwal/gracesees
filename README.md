# GRACE-Only Africa L3 Forecasting Experiment

This project tests whether neighboring GRACE/GRACE-FO terrestrial water
storage anomaly history improves next-month regional TWSA prediction over
strong own-region autoregressive baselines.

The current validated experiment is:

> Mainland Africa Level 3 GRACE mask-region next-month TWSA forecasting,
> excluding Madagascar.

The current canonical workflow is script-based. The notebooks are useful for
interactive exploration, but they are not the cleanest way to reproduce the
latest Africa L3 results.

## What To Send Someone

Send these files/folders:

- `README.md`
- `context.md`
- `requirements.txt`
- `src/`
- `scripts/`
- `notebooks/` if they want to inspect the earlier notebook workflow
- `masks/L3-20260709T200427Z-2-001.zip`
- `data/raw/GRCTellus.JPL.200204_202604.GLO.RL06.3M.MSCNv04.nc`

Do not send these unless you specifically want to share precomputed results:

- `.venv/`
- `outputs/`
- `data/processed/`
- `masks/L2-*.zip`
- `Level 2 Hydrobasins Layout.gif`

The two Level 2 mask zips are very large and are not needed for the current
Africa L3 rerun.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS/Linux, use:

```bash
python -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

## Required Data Layout

Place the GRACE NetCDF here:

```text
data/raw/GRCTellus.JPL.200204_202604.GLO.RL06.3M.MSCNv04.nc
```

Place the Africa L3 mask zip here:

```text
masks/L3-20260709T200427Z-2-001.zip
```

The L3 zip contains `.mask.csv` files. The pipeline parses those directly.

## Reproduce The Current Results

Run these commands from the project root:

```bash
.\.venv\Scripts\python.exe scripts\run_africa_l3.py
.\.venv\Scripts\python.exe scripts\run_africa_l3_extra_architectures.py
.\.venv\Scripts\python.exe scripts\make_africa_l3_figures.py
```

Outputs are written to:

```text
outputs/africa_l3_no_madagascar/
```

The main summary table is:

```text
outputs/africa_l3_no_madagascar/metrics_overall.csv
```

## Current Best Result

The current validated best model is:

```text
ridge_neighbor_residual_mlp | corr_top3_directed | test RMSE 2.3769 cm
```

Plain ridge autoregression remains strong:

```text
ridge_ar | none | test RMSE 2.7003 cm
```

The safest interpretation is that own-region lag history is the backbone, and
train-period correlation-neighbor information helps as an auxiliary residual
signal. Do not describe the correlation or centroid-kNN graphs as proven
hydrologic flow connectivity.

## Notebooks Versus Scripts

The notebooks were the original exploratory workflow. They import the same
package code from `src/grace_gnn`, so sending notebooks alone is not enough to
rerun the project.

For reproducibility, send the package code and scripts. If someone wants to
learn the workflow step by step, include the notebooks too, but treat the
scripts above as the current source of truth.
