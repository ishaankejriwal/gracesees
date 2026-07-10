# Sharing Checklist

Use this if you want someone else to rerun the current Africa L3 experiment.

## Send

- `README.md`
- `SHARE_WITH_FRIENDS.md`
- `context.md`
- `requirements.txt`
- `src/`
- `scripts/`
- `notebooks/` optional, for walkthrough/exploration
- `masks/L3-20260709T200427Z-2-001.zip`
- `data/raw/GRCTellus.JPL.200204_202604.GLO.RL06.3M.MSCNv04.nc`

## Do Not Send

- `.venv/`
- `outputs/`
- `data/processed/`
- `masks/L2-*.zip`
- `Level 2 Hydrobasins Layout.gif`
- `__pycache__/`

## Commands They Run

From the project root:

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_africa_l3.py
.\.venv\Scripts\python.exe scripts\run_africa_l3_extra_architectures.py
.\.venv\Scripts\python.exe scripts\make_africa_l3_figures.py
```

The key result file will be:

```text
outputs/africa_l3_no_madagascar/metrics_overall.csv
```

## Important

Do not send notebooks alone. They import helper code from `src/grace_gnn`, so
they need the package files to run correctly.
