# Africa Forecasting Context

Last updated: 2026-07-14

## Core Question

Do neighboring-region GRACE/GRACE-FO terrestrial water storage anomaly (TWSA) histories improve regional TWSA forecasting beyond strong own-region lag baselines?

Default benchmark constraint: model inputs are GRACE/GRACE-FO TWSA time series only. Masks/geometries define regions and some graph edges. ERA5 is now tracked as a separate add-on experiment, not part of the GRACE-only benchmark.

## Current Active Experiment

Main benchmark:

- Region set: mainland Africa Level 3, excluding Madagascar.
- Basin count: 37 L3 basins.
- Output folder: `outputs/africa_l3_no_madagascar/`.
- Processed basin-month source: `data/processed/basin_month_grace_africa_l3_no_madagascar.csv`.
- Canonical one-month lag dataset: `data/processed/lagged_grace_dataset_africa_l3_no_madagascar.csv`.
- Raw GRACE/GRACE-FO source: `data/raw/GRCTellus.JPL.200204_202604.GLO.RL06.3M.MSCNv04.nc`.
- L3 mask zip: `masks/L3-20260709T200427Z-2-001.zip`.

The Level 3 lag builder averages duplicate `(basin_id, month)` GRACE entries before creating lags. The current rerun averaged 74 duplicate basin-month rows from `2012-01` and `2015-04`.

Canonical one-month split:

| Split | Rows | Date range |
|---|---:|---|
| Train | 4,699 | 2003-04 to 2021-09 |
| Validation | 666 | 2021-10 to 2023-03 |
| Test | 1,369 | 2023-04 to 2026-04 |

## Canonical One-Month Results

Source: `outputs/africa_l3_no_madagascar/metrics_overall.csv`.

Top test RMSE results:

| Rank | Model | Graph type | RMSE cm | MAE cm | Pearson r |
|---:|---|---|---:|---:|---:|
| 1 | ridge_neighbor_residual_mlp | corr_top3_directed | 2.3769 | 1.5615 | 0.9818 |
| 2 | ridge_neighbor_residual_mlp | random_degree_matched | 2.4707 | 1.6543 | 0.9802 |
| 3 | ridge_neighbor_residual_mlp | random_incoming_top3 | 2.4906 | 1.6434 | 0.9799 |
| 4 | ridge_residual_mlp | own_lags | 2.5178 | 1.6602 | 0.9797 |
| 5 | xgboost_gnn_embedding_residual | corr_top3_directed | 2.5415 | 1.7262 | 0.9791 |
| 6 | random_forest_gnn_embedding_residual | corr_top3_directed | 2.5434 | 1.7250 | 0.9790 |
| 7 | ridge_gnn_embedding_residual | corr_top3_directed | 2.5695 | 1.7191 | 0.9787 |
| 8 | ridge_neighbor_residual_lstm | corr_top3_directed | 2.5932 | 1.7370 | 0.9785 |
| 9 | ridge_neighbor_ar | corr_top3_directed | 2.6279 | 1.7543 | 0.9782 |
| 10 | ridge_neighbor_residual_tcn | corr_top3_directed | 2.6288 | 1.7615 | 0.9781 |
| 11 | ridge_neighbor_ar | real_knn_undirected | 2.6531 | 1.7872 | 0.9779 |
| 12 | ridge_neighbor_ar | real_knn_reversed | 2.6556 | 1.7849 | 0.9779 |

Compact read:

- The best current one-month L3 model is `ridge_neighbor_residual_mlp | corr_top3_directed`.
- Plain ridge AR is still strong at 2.7003 cm RMSE, but ridge plus residual correction is better.
- Both random controls also perform well, but the train-correlation graph still beats them on the JPL final split. Describe the neighbor gain cautiously: it is predictive structure plus residual regularization, not proof of physical hydrologic transfer.
- GNN embedding residual models are competitive but do not beat the simpler ridge-neighbor residual MLP.

## ERA5 Add-On One-Month Result

Source folder: `outputs/africa_l3_era5_one_month/`.

Runner: `scripts/run_africa_l3_era5_one_month.py`.

This keeps the monthly `target_twsa_cm` target and adds only lagged ERA5 precipitation, runoff, and evaporation features. No target-month ERA5 columns are used.

Best test result:

| Model | Graph type | RMSE cm | MAE cm | Pearson r | Delta vs GRACE-only best |
|---|---|---:|---:|---:|---:|
| ridge_neighbor_residual_mlp_era5 | corr_top3_directed | 1.8494 | 1.2324 | 0.9890 | -0.5276 |

Compact read:

- ERA5 is useful in this first one-month pass: the best ERA5 model beats the GRACE-only best, 1.8494 vs 2.3769 cm RMSE.
- It also beats the ERA5 random controls: 1.8494 vs 1.8786 cm for `random_degree_matched` and 1.9615 cm for `random_incoming_top3`.
- Treat this as an add-on result until walk-forward ERA5 validation is run.

## GRACE-Only 1-6 Month Horizon Results

Source folder: `outputs/africa_l3_no_madagascar/grace_only_horizons/`.

Runner: `scripts/run_africa_l3_grace_only_horizons.py`.

This is a separate benchmark that keeps horizon 1 aligned with the canonical one-month setup, then extends the same GRACE-only question to forecasts 2-6 months ahead.

The horizon run now includes persistence, ridge AR, random forest AR, XGBoost AR, own-lag ridge residual MLP, correlation/random neighbor ridge, correlation/random neighbor residual MLP, and ridge/random-forest/XGBoost residual models using train-correlation GNN embeddings.

Horizon definition:

- Horizon 1: predict 1 month after the issue month.
- Horizon 2: predict 2 months after the issue month.
- Horizon 6: predict 6 months after the issue month.

The horizon dataset uses issue-date features only. For horizon 1, the issue-date lags map back to the old target-date lags:

- New issue-date lags: `lag_0`, `lag_1`, `lag_2`, `lag_5`, `lag_11`.
- Equivalent old target-date lags: `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12`.

Best test result by horizon:

| Horizon months | Best model | Graph type | Test rows | RMSE cm | MAE cm | Pearson r |
|---:|---|---|---:|---:|---:|---:|
| 1 | ridge_neighbor_residual_mlp | corr_top3_directed | 1,369 | 2.3793 | 1.5617 | 0.9817 |
| 2 | ridge_neighbor_residual_mlp | corr_top3_directed | 1,332 | 3.8095 | 2.5174 | 0.9536 |
| 3 | ridge_neighbor_residual_mlp | corr_top3_directed | 1,406 | 4.5057 | 2.9967 | 0.9340 |
| 4 | ridge_neighbor_residual_mlp | corr_top3_directed | 1,443 | 5.1281 | 3.3791 | 0.9147 |
| 5 | ridge_neighbor_residual_mlp | corr_top3_directed | 1,443 | 5.2432 | 3.4474 | 0.9150 |
| 6 | ridge_neighbor_residual_mlp | random_degree_matched | 1,369 | 5.5261 | 3.8571 | 0.9058 |

Compact read:

- Horizon 1 now matches the canonical one-month result closely: 2.3793 cm vs 2.3769 cm.
- Forecast error increases as the target moves farther into the future.
- The correlation-neighbor residual model wins horizons 1-5.
- The random graph control wins horizon 6, so the 6-month neighbor result should not be overinterpreted as meaningful graph structure.

## Walk-Forward Robustness

Source folder: `outputs/africa_l3_no_madagascar/walk_forward_top5/`.

Main files:

- `metrics_summary.csv`
- `metrics_by_fold.csv`
- `rankings_by_fold.csv`
- `graph_audit.csv`

Summary:

| Model | Graph type | Mean RMSE cm | Median RMSE cm | Worst-fold RMSE cm | Mean rank | Rank-1 folds |
|---|---|---:|---:|---:|---:|---:|
| ridge_neighbor_residual_mlp | corr_top3_directed | 2.5207 | 2.4813 | 2.9323 | 1.0 | 5/5 |
| xgboost_gnn_embedding_residual | corr_top3_directed | 2.6456 | 2.5321 | 2.9700 | 2.8 | 0/5 |
| random_forest_gnn_embedding_residual | corr_top3_directed | 2.6681 | 2.5742 | 3.0581 | 3.4 | 0/5 |
| ridge_neighbor_residual_mlp | random_degree_matched | 2.7069 | 2.6950 | 3.0284 | 3.8 | 0/5 |
| ridge_residual_mlp | own_lags | 2.7131 | 2.6763 | 3.0383 | 4.0 | 0/5 |

The walk-forward audit supports the final-split ranking for the selected top-five one-month models. The train-correlation graph is rebuilt inside each fold from that fold's training period only.

## Historical Africa L2 Baseline

Africa Level 2, excluding Madagascar, is an older 7-region benchmark. It is useful for context but should not be compared directly against L3 as a pure model-quality score because L2 regions are broader and smoother.

Best L2 result:

| Run | Best model | Graph type | Test RMSE cm |
|---|---|---|---:|
| Africa L2 no Madagascar | ridge_ar | none | 1.9869 |

## Graph And Model Notes

Graph types:

- `corr_top3_directed`: train-period positive-correlation top-3 graph.
- `random_degree_matched`: placebo graph preserving source out-degree.
- `real_knn_directed`: 3-nearest mask-centroid graph.
- `real_knn_undirected`: symmetrized centroid-kNN graph.
- `real_knn_reversed`: reversed centroid-kNN graph.

Reporting caveat:

Do not describe these graphs as proven hydrologic flow. The correlation graph is statistical similarity. The centroid-kNN graphs are geographic proximity. Gains may reflect shared climate, GRACE smoothing/leakage, broad regional storage behavior, model regularization, or real hydrologic relatedness.

## Best Files To Share

Small, low-overwhelm CSVs:

| Purpose | File |
|---|---|
| Canonical one-month model ranking | `outputs/africa_l3_no_madagascar/metrics_overall.csv` |
| One-month best-by-model-family summary | `outputs/africa_l3_no_madagascar/metrics_overall_best_by_architecture.csv` |
| Basin-level one-month results | `outputs/africa_l3_no_madagascar/metrics_by_region_best_by_architecture.csv` |
| 1-6 month best model per horizon | `outputs/africa_l3_no_madagascar/grace_only_horizons/metrics_summary_by_horizon.csv` |
| 1-6 month full model ranking by horizon | `outputs/africa_l3_no_madagascar/grace_only_horizons/rankings_by_horizon.csv` |
| 1-6 month basin-level horizon metrics | `outputs/africa_l3_no_madagascar/grace_only_horizons/metrics_by_basin_horizon.csv` |
| Walk-forward robustness summary | `outputs/africa_l3_no_madagascar/walk_forward_top5/metrics_summary.csv` |
| Walk-forward fold rankings | `outputs/africa_l3_no_madagascar/walk_forward_top5/rankings_by_fold.csv` |
| ERA5 one-month summary | `outputs/africa_l3_era5_one_month/era5_vs_grace_only_summary.csv` |

Avoid sending by default because they are large and too detailed for a first look:

- `outputs/africa_l3_no_madagascar/predictions.csv`
- `outputs/africa_l3_no_madagascar/grace_only_horizons/predictions_by_horizon.csv`
- `outputs/africa_l3_no_madagascar/metrics_by_region.csv`
- Edge CSVs, unless the recipient specifically wants graph structure.

## Current Short Summary

For the canonical one-month Africa L3 GRACE-only task, the best model is `ridge_neighbor_residual_mlp | corr_top3_directed` at 2.3769 cm test RMSE. Adding lagged ERA5 improves the first one-month pass to 1.8494 cm, beating its random-degree control, but still needs walk-forward validation. In the GRACE-only horizon benchmark, horizon 1 aligns with the canonical result, and RMSE rises from about 2.38 cm at 1 month to about 5.53 cm at 6 months.
