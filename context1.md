# CSR Africa L3 Forecasting Context

Last updated: 2026-07-14

## Short Answer

For CSR Africa L3, the clearest result is not the original `corr_top3_directed` run. The clearest result is the tuned top-k test:

| Dataset | Best neighbor setup | Model | RMSE cm | Matched random result |
|---|---|---|---:|---|
| CSR GRACE-only | `corr_top2_directed` | `ridge_neighbor_residual_mlp` | 3.2480 | Beats all matched random top-2 controls |
| CSR GRACE-only | `pred_lag1_top2_directed` | `ridge_neighbor_residual_mlp` | 3.2535 | Beats all matched random top-2 controls |
| CSR GRACE-only | `geo_incoming_top2` | `ridge_neighbor_residual_mlp` | 3.2925 | Beats all matched random top-2 controls |
| CSR+ERA5 | `corr_top2_directed` | `ridge_neighbor_residual_mlp_era5` | 2.7469 | Beats all matched random top-2 controls |
| CSR+ERA5 | `pred_lag1_top2_directed` | `ridge_neighbor_residual_mlp_era5` | 2.7499 | Beats all matched random top-2 controls |

Main interpretation:

- Top-2 basin context is the strongest and cleanest setting for CSR.
- Correlation top-2 and predictive lag-correlation top-2 are better evidence than the older top-3 result.
- Geographic top-2 helps for CSR GRACE-only, but it is weaker than correlation-selected neighbors.
- ERA5 improves the absolute RMSE a lot, but most of that gain appears to come from local lagged climate predictors. The correlation top-2 neighbor effect is still present, but smaller.
- These results show predictive value from selected basin context. They do not prove hydrologic flow or causality.

## Core Question

Do neighboring-region GRACE/GRACE-FO terrestrial water storage anomaly (TWSA) histories improve regional TWSA forecasting beyond strong own-region lag baselines when the raw GRACE source is CSR instead of JPL?

This file is the CSR companion to `context.md`. It uses the same mainland Africa Level 3 no-Madagascar mask setup, but uses `data/raw/grace_data.nc` as the GRACE source.

## Data Setup

- Region set: mainland Africa Level 3, excluding Madagascar.
- Basin count: 37 L3 basins.
- Output folder: `outputs/africa_l3_no_madagascar_csr/`.
- Raw GRACE/GRACE-FO source: `data/raw/grace_data.nc`.
- L3 mask zip: `masks/L3-20260709T200427Z-2-001.zip`.
- Processed basin-month source: `data/processed/basin_month_grace_africa_l3_no_madagascar_csr.csv`.
- One-month lag dataset: `data/processed/lagged_grace_dataset_africa_l3_no_madagascar_csr.csv`.

CSR NetCDF metadata from provenance:

- Source label: `csr`.
- Selected variable: `lwe_thickness`.
- Units: `cm`.
- Grid: 720 lat x 1440 lon, 0.25 degree cells.
- Time count: 257.
- Time handling: numeric days since `2002-01-01T00:00:00Z` decoded manually; lag builder normalizes to month starts.

The mask aggregation uses positive HydroBASINS mask cells with:

```text
mask weight * cos(latitude)
```

The CSR grid and L3 masks are both quarter-degree cell-center products, so CSR aggregation does not need the 0.25-to-0.5 nearest-cell downmapping used by the JPL grid.

The CSR lag builder averaged 74 duplicate basin-month entries after month normalization. These duplicates came from repeated CSR monthly solutions, not duplicate mask files.

CSR one-month split:

| Split | Rows | Date range |
|---|---:|---|
| Train | 4,736 | 2003-04 to 2021-10 |
| Validation | 666 | 2021-11 to 2023-04 |
| Test | 1,369 | 2023-05 to 2026-05 |

## Clean Neighbor Evidence

This is the main evidence section. These runs compare selected basin context against matched random controls with the same number of incoming basins per target.

Source folders:

- `outputs/africa_l3_no_madagascar_csr/corr_topk_sweep_grace_only/`
- `outputs/africa_l3_no_madagascar_csr/predictive_lag_topk_sweep_grace_only/`
- `outputs/africa_l3_no_madagascar_csr/geographic_topk_sweep_grace_only/`
- `outputs/africa_l3_no_madagascar_csr/corr_topk_sweep_era5/`
- `outputs/africa_l3_no_madagascar_csr/predictive_lag_topk_sweep_era5/`
- `outputs/africa_l3_no_madagascar_csr/geographic_topk_sweep_era5/`

Runners:

- `scripts/run_correlation_topk_sweep.py`
- `scripts/run_predictive_lag_topk_sweep.py`
- `scripts/run_geographic_topk_sweep.py`

### Correlation Top-k

`corr_top{k}_directed` means each target basin receives the top-k other basins whose training-period TWSA time series have the highest positive Pearson correlation with the target basin.

CSR GRACE-only:

| Model | Best k | Corr RMSE cm | Random mean RMSE cm | Random min RMSE cm | Beats all random seeds |
|---|---:|---:|---:|---:|---|
| `ridge_neighbor_residual_mlp` | 2 | 3.2480 | 3.5620 | 3.3514 | yes |
| `ridge_neighbor_ar` | 5 | 3.4482 | 3.6147 | 3.5968 | yes |

CSR+ERA5:

| Model | Best k | Corr RMSE cm | Random mean RMSE cm | Random min RMSE cm | Beats all random seeds |
|---|---:|---:|---:|---:|---|
| `ridge_neighbor_residual_mlp_era5` | 2 | 2.7469 | 2.7896 | 2.7588 | yes |
| `ridge_neighbor_ar_era5` | 1 | 2.9486 | 3.0009 | 2.9890 | yes |

Read this as the strongest evidence that selected basin context helps. The important change from the original run is `k=2`, not `k=3`.

### Predictive Lag-Correlation Top-k

`pred_lag1_top{k}_directed` chooses source basins by correlating each source basin's lag-1 TWSA with the target basin's current TWSA during training.

This asks a more forecast-oriented question:

```text
Which basins' past TWSA best predicts this basin's current TWSA?
```

| Dataset | Model | Best k | Predictive-corr RMSE cm | Random mean RMSE cm | Random min RMSE cm | Beats all random seeds |
|---|---|---:|---:|---:|---:|---|
| CSR GRACE-only | `ridge_neighbor_residual_mlp` | 2 | 3.2535 | 3.5620 | 3.3514 | yes |
| CSR+ERA5 | `ridge_neighbor_residual_mlp_era5` | 2 | 2.7499 | 2.7896 | 2.7588 | yes |

This is nearly tied with same-time correlation top-2 and is easier to explain as a forecasting feature.

### Geographic Top-k

`geo_incoming_top{k}` chooses source basins by nearest mask centroid distance. Each target basin receives its nearest k source basins.

| Dataset | Model | Best k | Geographic RMSE cm | Random mean RMSE cm | Random min RMSE cm | Beats all random seeds |
|---|---|---:|---:|---:|---:|---|
| CSR GRACE-only | `ridge_neighbor_residual_mlp` | 2 | 3.2925 | 3.5620 | 3.3514 | yes |
| CSR GRACE-only | `ridge_neighbor_ar` | 5 | 3.4937 | 3.6147 | 3.5968 | yes |
| CSR+ERA5 | `ridge_neighbor_residual_mlp_era5` | 2 | 2.7971 | 2.7896 | 2.7588 | no |
| CSR+ERA5 | `ridge_neighbor_ar_era5` | 5 | 2.9912 | 3.0050 | 2.9967 | yes |

Geographic neighbors support the idea that nearby basins can help in GRACE-only mode. They are not the best ERA5 evidence because the residual MLP geographic result does not beat matched random controls.

## Best Current Framing

Use this framing when explaining the CSR results:

> A small number of basins selected from training data improves CSR Africa L3 forecasting compared with matched random basin controls. The strongest setting is two selected basins per target. Same-time TWSA correlation top-2 gives the best RMSE, and predictive lag-correlation top-2 gives nearly the same result with a more forecast-oriented interpretation.

Avoid saying:

> The original top-3 correlation graph proves neighboring basins help.

That is not the cleanest result because the original `corr_top3_directed` run was beaten by random-degree controls on the final split.

## ERA5 Add-On

Source folders:

- `outputs/africa_l3_no_madagascar_csr/era5_one_month/`
- `outputs/africa_l3_no_madagascar_csr/era5_heavy_architectures/`
- `outputs/africa_l3_no_madagascar_csr/era5_svr/`

Runner: `scripts/run_africa_l3_csr_era5.py`.

Processed inputs:

- ERA5 basin-month file: `data/processed/basin_month_era5_africa_l3_no_madagascar_csr.csv`.
- Joined CSR GRACE+ERA5 lagged file: `data/processed/lagged_grace_era5_dataset_africa_l3_no_madagascar_csr.csv`.
- Joined rows: 6,771.
- Basin count: 37.
- Date range: 2003-04 to 2026-05.
- ERA5 predictors: lagged precipitation, runoff, and evaporation only; no target-month ERA5 columns are used.

Top CSR+ERA5 test RMSE results from the heavy architecture table:

| Rank | Model | Graph type | RMSE cm | MAE cm | Pearson r |
|---:|---|---|---:|---:|---:|
| 1 | `ridge_neighbor_residual_mlp_era5` | `random_degree_matched` | 2.7491 | 1.9034 | 0.9755 |
| 2 | `ridge_residual_mlp_era5` | `own_lags` | 2.7523 | 1.8959 | 0.9754 |
| 3 | `ridge_neighbor_residual_mlp_era5` | `corr_top3_directed` | 2.7708 | 1.9160 | 0.9751 |
| 4 | `ridge_neighbor_residual_mlp_era5` | `random_incoming_top3` | 2.8142 | 1.9346 | 0.9745 |
| 5 | `ridge_gnn_embedding_residual_era5` | `corr_top3_directed` | 2.8840 | 2.0251 | 0.9730 |
| 6 | `random_forest_gnn_embedding_residual_era5` | `corr_top3_directed` | 2.8842 | 2.0161 | 0.9731 |
| 7 | `xgboost_gnn_embedding_residual_era5` | `corr_top3_directed` | 2.9122 | 2.0326 | 0.9725 |
| 8 | `residual_neighbor_gnn_era5` | `random_degree_matched` | 2.9191 | 2.0107 | 0.9728 |

Compact read:

- ERA5 improves the CSR benchmark substantially.
- Best CSR GRACE-only final split in the original one-month table was 3.3533 cm RMSE.
- Best CSR+ERA5 original one-month result was 2.7491 cm RMSE.
- The tuned correlation top-2 ERA5 result is 2.7469 cm RMSE, which is slightly better than the original heavy architecture table.
- Own-lag ERA5 residual modeling is almost tied at 2.7523 cm, so much of the ERA5 gain likely comes from local lagged climate predictors rather than graph structure alone.

SVR follow-up results:

| Rank | Model | Graph type | RMSE cm | MAE cm | Pearson r |
|---:|---|---|---:|---:|---:|
| 1 | `rbf_svr_neighbor_residual_era5` | `corr_top3_directed` | 2.9298 | 2.0654 | 0.9731 |
| 2 | `linear_svr_era5` | `own_lags` | 3.0164 | 2.0830 | 0.9705 |
| 3 | `rbf_svr_era5` | `own_lags` | 5.5083 | 2.7946 | 0.9129 |

SVR did not beat the ridge residual MLP ERA5 results.

## Original One-Month CSR Run

Source: `outputs/africa_l3_no_madagascar_csr/metrics_overall.csv`.

This was the first full CSR one-month result. It used the original `corr_top3_directed` graph and older random controls.

Top test RMSE results:

| Rank | Model | Graph type | RMSE cm | MAE cm | Pearson r |
|---:|---|---|---:|---:|---:|
| 1 | `ridge_neighbor_residual_mlp` | `random_degree_matched` | 3.3533 | 2.3451 | 0.9631 |
| 2 | `ridge_neighbor_residual_mlp` | `random_incoming_top3` | 3.3629 | 2.3118 | 0.9629 |
| 3 | `ridge_neighbor_residual_mlp` | `corr_top3_directed` | 3.4366 | 2.3988 | 0.9613 |
| 4 | `ridge_neighbor_residual_lstm` | `corr_top3_directed` | 3.4496 | 2.4097 | 0.9611 |
| 5 | `ridge_neighbor_residual_tcn` | `corr_top3_directed` | 3.4535 | 2.4134 | 0.9609 |
| 6 | `ridge_neighbor_ar` | `corr_top3_directed` | 3.4645 | 2.4142 | 0.9609 |
| 7 | `ridge_neighbor_ar` | `real_knn_undirected` | 3.5282 | 2.4733 | 0.9594 |
| 8 | `ridge_neighbor_ar` | `real_knn_reversed` | 3.5286 | 2.4736 | 0.9594 |
| 9 | `residual_neighbor_gnn` | `real_knn_undirected` | 3.5638 | 2.4981 | 0.9596 |
| 10 | `ridge_neighbor_ar` | `real_knn_directed` | 3.5645 | 2.4979 | 0.9585 |
| 11 | `residual_neighbor_gnn` | `real_knn_reversed` | 3.5662 | 2.4950 | 0.9598 |
| 12 | `ridge_residual_mlp` | `own_lags` | 3.5920 | 2.5148 | 0.9576 |

Compact read:

- The original final-split CSR result was mixed.
- `random_degree_matched` beat `corr_top3_directed`, so this table should not be used as the main proof that correlation neighbors help.
- The result was still useful because it showed CSR is strongly predictable overall: best Pearson r was 0.9631 and NSE was 0.9275.
- CSR one-month RMSE is worse than the JPL benchmark in `context.md`; the degradation starts even at persistence/ridge baselines, so it is likely a product/source difference rather than only a graph-model issue.

## Walk-Forward Robustness

Source folder: `outputs/africa_l3_no_madagascar_csr/walk_forward_top5/`.

Main files:

- `metrics_summary.csv`
- `metrics_by_fold.csv`
- `rankings_by_fold.csv`
- `graph_audit.csv`

Summary:

| Model | Graph type | Mean RMSE cm | Median RMSE cm | Worst-fold RMSE cm | Mean rank | Rank-1 folds |
|---|---|---:|---:|---:|---:|---:|
| `ridge_neighbor_residual_mlp` | `corr_top3_directed` | 3.0333 | 3.1342 | 3.7452 | 1.2 | 4/5 |
| `ridge_residual_mlp` | `own_lags` | 3.2094 | 3.3904 | 3.8402 | 2.6 | 0/5 |
| `xgboost_gnn_embedding_residual` | `corr_top3_directed` | 3.2290 | 3.4995 | 3.7668 | 3.4 | 0/5 |
| `ridge_neighbor_residual_mlp` | `random_degree_matched` | 3.2363 | 3.5451 | 3.8809 | 3.6 | 1/5 |
| `random_forest_gnn_embedding_residual` | `corr_top3_directed` | 3.2476 | 3.5000 | 3.8191 | 4.2 | 0/5 |

The walk-forward result is more favorable to the train-correlation graph than the original final split: `ridge_neighbor_residual_mlp | corr_top3_directed` has the best mean RMSE and ranks first in 4 of 5 folds.

Report this as supporting evidence, not the main result, because the tuned top-2 sweeps are cleaner.

## Multi-Horizon CSR GRACE-Only

Source folder: `outputs/africa_l3_no_madagascar_csr/grace_only_horizons/`.

Runner: `scripts/run_africa_l3_csr.py`, which patches `scripts/run_africa_l3_grace_only_horizons.py` to CSR paths.

The horizon run uses issue-date features only. Horizon 1 predicts 1 month after the issue month; horizon 6 predicts 6 months after the issue month.

Best test result by horizon:

| Horizon months | Best model | Graph type | Test rows | RMSE cm | MAE cm | Pearson r |
|---:|---|---|---:|---:|---:|---:|
| 1 | `ridge_neighbor_residual_mlp` | `corr_top3_directed` | 1,369 | 3.2203 | 2.2268 | 0.9664 |
| 2 | `ridge_neighbor_residual_mlp` | `corr_top3_directed` | 1,332 | 4.7232 | 3.2235 | 0.9300 |
| 3 | `ridge_neighbor_residual_mlp` | `corr_top3_directed` | 1,406 | 5.5466 | 3.8723 | 0.9002 |
| 4 | `ridge_neighbor_residual_mlp` | `corr_top3_directed` | 1,406 | 5.9365 | 4.1672 | 0.8836 |
| 5 | `random_forest_gnn_embedding_residual` | `corr_top3_directed` | 1,443 | 7.1794 | 5.0401 | 0.8200 |
| 6 | `ridge_neighbor_residual_mlp` | `random_degree_matched` | 1,369 | 5.6842 | 3.8484 | 0.8948 |

Compact read:

- Error generally increases from horizons 1 to 5.
- Horizon 6 is lower than horizon 5 in this run, so do not interpret the horizon curve as perfectly monotonic.
- The correlation-neighbor residual MLP wins horizons 1-4.
- Horizon 6 is won by the random-degree control, so keep graph interpretation cautious.

## Files To Share

Small CSR CSVs:

| Purpose | File |
|---|---|
| One-month original model ranking | `outputs/africa_l3_no_madagascar_csr/metrics_overall.csv` |
| Basin-level one-month original results | `outputs/africa_l3_no_madagascar_csr/metrics_by_region.csv` |
| CSR correlation top-k GRACE-only sweep | `outputs/africa_l3_no_madagascar_csr/corr_topk_sweep_grace_only/topk_summary.csv` |
| CSR predictive lag top-k GRACE-only sweep | `outputs/africa_l3_no_madagascar_csr/predictive_lag_topk_sweep_grace_only/topk_summary.csv` |
| CSR geographic top-k GRACE-only sweep | `outputs/africa_l3_no_madagascar_csr/geographic_topk_sweep_grace_only/topk_summary.csv` |
| CSR correlation top-k ERA5 sweep | `outputs/africa_l3_no_madagascar_csr/corr_topk_sweep_era5/topk_summary.csv` |
| CSR predictive lag top-k ERA5 sweep | `outputs/africa_l3_no_madagascar_csr/predictive_lag_topk_sweep_era5/topk_summary.csv` |
| CSR geographic top-k ERA5 sweep | `outputs/africa_l3_no_madagascar_csr/geographic_topk_sweep_era5/topk_summary.csv` |
| CSR ERA5 one-month summary | `outputs/africa_l3_no_madagascar_csr/era5_one_month/era5_vs_grace_only_summary.csv` |
| CSR ERA5 heavy architecture ranking | `outputs/africa_l3_no_madagascar_csr/era5_heavy_architectures/heavy_architecture_summary.csv` |
| CSR ERA5 SVR ranking | `outputs/africa_l3_no_madagascar_csr/era5_svr/svr_summary.csv` |
| 1-6 month best model per horizon | `outputs/africa_l3_no_madagascar_csr/grace_only_horizons/metrics_summary_by_horizon.csv` |
| 1-6 month full model ranking by horizon | `outputs/africa_l3_no_madagascar_csr/grace_only_horizons/rankings_by_horizon.csv` |
| Walk-forward robustness summary | `outputs/africa_l3_no_madagascar_csr/walk_forward_top5/metrics_summary.csv` |
| Walk-forward fold rankings | `outputs/africa_l3_no_madagascar_csr/walk_forward_top5/rankings_by_fold.csv` |
| Shareable spreadsheet folder | `outputs/share_csr_spreadsheets/` |

CSR figures:

- `outputs/africa_l3_no_madagascar_csr/figures/rmse_by_model.png`
- `outputs/africa_l3_no_madagascar_csr/figures/observed_vs_predicted.png`
- `outputs/africa_l3_no_madagascar_csr/figures/timeseries_observed_vs_predicted.png`
- `outputs/africa_l3_no_madagascar_csr/figures/worst_basin_timeseries_observed_vs_predicted.png`
- `outputs/africa_l3_no_madagascar_csr/figures/basin_timeseries_selected_models/`

Avoid sending by default because they are large:

- `outputs/africa_l3_no_madagascar_csr/predictions.csv`
- `outputs/africa_l3_no_madagascar_csr/grace_only_horizons/predictions_by_horizon.csv`
- `outputs/africa_l3_no_madagascar_csr/walk_forward_top5/predictions_walk_forward.csv`
- `outputs/africa_l3_no_madagascar_csr/era5_one_month/predictions.csv`
- `outputs/africa_l3_no_madagascar_csr/era5_heavy_architectures/predictions.csv`
- `outputs/africa_l3_no_madagascar_csr/era5_svr/predictions.csv`
- `outputs/africa_l3_no_madagascar_csr/corr_topk_sweep_grace_only/predictions.csv`
- `outputs/africa_l3_no_madagascar_csr/corr_topk_sweep_era5/predictions.csv`
- `outputs/africa_l3_no_madagascar_csr/predictive_lag_topk_sweep_grace_only/predictions.csv`
- `outputs/africa_l3_no_madagascar_csr/predictive_lag_topk_sweep_era5/predictions.csv`
- `outputs/africa_l3_no_madagascar_csr/geographic_topk_sweep_grace_only/predictions.csv`
- `outputs/africa_l3_no_madagascar_csr/geographic_topk_sweep_era5/predictions.csv`
