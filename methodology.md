# Methodology

Last updated: 2026-07-14

## Goal

This project tests whether information from other African Level 3 basins can improve monthly terrestrial water storage anomaly forecasting.

The target variable is GRACE or GRACE-FO terrestrial water storage anomaly, reported as liquid water equivalent thickness in centimeters. The main forecast task is one-month-ahead prediction for mainland Africa Level 3 HydroBASINS regions, excluding Madagascar.

The main question is:

> Can basin context, especially correlation-selected basin context, improve forecasts beyond strong own-basin lag baselines?

## Study Region And Basin Masks

The experiment uses mainland Africa Level 3 HydroBASINS masks:

- Mask archive: `masks/L3-20260709T200427Z-2-001.zip`
- Region count: 37 basins after excluding Madagascar
- Mask format: `.mask.csv` files with longitude, latitude, and mask weight

Each basin mask is converted into a basin-average time series by reading all positive mask cells and aggregating gridded data over those cells.

The aggregation uses:

```text
mask weight * cos(latitude)
```

This makes the basin time series area-aware. It produces basin-average TWSA, not total water mass.

## GRACE Data Sources

Two GRACE/GRACE-FO mascon products are used:

### JPL

Canonical JPL source:

```text
data/raw/GRCTellus.JPL.200204_202604.GLO.RL06.3M.MSCNv04.nc
```

The JPL grid is 0.5 degree. Quarter-degree HydroBASINS mask cells are mapped to the nearest JPL grid cell before basin averaging.

### CSR

CSR source:

```text
data/raw/grace_data.nc
```

The CSR grid is 0.25 degree, matching the quarter-degree HydroBASINS mask cell centers. CSR time values are stored as numeric days since `2002-01-01T00:00:00Z`, so the pipeline decodes them manually and then normalizes to monthly timestamps during lag creation.

CSR is the main focus of the later experiments and shareable spreadsheets.

## Basin-Month Dataset Construction

For each basin and month, the pipeline computes an area-weighted average TWSA value.

The basin-month file for CSR is:

```text
data/processed/basin_month_grace_africa_l3_no_madagascar_csr.csv
```

The one-month lagged CSR dataset is:

```text
data/processed/lagged_grace_dataset_africa_l3_no_madagascar_csr.csv
```

Duplicate basin-month records are averaged before lag features are created. This matters because some GRACE products contain repeated monthly solutions after timestamps are normalized to month starts.

The lag features are:

```text
lag_1, lag_2, lag_3, lag_6, lag_12
```

The prediction target is:

```text
target_twsa_cm
```

No target-month TWSA is used as a predictor.

## Train, Validation, And Test Split

The split is chronological by month. The default split fractions are:

- 70 percent train
- 10 percent validation
- 20 percent test

For CSR one-month experiments:

| Split | Rows | Date range |
|---|---:|---|
| Train | 4,736 | 2003-04 to 2021-10 |
| Validation | 666 | 2021-11 to 2023-04 |
| Test | 1,369 | 2023-05 to 2026-05 |

All reported headline metrics are test-split metrics.

## Baseline Models

The baseline models use only the target basin's own lagged TWSA values.

Important baselines:

- `persistence`: uses the most recent available lag as the forecast
- `ridge_ar`: ridge regression on own-basin lag features
- `ridge_residual_mlp`: ridge baseline plus a small neural network trained on the residual
- `basin_only_nn`: neural network using own-basin information only

These baselines are important because neighbor models must beat them to show that basin context is useful.

## Neighbor And Basin-Context Graphs

The project tests several ways to choose which other basins provide context.

### Geographic Neighbors

Geographic graphs are based on mask centroids:

- `real_knn_directed`: three nearest basin centroids
- `real_knn_undirected`: symmetrized centroid nearest-neighbor graph
- `real_knn_reversed`: reversed centroid nearest-neighbor graph
- `geo_incoming_top{k}`: each target basin receives its k nearest source basins by centroid distance

These are physically intuitive, but they do not always give the best forecast signal.

### Correlation Neighbors

Correlation neighbors are selected using the training period only.

For each target basin:

1. Build a training-period time series for every basin.
2. Compute Pearson correlation between the target basin and every other basin.
3. Keep the highest positive correlations.
4. Use those selected basins as context features.

The original version used:

```text
corr_top3_directed
```

Later tuning showed that CSR works better with:

```text
corr_top2_directed
```

This means each target basin receives the two basins whose training-period TWSA time series move most similarly.

### Predictive Lag-Correlation Neighbors

The predictive lag version is more forecasting-oriented.

Instead of asking:

> Which basins move similarly at the same time?

it asks:

> Which basins' past TWSA best predicts this basin's current TWSA?

The strongest CSR setting was:

```text
pred_lag1_top2_directed
```

This selects two source basins for each target basin using the correlation between source-basin `lag_1` and target-basin current TWSA during training.

This is useful because it directly matches the forecasting task.

### Random Controls

Random controls are used to check whether selected basin context is actually meaningful.

The cleaner control is:

```text
random_incoming_top2
```

For each target basin, it chooses exactly two random source basins. This matches the structure of `corr_top2_directed` and `pred_lag1_top2_directed`.

The older control is:

```text
random_degree_matched
```

This preserves graph degree from another graph, but it does not always give each target basin the same number of incoming neighbors. It can act more like a random regional smoothing feature, so it is useful to report but less clean as the main scientific control.

## How Neighbor Features Are Used

After a graph is selected, the model builds neighbor-lag features.

For each target basin and month:

1. Find the source basins pointing into that target basin.
2. Pull the source basins' lag features for that same month.
3. Average those source lag features using graph weights.
4. Add the resulting neighbor-lag features to the model.

The model then trains on both:

- own-basin lag features
- neighbor-lag features

The strongest model family is:

```text
ridge_neighbor_residual_mlp
```

This first fits a ridge model, then trains a small residual MLP to correct the remaining error.

## ERA5 Add-On

ERA5 is added as a separate experiment, not part of the pure GRACE-only benchmark.

ERA5 variables:

- total precipitation
- runoff
- evaporation

The ERA5 basin-month file for CSR is:

```text
data/processed/basin_month_era5_africa_l3_no_madagascar_csr.csv
```

The joined CSR GRACE+ERA5 lagged dataset is:

```text
data/processed/lagged_grace_era5_dataset_africa_l3_no_madagascar_csr.csv
```

ERA5 predictors are lagged only. No target-month ERA5 values are used.

ERA5 lag features are created for the same lags:

```text
1, 2, 3, 6, 12 months
```

ERA5 improves the CSR final-split forecasts substantially, but graph interpretation still requires controls.

## Main CSR Results

### CSR GRACE-Only

The original top-3 correlation graph was not the best CSR setting. The tuned top-k sweep found that top 2 was stronger.

Best clean correlation-context result:

| Model | Graph | RMSE cm |
|---|---|---:|
| ridge_neighbor_residual_mlp | corr_top2_directed | 3.2480 |

Predictive lag-correlation result:

| Model | Graph | RMSE cm |
|---|---|---:|
| ridge_neighbor_residual_mlp | pred_lag1_top2_directed | 3.2535 |

Matched random top-2 controls were worse. This supports the claim that selected basin context helps.

Geographic nearest neighbors also helped for GRACE-only:

| Model | Graph | RMSE cm |
|---|---|---:|
| ridge_neighbor_residual_mlp | geo_incoming_top2 | 3.2925 |

This beat matched random top-2 controls, but it was weaker than the correlation and predictive lag-correlation top-2 models.

### CSR+ERA5

Best tuned correlation-context result:

| Model | Graph | RMSE cm |
|---|---|---:|
| ridge_neighbor_residual_mlp_era5 | corr_top2_directed | 2.7469 |

Predictive lag-correlation result:

| Model | Graph | RMSE cm |
|---|---|---:|
| ridge_neighbor_residual_mlp_era5 | pred_lag1_top2_directed | 2.7499 |

ERA5 improves forecasts beyond CSR GRACE-only. The correlation and predictive-lag neighbor models also beat matched random top-2 controls.

The best geographic residual MLP result with ERA5 was:

| Model | Graph | RMSE cm |
|---|---|---:|
| ridge_neighbor_residual_mlp_era5 | geo_incoming_top2 | 2.7971 |

This did not beat the matched random top-2 controls, so geographic neighbors are supporting evidence for GRACE-only but not the strongest ERA5 result.

## Metrics

The main metrics are:

- RMSE in cm
- MAE in cm
- Pearson correlation
- NSE
- normalized RMSE

Normalized RMSE is:

```text
RMSE / observed test-set standard deviation
```

For by-basin metrics, normalized RMSE uses that basin's own observed test-set standard deviation.

Shareable metric spreadsheets are stored in:

```text
outputs/share_csr_spreadsheets/
```

Recommended files to send first:

- `csr_grace_only_brief_metrics_overall.csv`
- `csr_grace_only_brief_metrics_by_basin.csv`
- `csr_era5_brief_metrics_overall.csv`
- `csr_era5_brief_metrics_by_basin.csv`

## Interpretation

The strongest scientific claim is not that geographic neighbors always help.

The stronger claim is:

> A small number of basins selected by training-period correlation or predictive lag-correlation improves CSR Africa L3 forecasting compared with matched random basin controls.

For CSR, the useful number of selected basins is two. Adding more basins can add noise and reduce performance.

The result should be described as predictive basin context, not proof of hydrologic flow. Correlated basins may share climate forcing, regional storage behavior, GRACE smoothing, or real hydrologic relationships. The current experiments show forecasting value, not causality.

## Key Scripts

Main CSR runners:

- `scripts/run_africa_l3_csr.py`
- `scripts/run_africa_l3_csr_era5.py`

Neighbor/context experiments:

- `scripts/run_correlation_topk_sweep.py`
- `scripts/run_predictive_lag_topk_sweep.py`
- `scripts/run_geographic_topk_sweep.py`
- `scripts/run_random_incoming_top3_control.py`

Shareable spreadsheet output:

- `outputs/share_csr_spreadsheets/`
