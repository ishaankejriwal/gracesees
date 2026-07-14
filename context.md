# GRACE-Only Mask-Region GNN Experiment Context

Last updated: 2026-07-14

## Core Question

Does neighboring-region GRACE/GRACE-FO terrestrial water storage anomaly (TWSA) history improve next-month regional TWSA prediction compared with strong own-region history baselines?

Important constraint: model features are only GRACE/GRACE-FO TWSA time series. Masks/geometries are used only to define nodes, aggregate GRACE grid cells, and build graph edges.

## Data And Pipeline

Raw/current inputs:

- GRACE/GRACE-FO JPL mascon NetCDF: `data/raw/GRCTellus.JPL.200204_202604.GLO.RL06.3M.MSCNv04.nc`
- Uploaded Level 2 mask zips:
  - `masks/L2-20260709T042114Z-3-001.zip`
  - `masks/L2-20260709T042114Z-3-002.zip`
- Uploaded Level 3 mask CSV zip:
  - `masks/L3-20260709T200427Z-2-001.zip`
- Visual reference: `Level 2 Hydrobasins Layout.gif`

Processed global Level 2 data:

- `data/processed/basin_month_grace.csv`: 18,176 rows, 71 regions, 2002-04-17 to 2026-04-16.
- `data/processed/lagged_grace_dataset.csv`: 12,922 rows, 71 regions, 2003-04-01 to 2026-04-01.
- Lag features: `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12`.

Pipeline:

1. Notebook 01 loads an existing basin-month CSV or aggregates GRACE mascon values through `.mask.xyz` files in uploaded mask zips.
2. Notebook 02 creates calendar-safe monthly lags; GRACE gaps create missing lag/target rows that are dropped rather than interpolated.
3. Notebook 03 trains non-graph baselines: persistence, ridge AR, random forest, XGBoost, basin-only NN, and correlation-weighted neighbor baseline.
4. Notebook 04 builds graph variants and trains residual GNNs.
5. Notebook 05 writes metrics, diagnostics, improvement tables, and plots.

Important fixed bug: early GNN outputs inverse-transformed predictions to cm but left observed targets standardized. `src/grace_gnn/models.py` now inverse-transforms GNN observed targets before writing predictions.

## Current Africa Experiment

Current active run:

> Mainland Africa Level 3 GRACE mask-region next-month TWSA forecasting, excluding Madagascar.

Output folder:

- `outputs/africa_l3_no_madagascar/`

Processed Level 3 files:

- `data/processed/basin_month_grace_africa_l3_no_madagascar.csv`
- `data/processed/lagged_grace_dataset_africa_l3_no_madagascar.csv`

Note: the Level 3 lag builder now averages duplicate GRACE entries that fall in the same basin-month before creating lags. The current L3 rerun averaged 74 duplicate basin-month entries, all from 2012-01 and 2015-04, instead of silently keeping the first row.

Level 3 split:

- 37 L3 basins after excluding Madagascar.
- Train: 4,699 rows, 69.8%, 2003-04 to 2021-09.
- Validation: 666 rows, 9.9%, 2021-10 to 2023-03.
- Test: 1,369 rows, 20.3%, 2023-04 to 2026-04.

Level 3 test RMSE:

| Model | Graph type | Test RMSE cm |
|---|---|---:|
| ridge_neighbor_residual_mlp | corr_top3_directed | 2.3769 |
| ridge_neighbor_residual_mlp | random_degree_matched | 2.4707 |
| ridge_residual_mlp | own_lags | 2.5178 |
| xgboost_gnn_embedding_residual | corr_top3_directed | 2.5415 |
| random_forest_gnn_embedding_residual | corr_top3_directed | 2.5434 |
| ridge_gnn_embedding_residual | corr_top3_directed | 2.5695 |
| ridge_neighbor_residual_lstm | corr_top3_directed | 2.5932 |
| ridge_neighbor_ar | corr_top3_directed | 2.6279 |
| ridge_neighbor_residual_tcn | corr_top3_directed | 2.6288 |
| ridge_neighbor_ar | real_knn_undirected | 2.6531 |
| ridge_neighbor_ar | real_knn_reversed | 2.6556 |
| ridge_neighbor_ar | real_knn_directed | 2.6734 |
| ridge_ar | none | 2.7003 |
| ridge_neighbor_ar | random_degree_matched | 2.7120 |
| basin_only_nn | none | 2.7830 |
| residual_neighbor_gnn | random_degree_matched | 2.7909 |
| residual_neighbor_gnn | real_knn_undirected | 2.8088 |
| residual_neighbor_gnn | real_knn_directed | 2.8276 |
| residual_neighbor_gnn | real_knn_reversed | 2.8724 |
| rnn_lag_nn | none | 3.3000 |
| gru_lag_nn | none | 3.3302 |
| random_forest_ar | none | 3.3672 |
| persistence | none | 3.5534 |
| xgboost_ar | none | 3.6633 |
| correlation_neighbor | train_positive_corr_lag1 | 9.3970 |

Interpretation:

- Ridge AR is no longer the best L3 model after adding ridge-anchored residual/neighbor models.
- The best current L3 model is `ridge_neighbor_residual_mlp` with a train-only correlation top-3 neighbor graph. It improves test RMSE from ridge's 2.7003 cm to 2.3769 cm.
- `ridge_residual_mlp` using only own-region lags also beats ridge, which suggests the strongest neural contribution is learning nonlinear residual structure on top of the linear AR baseline rather than replacing it.
- The random degree-matched `ridge_neighbor_residual_mlp` also beats ridge at 2.4707 cm, so the residual MLP gain should be interpreted partly as regularized residual learning rather than only as evidence for meaningful neighbor structure.
- GNN embedding residual follow-ups with ridge, random forest, and XGBoost backbones fall between 2.5415 and 2.5695 cm. They improve on plain ridge but do not beat the simpler ridge-neighbor residual MLP.
- Sequence residual variants using the selected correlation graph underperform the MLP residual correction: LSTM reaches 2.5932 cm and TCN reaches 2.6288 cm.
- Neighbor-augmented ridge models beat plain ridge for the train-correlation graph and the real kNN graphs, while the random degree-matched neighbor ridge is slightly worse than ridge.
- `ridge_neighbor_residual_mlp` beats ridge in 30 of 37 L3 basins, with median per-basin RMSE improvement of about 0.22 cm.
- The L3 kNN GNNs are close to basin-only NN but do not beat the ridge-anchored residual models.
- Added GRU/RNN lag-sequence baselines with basin embeddings. They beat random forest, persistence, XGBoost, and correlation-neighbor, but underperform ridge, basin-only NN, and all current residual GNN variants.
- The original residual GNN random degree-matched graph slightly beats the real centroid-kNN variants, so those GNN results should not be framed as evidence that this kNN graph captures physical hydrologic connectivity.
- RF and XGBoost again underperform the linear autoregressive baseline.

Overall L3 breakdown:

- Best overall model: `ridge_neighbor_residual_mlp` using `corr_top3_directed`, test RMSE 2.3769 cm, MAE 1.5615 cm, Pearson r 0.9818.
- Best own-region-only model: `ridge_residual_mlp`, test RMSE 2.5178 cm. This beats plain ridge by 0.1825 cm without using neighbors.
- Best random-control residual model: `ridge_neighbor_residual_mlp` using `random_degree_matched`, test RMSE 2.4707 cm. This is worse than the correlation-neighbor residual MLP but better than the own-lag residual MLP.
- Best GNN embedding residual model: `xgboost_gnn_embedding_residual` using `corr_top3_directed`, test RMSE 2.5415 cm. It is competitive with the other embedding residual variants but does not improve on the simpler residual MLP models.
- Best purely linear neighbor model: `ridge_neighbor_ar` using `corr_top3_directed`, test RMSE 2.6279 cm. This beats plain ridge by 0.0725 cm.
- Best geographic-kNN neighbor ridge: `ridge_neighbor_ar` using `real_knn_undirected`, test RMSE 2.6531 cm. This beats plain ridge by 0.0473 cm.
- Plain ridge AR remains a strong baseline at 2.7003 cm, but the ridge-anchored residual models now provide the clearest L3 gain.
- Standalone neural models are weaker: basin-only NN is 2.7830 cm, residual GNNs are 2.7909-2.8724 cm, and RNN/GRU lag models are 3.3000-3.3302 cm.
- The best improvement is not from replacing ridge with a neural network. It is from keeping ridge as the backbone and learning residual corrections, especially with train-correlation neighbor lags.
- The correlation-neighbor result is stronger than centroid-kNN for L3, matching the broader pattern that statistical similarity graphs can outperform simple geographic-nearest graphs for this GRACE-only task.

## Africa L3 Walk-Forward Robustness

Source files:

- Runner: `scripts/run_africa_l3_walk_forward_top5.py`
- Outputs: `outputs/africa_l3_no_madagascar/walk_forward_top5/`
- Main tables: `predictions_walk_forward.csv`, `metrics_by_fold.csv`, `metrics_summary.csv`, `rankings_by_fold.csv`, `graph_audit.csv`

This is a robustness check for the current RMSE top-five Africa L3 models. It does not replace the canonical final 70/10/20 split above. Each fold preserves time order: training starts at 2003-04 and expands, validation is the 12 available GRACE dates immediately before the test block, and test is the next 24 available GRACE dates. Because GRACE has missing months, a 24-date fold is not always a clean two-calendar-year span.

Walk-forward summary:

| Model | Graph type | Mean RMSE cm | Median RMSE cm | Worst-fold RMSE cm | Mean rank | Rank-1 folds |
|---|---|---:|---:|---:|---:|---:|
| ridge_neighbor_residual_mlp | corr_top3_directed | 2.5207 | 2.4813 | 2.9323 | 1.0 | 5/5 |
| xgboost_gnn_embedding_residual | corr_top3_directed | 2.6456 | 2.5321 | 2.9700 | 2.8 | 0/5 |
| random_forest_gnn_embedding_residual | corr_top3_directed | 2.6681 | 2.5742 | 3.0581 | 3.4 | 0/5 |
| ridge_neighbor_residual_mlp | random_degree_matched | 2.7069 | 2.6950 | 3.0284 | 3.8 | 0/5 |
| ridge_residual_mlp | own_lags | 2.7131 | 2.6763 | 3.0383 | 4.0 | 0/5 |

Interpretation:

- The old headline result was a single final split: `ridge_neighbor_residual_mlp | corr_top3_directed` won on 2023-04 to 2026-04 with 2.3769 cm test RMSE.
- The walk-forward audit asks whether that result is stable across earlier chronological test windows among the selected top-five models. It is: the same model ranks first in all five folds.
- The mean walk-forward RMSE is higher than the final-split RMSE because it averages across multiple periods with different difficulty, not because it reruns the same test window.
- `corr_top3_directed` is rebuilt inside each fold from that fold's training period only. `graph_audit.csv` records the train period and edge count for each fold.
- The random degree-matched residual MLP remains useful as a control. It performs worse than the train-correlation neighbor model in the walk-forward audit, but it is close to the own-lag residual MLP, so the neighbor result should still be framed as predictive structure plus residual regularization rather than pure physical neighbor transfer.

## Africa L2 Historical Baseline

> Mainland Africa Level 2 GRACE mask-region next-month TWSA forecasting, excluding Madagascar.

Regions:

- Greater Nile Coastal
- Southern Africa Coastal
- West Africa Coastal
- South Central Africa Coastal
- North Africa Coastal
- East Africa Coastal
- Chad Endorheic

Output folder: `outputs/africa_l2_no_madagascar/`

Corrected Africa L2 test RMSE:

| Model | Graph type | Test RMSE cm |
|---|---|---:|
| ridge_ar | none | 1.9869 |
| basin_only_nn | none | 2.1034 |
| residual_neighbor_gnn | real_knn_undirected | 2.2018 |
| residual_neighbor_gnn | real_knn_reversed | 2.2483 |
| residual_neighbor_gnn | real_knn_directed | 2.2782 |
| residual_neighbor_gnn | random_degree_matched | 2.4602 |
| random_forest_ar | none | 2.8786 |
| xgboost_ar | none | 2.8835 |
| persistence | none | 2.8921 |
| correlation_neighbor | train_positive_corr_lag1 | 6.5513 |

Interpretation:

- Ridge AR wins overall because own-region lag history is the most stable signal.
- Real GNNs beat the random graph but do not beat ridge or basin-only NN overall.
- GNN helps strongly in West Africa, but neighbor mixing hurts or adds little in several other Africa regions.
- North Africa has very low test variability (`test_std_cm` about 0.60), so its absolute RMSE values are small but not especially strong after normalization.

## Africa Run Comparison

Best test RMSE by Africa run:

| Run | Best model | Graph type | Test RMSE cm | Notes |
|---|---|---|---:|---|
| Africa L2 no Madagascar | ridge_ar | none | 1.9869 | Coarser 7-region problem; own-region AR wins. |
| Africa L3 no Madagascar | ridge_neighbor_residual_mlp | corr_top3_directed | 2.3769 | Finer 37-region problem; ridge plus correlation-neighbor residual correction wins. |
| Africa L3 walk-forward top-five audit | ridge_neighbor_residual_mlp | corr_top3_directed | 2.5207 mean | Same model ranks first in all five chronological robustness folds. |

How to read this:

- Absolute RMSE is not directly comparable across L2 and L3 as a pure model-quality score because the region definitions and test variability differ.
- Africa L2 has the lowest RMSE partly because it has only 7 broad regions, so regional averaging smooths the target.
- Africa L3 is the stronger test of finer spatial structure. In that setting, plain ridge is strong, but adding correlation-neighbor lags plus a residual MLP produces the best current L3 result.
- The walk-forward audit supports the final-split ranking for the selected top-five L3 models, but it should be reported as a robustness/stability check rather than a replacement benchmark.
- The emerging Africa pattern is: own-region lag history is the foundation; neighbor information helps most when added carefully as residual/auxiliary structure rather than used as a standalone replacement.

## Graph And Model Notes

Current graph variants:

- `real_knn_directed`: 3-nearest mask-centroid graph.
- `real_knn_undirected`: symmetrized kNN graph.
- `real_knn_reversed`: reversed directed kNN graph.
- `random_degree_matched`: placebo preserving source out-degree.
- `corr_top3_directed`: train-period positive-correlation top-3 graph.

Important graph caveats:

- Current real edges are nearest-mask-region or correlation edges, not verified hydrologic flow edges.
- Directed edge semantics matter: graph normalization stores messages as `adj[dst, src]`, so `src -> dst` means `dst` receives from `src`.
- Useful spatial dependence could reflect hydrologic connection, shared climate, GRACE smoothing/leakage, or broad regional storage behavior.

## Reporting Rules

Do not claim:

- The GNN proves physical water movement between basins.
- Current kNN/correlation edges are hydrologic flow connectivity.
- Level 2 experiments are fine Africa sub-basin experiments.

Acceptable phrasing:

> This tests whether neighboring coarse GRACE mask-region history improves next-month regional TWSA forecasts. It does not test fine sub-basin hydrologic transfer.

Best current summary:

> Own-region lag history is a very strong baseline. Plain ridge wins the older Africa L2 run, while Africa L3 is now best with ridge plus train-correlation neighbor lags and a small residual MLP. A five-fold walk-forward audit of the current L3 top-five models keeps that same model ranked first in every fold. The correlation graph should be described as statistical similarity structure, not proven hydrologic flow.

## Near-Term Direction

The next scientific improvement is data/graph definition and robustness reporting, not larger neural networks:

- Keep the same GRACE-only feature rule unless the scientific question changes.
- Keep walk-forward results as a robustness audit beside the canonical final split.
- Add validation checks for mask count, parsed region names, basin IDs, monthly row counts, graph edge counts, and split counts.
- Prefer true topology or carefully justified graph definitions over presenting centroid kNN as hydrologic adjacency.
