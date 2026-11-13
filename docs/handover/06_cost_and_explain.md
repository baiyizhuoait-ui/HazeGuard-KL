# Member 06 Handover — Cost-Sensitive Optimization, Interpretability and Generalization

## 1. Overview

Member 06 implemented three major components in accordance with the HazeGuard-KL v3.1 project specification:

1. M7 cost-sensitive threshold optimization using grid search.
2. Model interpretability using Permutation Importance and Partial Dependence Plots.
3. Generalization experiments covering spatial, temporal and seasonal performance.

The implementation does not use GPSO, evolutionary optimization or SHAP.

## 2. Source Code

| File | Description |
|---|---|
| `src/06_cost_thresholds.py` | Validation-based cost-sensitive grid search and Bayesian comparison |
| `src/06_m7_final_test.py` | Independent final test evaluation using frozen thresholds |
| `src/06_interpretability.py` | M6 Permutation Importance and M1 Partial Dependence Plots |
| `src/06_generalization_study.py` | LOSO, temporal holdout and seasonal evaluation |

## 3. M7 Cost-Sensitive Threshold Optimization

### 3.1 Method

M7 uses a three-dimensional grid search over ordinal decision thresholds.

- Grid range: 0.05–0.95
- Step size: 0.05
- Threshold combinations: 6,859
- Optimization objective: `Avg_Asymmetric_Cost`
- Threshold selection dataset: validation only
- Evaluation: unified `src/utils_metrics.py`

The asymmetric cost matrix is imported from `DEFAULT_COST_MATRIX` rather than redefined independently.

### 3.2 Selected Thresholds

The validation-selected thresholds are:

- τ₁ = 0.20
- τ₂ = 0.30
- τ₃ = 0.45

They are saved in `models/cost_thresholds.json`.

### 3.3 Results

| Metric | Validation | Test |
|---|---:|---:|
| M7 Average Asymmetric Cost | 0.4495 | 0.5405 |
| M7 Macro-F1 | 0.6884 | 0.6659 |
| M7 High-Risk Recall | 0.9360 | 0.9376 |

Compared with default thresholds, M7 reduced Average Asymmetric Cost by approximately 20.01% on validation and 16.81% on the independent test set.

The final test results are stored in `models/m7_test_metrics.json`.

**Important:** The thresholds are frozen. The test set must not be reused for threshold tuning.

## 4. Model Interpretability

### 4.1 Permutation Importance

The M6 Ordinal Random Forest is evaluated using validation observations.

- Sample size: 1,000
- Permutation repetitions: 5
- Evaluation metric: Macro-F1
- Output: `figures/fig_perm_importance.png`

The results show that `pm2_5` is the most influential feature, followed by `hour_sin` and recent PM2.5 features.

### 4.2 Partial Dependence Plots

Wind speed and relative humidity PDPs are generated using the M1 Baseline Decision Tree.

The original wind speed and humidity variables were not selected as M6 input features. Therefore, M1 was used to provide the requested analysis.

Output: `figures/fig_pdp_wind_humidity.png`

The plots show relatively small variations in average predicted high-risk probability over the evaluated feature ranges.

These findings describe model behavior and should not be interpreted as physical causal effects.

## 5. Generalization Experiments

### 5.1 Spatial Generalization — LOSO

The experiment covers 26 spatial nodes:

- 3 receptor nodes
- 10 corridor nodes
- 13 source nodes

Each fold excludes one node from model training. A fresh M1 Decision Tree is fitted using the training portion of the remaining nodes and evaluated using the held-out node's validation observations.

A target-consistency audit retained 966,510 of 966,677 original records. The other 167 records could not be verified against available exact three-hour future observations and were excluded from this experiment.

Mean receptor-node results:

- Macro-F1: 0.5873
- High-Risk Recall: 0.6221
- Average Asymmetric Cost: 0.9178

Output: `models/loso_results.csv`

**Limitation:** Corridor and source nodes are not independent ground-monitoring stations. The experiment also uses precomputed features, and full upstream spatial-leakage independence has not been established.

### 5.2 Temporal Generalization

The temporal holdout experiment uses receptor observations from 2025 and earlier for model training and 2026 observations for evaluation.

- Training samples: 89,677
- Evaluation samples: 21,870
- Evaluation Macro-F1: 0.6741
- Evaluation High-Risk Recall: 0.7949
- Evaluation Average Asymmetric Cost: 0.8171

The implementation uses a fresh M1 Decision Tree and excludes 2026 observations from training.

### 5.3 Seasonal Generalization

The same trained temporal-holdout model is evaluated over two 2026 periods.

| Period | Samples | Macro-F1 | High-Risk Recall | Avg Asymmetric Cost |
|---|---:|---:|---:|---:|
| June–October | 10,998 | 0.6812 | 0.8127 | 0.8956 |
| January–May | 10,872 | 0.6625 | 0.7705 | 0.7378 |

Output: `models/seasonal_metrics.csv`

The available 2026 records end in October. Therefore, the January–May group does not represent a complete non-monsoon year.

## 6. Dependencies

The following upstream artifacts are required to reproduce the experiments:

**Member 01 / shared project**

- `data/haze_panel_clean.parquet`
- `models/baseline_dt.joblib`
- `src/utils_metrics.py`

**Member 03**

- `data/features_final.parquet`
- `models/selected_features.json`

**Member 05**

- `models/ordinal_rf.joblib`
- `models/ordinal_probs_val_test.npz`
- `src/ordinal_forest.py`

The original model artifacts were serialized using scikit-learn 1.9.0. Loading them under scikit-learn 1.9.1 produced compatibility warnings. The environment should be aligned with the original training environment when reproducing the final results.

## 7. Execution Instructions

Run from the repository root using the activated Python virtual environment.

Cost threshold optimization:

`python src/06_cost_thresholds.py`

Permutation Importance:

`python src/06_interpretability.py`

Wind-speed and humidity PDP:

`python src/06_interpretability.py --pdp`

Two-node LOSO pilot:

`python src/06_generalization_study.py`

Complete 26-node LOSO:

`python src/06_generalization_study.py --all`

Temporal holdout and seasonal evaluation:

`python src/06_generalization_study.py --temporal`

The M7 final test evaluation is implemented in `src/06_m7_final_test.py`, but **should not be rerun routinely**. The locked test results already exist in `models/m7_test_metrics.json`.

Reproduction should preserve the original training configurations, dataset versions and feature definitions.

## 8. Outputs for Member 07

Member 07 can integrate the following outputs into the Streamlit dashboard and final report:

| Output | Purpose |
|---|---|
| `models/cost_thresholds.json` | Frozen M7 decision thresholds and validation metrics |
| `models/m7_test_metrics.json` | Independent final test comparison |
| `models/loso_results.csv` | Spatial generalization results |
| `models/seasonal_metrics.csv` | Seasonal generalization results |
| `figures/fig_perm_importance.png` | M6 permutation feature importance |
| `figures/fig_pdp_wind_humidity.png` | M1 PDP analysis |
| `figures/fig_roc_high_risk.png` | Validation high-risk ROC |
| `figures/fig_confusion_matrices_comparison.png` | Validation confusion matrix comparison |
| `reports/sec3_cost_and_explain.md` | Report section covering Member 06 experiments |

The dashboard should read the saved results rather than rerun the optimization or training code during normal use.

The team-wide `models/all_models_benchmark.csv` must also be coordinated with the other members. Its model names and metric definitions should follow the shared v3.1 schema.

## 9. Important Limitations

- M7 thresholds were optimized using validation data only.
- Independent test data must not be reused for model selection.
- Permutation Importance analyzes M6, whereas PDP and generalization experiments use M1.
- The spatial experiment does not establish complete independence of upstream feature engineering.
- Model interpretation results describe learned predictive relationships rather than causal effects.
- Source and corridor nodes must not be presented as independent ground-monitoring stations.
- Seasonal results reflect available 2026 data only.

## 10. Handover Status

The main M7, interpretability and generalization experiments have been implemented and their primary outputs generated.

Remaining integration tasks include validating upstream artifact availability, incorporating the results into the team-wide model benchmark, and checking the final dashboard and report against the v3.1 specification.