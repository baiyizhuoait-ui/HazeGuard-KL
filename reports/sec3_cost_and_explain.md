# Section 3: Cost-Sensitive Optimization, Model Interpretability and Generalization

## 3.1 Cost-Sensitive Threshold Optimization (M7)

### 3.1.1 Objective

The HazeGuard-KL system predicts four ordinal haze-risk levels (Level 0–3). Since underestimating severe haze pollution may result in delayed warnings, the project uses an asymmetric cost matrix that assigns higher penalties to under-prediction than over-prediction.

The objective of M7 is to reduce the Average Asymmetric Cost through decision-threshold optimization while preserving the original trained model.

### 3.1.2 Methodology

A three-dimensional grid search was implemented over three ordinal decision thresholds:

\[
\tau = (\tau_1,\tau_2,\tau_3)
\]

Each threshold was searched from 0.05 to 0.95 with a step size of 0.05, resulting in:

\[
19^3 = 6859
\]

candidate threshold combinations.

The optimization objective was:

\[
\min_{\tau} \frac{1}{N}\sum_{i=1}^{N}C[y_i,\hat{y}_i(\tau)]
\]

where \(C\) is the project's predefined asymmetric cost matrix.

Threshold selection was performed exclusively on the validation set containing 16,656 samples. No GPSO or evolutionary optimization algorithm was used.

### 3.1.3 Experimental Results

The optimal thresholds were:

- τ₁ = 0.20
- τ₂ = 0.30
- τ₃ = 0.45

The validation-set results are summarized below.

| Metric | Result |
|---|---:|
| Default Average Asymmetric Cost | 0.5620 |
| M7 Grid Search Cost | 0.4495 |
| Bayesian Decision Cost | 0.4890 |
| Macro-F1 | 0.6884 |
| Balanced Accuracy | 0.7143 |
| High-Risk Recall | 0.9360 |
| MAOE | 0.3100 |
| Severe Jumping Error Rate | 0.0162 |

Compared with the default thresholds, grid search reduced the validation asymmetric cost by approximately 20.01%.

The achieved High-Risk Recall of 93.60% indicates that the optimized decision thresholds identified a high proportion of actual Level 2/3 cases.

However, cost-sensitive optimization involves trade-offs. Improvements in high-risk detection do not necessarily imply improvements in all classification metrics.

### 3.1.4 Bayesian Decision Comparison

A Bayesian minimum expected cost decision rule was used as a theoretical comparison:

\[
\hat y = \arg\min_j \sum_i P(Y=i\mid x)C[i,j]
\]

The Bayesian method achieved an empirical validation cost of 0.4890, compared with 0.4495 for grid search.

This comparison should not be interpreted as evidence that grid search is theoretically superior to Bayesian decision-making. Differences may be influenced by model probability calibration and the specific validation distribution.

The final thresholds were saved in `models/cost_thresholds.json`.

### 3.1.5 Independent Test-Set Evaluation and Performance Trade-offs

After selecting the optimal thresholds on the validation set, the M7 Grid Search decision rule was evaluated once on the independent test set containing 16,647 samples. The selected thresholds were **[0.20, 0.30, 0.45]** and were fixed before test evaluation. No threshold tuning or model selection was performed using the test labels.

Three decision strategies were compared using the same test-set probability predictions: the default decision rule, the M7 cost-sensitive grid-search thresholds, and the Bayesian minimum expected cost decision rule.

**Table: Independent test-set performance comparison**

| Metric | Default | M7 Grid Search | Bayesian |
|---|---:|---:|---:|
| Macro-F1 | 0.6972 | 0.6659 | 0.6247 |
| Balanced Accuracy | 0.7192 | 0.6656 | 0.6524 |
| High-Risk Recall | 0.8737 | **0.9376** | 0.9594 |
| MAOE | 0.3013 | 0.3300 | 0.3993 |
| Severe Jumping Error Rate | 0.0171 | 0.0173 | 0.0320 |
| Avg Asymmetric Cost | 0.6497 | **0.5405** | 0.5458 |

**High-risk recall and asymmetric cost improvement.**

Compared with the default decision rule, M7 increased High-Risk Recall from **0.8737 to 0.9376**, an improvement of **6.39 percentage points**. At the same time, Avg Asymmetric Cost decreased from **0.6497 to 0.5405**, representing a relative reduction of approximately **16.81%**.

These improvements indicate that the validation-selected cost-sensitive thresholds successfully transferred to the independent test set. M7 identified a larger proportion of genuinely high-risk haze cases while reducing the average penalty defined by the project's asymmetric cost matrix.

**Macro-F1 deterioration and performance trade-off.**

The improvement in high-risk detection was not achieved without sacrificing other classification metrics. In particular, **Macro-F1 decreased from 0.6972 to 0.6659**, an absolute reduction of **0.0313**. Balanced Accuracy also decreased from 0.7192 to 0.6656, while MAOE increased from 0.3013 to 0.3300. The Severe Jumping Error Rate rose slightly from 0.0171 to 0.0173.

These results demonstrate that M7 is not universally superior to the default classifier. Its threshold adjustment changes the balance between missed hazardous events and precautionary warnings, prioritizing the project's asymmetric cost objective rather than overall class-balanced classification performance.

**Justification based on asymmetric misclassification costs.**

The project's cost matrix assigns substantially higher penalties to certain dangerous underprediction errors than to many overprediction errors. For example, predicting Level 0 when the true condition is Level 2 incurs **8 cost units**, while predicting Level 0 when the true condition is Level 3 incurs **15 cost units**. By comparison, several less severe errors incur penalties of only 1–5 cost units.

Therefore, a reduction in Macro-F1 may be acceptable when it accompanies fewer costly high-risk misses and a meaningful reduction in expected operational loss. The observed trade-off is consistent with a precautionary haze-warning application in which missing severe pollution conditions can have greater consequences than issuing some additional warnings.

Nevertheless, this justification depends on the validity of the specified cost matrix. M7 should be considered a **cost-sensitive operational alternative**, rather than a classifier that dominates the default approach on every metric.

**Comparison with the Bayesian decision rule.**

The Bayesian minimum expected cost strategy achieved the highest High-Risk Recall of 0.9594, but its test-set Avg Asymmetric Cost was 0.5458, slightly higher than M7's 0.5405. It also produced a lower Macro-F1 of 0.6247, a higher MAOE of 0.3993, and a higher Severe Jumping Error Rate of 0.0320.

This comparison provides a useful theoretical decision-making reference. Although the Bayesian rule minimizes expected cost for each probability vector under the specified cost matrix, its empirical performance depends on the quality and calibration of those probabilities. Consequently, it is not guaranteed to achieve the lowest realized cost on a finite test set.

**Conclusion.**

The independent test evaluation supports the use of M7 when the primary objective is to reduce the operational consequences of missed high-risk haze events. Relative to the default decision rule, M7 improved High-Risk Recall by **6.39 percentage points** and reduced Avg Asymmetric Cost by approximately **16.81%**, while lowering Macro-F1 by **0.0313**.

The result should therefore be presented as an explicit trade-off: **better high-risk protection and lower asymmetric cost at the expense of some general classification performance**. All M7 thresholds were selected using validation data only, and the independent test results were used exclusively for final evaluation.

*Results source: `models/m7_test_metrics.json`; threshold-selection record: `models/cost_thresholds.json`.*

## 3.2 Model Interpretability

### 3.2.1 Permutation Feature Importance

Permutation Importance was applied to the M6 Ordinal Random Forest using validation data.

The evaluation used 1,000 sampled validation observations and five permutation repetitions, with Macro-F1 as the scoring metric.

The results indicate that `pm2_5` is the most influential feature, followed by `hour_sin` and recent PM2.5 historical features.

This suggests that the model relies strongly on current pollution concentration and temporal patterns when predicting future haze risk.

The contribution of upstream spatial features was comparatively limited in this experiment. However, low permutation importance does not establish that upstream pollution transport is physically unimportant. Correlated predictors may reduce the apparent individual contribution of a feature.

Figure: `figures/fig_perm_importance.png`

### 3.2.2 Partial Dependence Analysis

Partial Dependence Plots were generated for wind speed and relative humidity using the M1 Baseline Decision Tree.

The baseline model was selected for this analysis because the M6 Ordinal Random Forest did not include the original `wind_speed_ms` and `relative_humidity_2m` variables among its 25 selected features.

Both PDP curves were relatively flat across the evaluated ranges, indicating limited variation in the baseline model's average predicted high-risk probability when these features were varied individually.

These plots describe model behavior rather than causal effects. Correlations between meteorological conditions and pollution features may limit the interpretation of partial dependence.

Figure: `figures/fig_pdp_wind_humidity.png`

## 3.3 Generalization Study

### 3.3.1 Spatial Generalization: Leave-One-Station-Out

A 26-fold Leave-One-Station-Out (LOSO) experiment was conducted using the M1 Decision Tree configuration.

The spatial dataset contained:

- 3 receptor nodes
- 10 corridor nodes
- 13 source nodes

For each fold, one node was excluded from training and used for evaluation.

A temporal target-consistency audit was performed before the experiment. Of 966,677 records, 966,510 satisfied the exact three-hour target verification criteria. The remaining 167 records were excluded from this analysis because their targets could not be verified against available future observations.

The receptor-node results were:

| Station | Macro-F1 | High-Risk Recall | Avg Asymmetric Cost |
|---|---:|---:|---:|
| KL_City | 0.5653 | 0.6598 | 1.3262 |
| Port_Klang | 0.5947 | 0.5543 | 0.4259 |
| Sepang | 0.6020 | 0.6521 | 1.0013 |

The mean receptor High-Risk Recall was 0.6221, with an average asymmetric cost of 0.9178 across the three receptor folds.

These results suggest that the baseline model faces challenges when predicting previously unseen spatial nodes, particularly in high-risk detection.

The corridor and source nodes represent spatial background or transport-related grid locations and should not be described as independent ground-monitoring stations.

Additionally, this experiment reused previously constructed features. Therefore, the findings should be interpreted as a spatial holdout experiment with partially precomputed inputs, rather than a fully leakage-audited end-to-end spatial validation.

Results: `models/loso_results.csv`

### 3.3.2 Temporal Generalization: 2025 to 2026

A temporal holdout experiment was conducted by training a fresh M1 Decision Tree using receptor observations from 2025 and earlier, followed by evaluation on 2026 observations.

The experiment used 89,677 training samples and 21,870 evaluation samples.

| Metric | 2026 Evaluation |
|---|---:|
| Macro-F1 | 0.6741 |
| Balanced Accuracy | 0.6579 |
| High-Risk Recall | 0.7949 |
| MAOE | 0.3089 |
| Severe Jumping Error Rate | 0.0171 |
| Avg Asymmetric Cost | 0.8171 |

The model achieved a High-Risk Recall of 79.49% on the 2026 holdout data.

This demonstrates measurable predictive performance on a later period, but a historical comparison is required before quantifying temporal performance degradation.

### 3.3.3 Seasonal Generalization

Seasonal generalization was evaluated using the 2026 temporal holdout predictions from the M1 Baseline Decision Tree. The evaluation samples were divided into two periods: June–October (the project's defined monsoon-related period) and the remaining available months (January–May).

The same model, trained exclusively on observations from 2025 and earlier, was used for both groups. No seasonal retraining or threshold optimization was performed.

The seasonal evaluation results are summarized below.

| Metric | June–October | January–May |
|---|---:|---:|
| Number of Samples | 10,998 | 10,872 |
| Macro-F1 | 0.6812 | 0.6625 |
| High-Risk Recall | 0.8127 | 0.7705 |
| Average Asymmetric Cost | 0.8956 | 0.7378 |

The June–October period achieved a higher Macro-F1 and High-Risk Recall than January–May. Specifically, High-Risk Recall increased by 4.22 percentage points, from 77.05% to 81.27%.

However, the Average Asymmetric Cost was also higher during June–October, increasing from 0.7378 to 0.8956, approximately 21.39% higher than the January–May value.

This indicates that improved high-risk recall does not necessarily lead to a lower overall asymmetric prediction cost. Differences in risk-level distributions and the severity of classification errors may contribute to this result.

The findings suggest that model performance varies across the evaluated seasonal periods. Nevertheless, the comparison is descriptive rather than causal. Since the available 2026 observations end in October, the January–May group does not represent a complete November–May seasonal cycle.

Further analysis of class distributions and error types would be required to explain the observed cost differences.

The results were saved in `models/seasonal_metrics.csv`.

## 3.4 Limitations

The experimental results should be interpreted with the following limitations:

1. The M7 threshold-search results were selected on the validation set. Validation performance should not be treated as independent test performance.
2. The permutation analysis explains M6, whereas the PDP and generalization experiments use an M1 Decision Tree configuration.
3. The 26 spatial nodes include modeled background, corridor and source locations rather than 26 independent ground-monitoring stations.
4. Existing upstream feature engineering may introduce dependencies that are not fully addressed by the model-level holdout procedures.
5. Partial Dependence and Permutation Importance measure statistical model behavior rather than physical causality.
6. Temporal results are limited to the available observation period, ending in October 2026.

These limitations are important when considering the reliability and deployment readiness of the proposed haze early-warning system.
