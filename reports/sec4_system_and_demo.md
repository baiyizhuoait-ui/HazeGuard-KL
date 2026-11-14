# Section 4: System and Demonstration

## 4.1 System Role and Design Principles

Sections 2 and 3 establish the three-hour forecast task, the regression and ordinal models, and the cost of changing how a high-risk alert is issued. This section uses historical replay at KL City, Port Klang and Sepang to assess what those choices mean for residents, schools and event organisers planning outdoor activity in the Klang Valley. [1, 2]

At issue time *t*, the system associates a risk level and concentration estimate with the valid time *t* + 3 h. The target levels L0 through L3 are defined by CAMS PM2.5 boundaries of 15, 35 and 55 µg/m³. L2 and L3 together constitute high risk. The default M5 policy converts a random-forest mean concentration forecast into a level using validation-selected cut points of 17, 33 and 55 µg/m³. The replay also supports M5 with fixed cut points, M6 with the most probable level, and M7 with the frozen cumulative probability thresholds. The concentration and adaptive interval come from M5. The class probabilities come from M6. A qualitative reliability note combines maximum class probability, raw M5 tree spread and whether that spread crosses a level boundary. Its reference quantiles come from validation. [1, 3, 4]

## 4.2 Data Flow and Integrity Checks

The replay consumes saved validation and test predictions. Member 5 supplied M5 point forecasts and levels in stage5_predictions.npz, M6 class probabilities in ordinal_probs_val_test.npz, and adaptive interval bounds in stage5_intervals_adaptive.npz. Member 6 supplied the M7 rule in cost_thresholds.json. M7 checks the probabilities of reaching L1, L2 and L3 in sequence against 0.20, 0.30 and 0.45. The v1 feature table supplies the CAMS values at issue and valid times. The cleaned panel provides context for the full 26-node grid. The replay fits no models and confines exploratory changes to the alert thresholds to validation rows. [1, 3, 4]

The system associates forecast records, interval bounds and CAMS outcomes by node and issue time. The M5 and M6 prediction arrays are read in their stored order. Repository tests verify equal row counts and agreement between the M6 probability argmax and saved M6 levels. They also test the interval keys and bound ordering, onset logic, calibration bins and benchmark schema. At startup, the system uses the shared metric implementation to check the M7 validation cost against the recorded value of 0.4495, with a tolerance of 0.0005. The reconciled test benchmark places M1 and the other registered rows on the same 16,647 receptor records. The benchmark labels M8 as v2. The Stage 7 results file records input hashes and the definitions behind the replay metrics. [1, 2, 4, 5]

## 4.4 Early Warning Performance

The replay analysis uses saved forecasts on an hourly grid by receptor and split. An onset is the first CAMS hour at or above 35 µg/m³ after six consecutive hours with available concentrations below that threshold. An episode continues while consecutive hours remain at or above 35 µg/m³. For each onset, the analysis searches for the earliest forecast of L2 or L3 issued from three hours before onset through the earlier of the episode’s last high-risk hour or 12 hours after onset. Lead is onset time minus issue time. A positive lead means the alert was issued before onset, with three hours as the maximum by construction. An episode is counted as missed if the search finds no alert. This matching rule associates alerts with an onset window. A forecast’s valid hour can fall after the last high-risk hour. [2]

| Test onset or alert measure | M5 tuned (default) | M7       |
|---------------------------------|------------------------|--------------|
| Alert issued 3 h before onset   | 225                    | 281          |
| Alert issued 2 h before onset   | 68                     | 57           |
| Alert issued 1 h before onset   | 39                     | 29           |
| Alert issued at onset           | 28                     | 23           |
| Alert issued after onset        | 17                     | 10           |
| Missed onsets                   | 42                     | 19           |
| Alert issued before onset       | 79.2%                  | 87.6%        |
| Full 3 h lead                   | 53.7%                  | 67.1%        |
| Fresh alerts confirmed          | 82.7% of 427           | 71.0% of 407 |

Table 4.1. Test replay results across 419 onsets at the three receptor nodes. The fresh-alert row uses a separate denominator for each rule. [2]

A fresh alert follows six consecutive forecast hours below L2. It is confirmed when at least one available CAMS hour is high risk within the window from one hour before to three hours after forecast validity. M7’s higher early-warning share is consistent with its higher per-row test recall for high-risk hours (0.9376 versus 0.8788 for tuned M5). The lower confirmation rate shows a cost of acting earlier in replay. Pure persistence uses the current level as the forecast, so it cannot alert before an onset preceded by six low-risk hours. [2, 4, 5, 6]

Of the 223 test episodes with a peak of at least 55 µg/m³, M5 alerted before onset in 91.0% and M7 in 96.0%. Each flagged every major episode within the search window, which allows alerts at and after onset. M5 alerted before 65.8% of the 196 episodes below that peak threshold. All 42 M5 misses occurred in this lower-peak group. [2]

The comparison baseline was a trailing 24-hour mean of the same CAMS series, with at least 18 available hours and a 35 µg/m³ threshold. The mean was already high one hour before 162 of the 419 test onsets. Of the remaining 257, it reached the threshold within the available 48-hour search window for 142, with a median lag of 10 hours among those crossings. No crossing was found for the other 115. This comparison characterises smoothing delay in the replay. The baseline is a project proxy for a daily-average index, and the analysis makes no timing comparison with Malaysia’s official Air Pollutant Index. [2]

## 4.5 Effect of the CAMS Run Boundary

Section 2.1.5 identifies a discontinuity where successive CAMS runs meet at 08:00 and 20:00 local time. At KL City, the mean absolute hourly change at those times was 3.22 times the change at other hours. The test onset counts were 33 at 08:00, 99 at 19:00 and 115 at 20:00. Together, those hours accounted for 247 of the 419 onsets. [2, 6]

| Test onset hour | Onsets | M5 before onset | M5 full 3 h lead | M5 missed |
|---------------------|------------|---------------------|----------------------|---------------|
| 08:00 or 20:00      | 148        | 75.0%               | 69.6%                | 18            |
| Other hours         | 271        | 81.5%               | 45.0%                | 24            |

Table 4.2. Test onsets by local CAMS run-boundary hour. [2, 6]

The tuned M5 rule attained the full three-hour lead more often at the run boundary, even though it warned before onset more often in the other-hours group. The concentration of full-horizon alerts at recurring clock times suggests that part of the measured warning skill reflects the structure of the CAMS product. The 19:00 peak falls in the other-hours group, so this split cannot account for all onset clustering near the evening boundary. These results therefore do not establish how much lead would be available for a ground-measured haze episode. [2, 6]

## 4.6 Probability Calibration, Concentration Error and Interval Coverage

Calibration of M6’s probability of L2 or higher was assessed in ten equal-width bins. Expected calibration error, the sample-weighted absolute gap between bin mean probability and observed frequency, was 0.0375 on validation and 0.0261 on test. The corresponding Brier scores were 0.0776 and 0.0907. Predicted probabilities generally exceeded the observed frequencies in the low and middle bins. The differing patterns across the two metrics show that a smaller average bin gap can coexist with a larger squared probability error. The calibration gaps also help interpret why the minimum-expected-cost policy in Section 3 could have a higher realised cost than M7 despite using the same probability estimates. The calibration analysis alone does not establish the cause of that difference. [2, 4]

The test concentration forecast error (M5 forecast minus outcome) averaged +3.70, +1.92, −1.29 and −16.10 µg/m³ across true levels L0 to L3. The signed errors indicate that the forest’s mean forecast attenuated extremes, particularly steep rises. Across all test rows M5 had a mean absolute error (MAE) of 8.07 µg/m³, compared with 10.53 µg/m³ for concentration persistence. The test results below show where that improvement came from. [2]

| Test PM2.5 change from issue to valid time | M5 mean error | M5 MAE | Persistence MAE |
|------------------------------------------------|-------------------|------------|---------------------|
| Rising by more than 10 µg/m³                   | −15.05            | 17.98      | 24.42               |
| Steady within ±10 µg/m³                        | −0.63             | 4.94       | 3.98                |
| Falling by more than 10 µg/m³                  | +4.96             | 10.03      | 20.68               |

Table 4.3. Forecast error by realised three-hour change on the test split. Error values are in µg/m³. [2]

The adaptive concentration interval is centred on the M5 tree median and scales with the tree spread. Its scale multiplier was calibrated using absolute residuals normalised by that spread. At nominal 90% coverage, it achieved 90.0% on validation with a mean width of 25.4 µg/m³, and 88.0% on test with a mean width of 32.1 µg/m³. KL City had the lowest coverage on both splits at 87.7% on validation and 84.8% on test. The raw tree band covered 71.8% on test and the fixed-width calibration control covered 83.4%. The adaptive method was selected on forward validation blocks after the original test results were known. Validation was also used for model selection. Temporal dependence, distribution shift and the shortfall in observed test coverage limit the interpretation of the nominal target. [3, 7]

## 4.7 M8 Feature Table Control for KD-01

The registered v1 table encodes hour of day with hour_sin alone, which collapses 24 hours into 13 encoded values. In particular, 04:00 and 08:00 share a value, as do 16:00 and 20:00. The v2 Top-25 set includes hour_cos in place of gran_low_12h. M8 uses v2 with M5’s forest hyperparameters and tree-mean forecast. The cut-point search was performed on validation with seed 42 and the resulting cut points reused for seeds 43 and 44. The M8 cuts were 17, 33 and 53 µg/m³. This procedure tests the feature-table revision as a whole. The project retains v1 for the M0–M7 comparison and evaluates M8 against M5 only. [3, 5, 8]

| Metric                  | M5 v1 validation | M8 v2 validation | M5 v1 test | M8 v2 test |
|-----------------------------|----------------------|----------------------|----------------|----------------|
| Macro-F1                    | 0.7195               | 0.7246               | 0.7078         | 0.7109         |
| Balanced accuracy           | 0.7234               | 0.7342               | 0.7117         | 0.7177         |
| High-risk recall            | 0.8600               | 0.8607               | 0.8788         | 0.8770         |
| Mean absolute ordinal error | 0.2644               | 0.2612               | 0.2873         | 0.2852         |
| Severe jump rate            | 0.0061               | 0.0073               | 0.0091         | 0.0106         |
| Mean asymmetric cost        | 0.5871               | 0.5554               | 0.7083         | 0.6711         |

Table 4.4. M5 and M8 validation means over three seeds and registered single-pass test results for seed 42. Severe jumps are errors of at least two levels. [3, 5, 8]

On test, v2 reduced average cost by 0.0372 and raised Macro-F1 by 0.0031. High-risk recall fell by 0.0018 and the severe jump rate rose by 0.0015. [5, 8]

The registered M5 run used Python 3.12.14 and scikit-learn 1.9.0. The registered M8 run used Python 3.10.11, scikit-learn 1.7.2, NumPy 1.24.3 and pandas 2.3.3. On validation, rerunning M5 under Python 3.14.6 with scikit-learn 1.9.0 reproduced the registered figures. In that same environment, switching from M5 v1 to M8 v2 increased validation Macro-F1 by 0.0046 and reduced cost by 0.0311. Across the two recorded M8 environments, the validation differences were 0.0005 in Macro-F1 and 0.0007 in cost. The F1 difference was below M8’s own seed standard deviation in each environment. The environment checks used validation only, so the held-out comparison retains the original test results and their recorded environments. [3, 8]

## 4.8 Limitations

All outcomes are CAMS grid-cell values. Evidence about ground-level forecast skill or the physical source of an episode would require independent data. The warning record is conditional on the six-hour quiet period and the 35 µg/m³ onset threshold. The clock-hour and severity splits show how results vary within that definition. [2, 6]

M7’s gain in early alerts comes with a lower confirmed-alert share. M5 underestimates high concentrations on test, and its interval has uneven coverage across receptors. The archived replay has fixed inputs and supports retrospective evaluation. An operational service would require an online feature pipeline and prospective evaluation against independent observations. [1, 2, 7]

## 4.9 Summary

The HazeGuard-KL replay delivers three-hour risk estimates, concentration forecasts and uncertainty information from traceable model artifacts. M5 warned before 79.2% of the 419 test onsets, while M7 raised that share to 87.6% at the expense of alert confirmation. The run-boundary pattern, underprediction of concentration peaks and 88.0% test coverage for the nominal 90% interval qualify these results. For the team’s conclusions, the evidence supports historical replay of the warning policies on CAMS and identifies independent ground validation as a prerequisite for operational claims. [1, 2, 7]

## Section 4 Sources

[1] [Streamlit replay implementation](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/app.py) and [project specification](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/docs/spec_v3.md).

[2] [Stage 7 analysis code](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/src/07_system_analysis.py#L113-L373) and [committed validation and test results](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/models/metrics_stage7.json).

[3] [Ordinal model and interval handoff](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/docs/handover/05_ordinal_and_interval.md), [M5/M6 metrics](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/models/metrics_stage5.json) and [M5 selection record](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/models/stage5_selection.json).

[4] [Section 3 cost and explainability](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/reports/sec3_cost_and_explain.md), [M7 validation thresholds](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/models/cost_thresholds.json) and [held-out policy comparison](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/models/m7_test_metrics.json).

[5] [Reconciled test benchmark](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/models/all_models_benchmark.csv) and [Stage 7 tests](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/tests/test_stage7.py).

[6] [Team report on the CAMS run boundary](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/reports/sec1_data_and_baseline.md#L185-L229) and [run-boundary statistics](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/models/seam_report.json).

[7] [Adaptive interval method and coverage](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/models/metrics_stage5_interval_adaptive.json) and [raw and fixed-width interval results](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/models/metrics_stage5_interval_calibrated.json).

[8] [KD-01 defect record](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/docs/known_defects.json), [feature-table manifest](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/data/feature_manifest.json), [M8 script](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/src/07_m8_v2_control.py), [registered M8 results](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/models/metrics_m8.json), [v1 feature list](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/models/selected_features.json#L9-L35), [v2 feature list](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/models/selected_features_v2.json#L9-L35) and [runtime replication](https://github.com/baiyizhuoait/HazeGuard-KL/blob/fd09d2f95af6b7f09fc93b6d2833707e6db95611/models/runtime_environments.json).
