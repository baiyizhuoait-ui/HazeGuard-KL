# Agent 2 exploratory analysis and spatiotemporal features

*Technical draft for verification and integration by the student authors.*

## 2.3 EDA & Key Spatiotemporal Insights

### Data scope and departures from specification v2.0

This analysis uses the existing `data/haze_panel_clean.parquet` at upstream revision `2cd57c5`. The input contains 966,677 rows and 35 columns, spanning 4 August 2022 to 31 October 2026 in Kuala Lumpur local time. Its SHA256 was verified against Agent 1's manifest. Agent 2 neither downloaded another dataset nor changed any upstream file. The four upstream Gate 0 checks pass under the repository's revised receptor-based rules. This does not constitute a new validation against ground observations.

The supplied *HazeGuard-KL 技术规范 v2.0*, particularly Sections 3.1, 3.2 and the Agent 2 deliverables, defines the feature families and figure filenames. The repository has since changed its spatial and temporal scope. The following adaptations preserve the actual upstream data contract. They do not satisfy the original requirement for eight independently available location series.

| Item | Specification v2.0 | Implemented scope |
|---|---|---|
| Spatial units | Eight named KL locations | Twenty-six distinct CAMS grid-cell series already in the repository |
| Main prediction population | Eight locations | Three receptor cells: KL_City, Port_Klang and Sepang; other cells supply spatial context |
| Period | November 2024 to October 2026 | Existing panel from August 2022 to October 2026 |
| Input path | `data/kl_8stations_clean.parquet` | `data/haze_panel_clean.parquet`; the legacy input is supported if supplied |
| Diurnal and wind summaries | Eight locations | The three retained receptor cells |
| Cross-correlation pairs | Port_Klang/Banting to Cheras_KL/XMUM_Sepang | Port_Klang to KL_City and Sepang; no repeated alias series |
| Spatial kernel | Gaussian bandwidth of 15 km | Retained as specified, with availability and weight diagnostics |

The node registry groups Banting with Sepang. It contains neither Cheras_KL nor XMUM_Sepang, and does not establish their exact equivalence to the available receptor cells. Port_Klang to KL_City and Sepang is therefore an available coastal/inland comparison, not a verified alias mapping. Separate curves for missing locations are not fabricated. The 26-cell distribution is shown for context, while receptor rows define the primary analysis population.

An additional check aligns clean training PM2.5 values by exact timestamp for every pair of the 26 retained cells. None of the 325 pairs is exactly identical on its overlapping training hours. The largest absolute Pearson correlation is 0.9086 for Alor_Setar and Penang, using 26,034 paired hours; no pair reaches the disclosed diagnostic threshold of 0.95. These results do not establish statistical independence or independent model errors. The historical discovery file documents duplicate candidate locations before deduplication, whereas this new check applies to the delivered panel.

Air quality values are CAMS model estimates obtained through Open-Meteo, not measurements at monitoring stations. Open-Meteo documents the global product at approximately 45 km spatial resolution and three-hour native temporal resolution, although the supplied API table is hourly. An hourly table alone cannot establish independent hourly information. The weather acquisition code does not select a specific `models` value, and the repository does not retain response-level model metadata. Therefore, the weather archive cannot be identified conclusively as pure ERA5 or assigned a single verified resolution. Open-Meteo's default archive combines products. Both the archive and derived features are retrospective; operational availability at issue time has not been demonstrated. [Open-Meteo air quality documentation](https://open-meteo.com/en/docs/air-quality-api), [historical weather documentation](https://open-meteo.com/en/docs/historical-weather-api).

Figures 1, 2 and 4 use training receptor rows after excluding rows marked as interpolated or still missing. Figure 2 also requires an exact, quality-checked target at t+3h within the same split. Figures 3 and 5 describe all splits, as permitted for descriptive analysis; they were not used to choose features, bandwidths or thresholds. No prediction model was fitted in this stage. All numerical results below are recorded in `models/metrics_stage2_eda.json` and `models/metrics_stage2_features.json`.

### Figure 1 Diurnal and monsoon-period distributions

![Figure 1](../figures/fig_eda_diurnal_seasonal.png)

**Observation.** KL_City and Sepang have local maxima at 09:00 and 21:00, whereas Port_Klang has local maxima at 01:00 and 15:00. The southwest-monsoon receptor median is 26.6 µg/m³, compared with 26.0 µg/m³ during the northeast-monsoon months. Their interquartile ranges are 18.3–39.7 and 17.0–38.5 µg/m³, respectively. The seasonal contrast is small in this training sample. The plot does not impose an early-morning and evening pattern on every cell.

**Physical interpretation.** Repeated emissions, boundary-layer evolution and the atmospheric model's temporal structure could contribute to the diurnal pattern. The different Port_Klang phase argues against treating one receptor curve as a universal local cycle. Seasonal boxplots pool cells, years and hours, so their difference does not isolate a monsoon effect or establish a biomass-burning contribution.

**Implication for modelling.** Hour and month are encoded cyclically, and short historical changes remain separate from calendar features. The shaded intervals are pointwise percentile intervals from a circular moving-block bootstrap of dates: seven-day blocks, 400 replicates and seed 42. Each resampled date retains its hourly profile. These bands account for short-range dependence within the chosen block length, but do not quantify model bias, spatial representativeness or simultaneous coverage over the whole curve.

### Figure 2 Associations with the three-hour target

![Figure 2](../figures/fig_eda_correlation_matrix.png)

**Observation.** Current PM2.5 is associated with the future target in both Pearson correlation (0.737) and Spearman correlation (0.779). PM10 has corresponding coefficients of 0.721 and 0.760. Wind speed has weaker pooled associations, at −0.107 and −0.091. Only cells with an absolute coefficient above 0.3 are annotated; this is a display rule rather than a selection threshold.

**Physical interpretation.** Concentration persistence and shared model structure can produce strong pollutant associations. The PM2.5–PM10 association also reflects related particle quantities. Weak marginal wind correlations do not exclude conditional relationships with direction, humidity or cell geometry. The matrix pools receptor cells and contains repeated time-series samples; its coefficients are descriptive associations without independent-sample significance claims.

**Implication for modelling.** Lagged pollutant values provide a persistence-oriented reference against which spatial features should be tested. Wind direction is represented by vector components rather than correlated as a linear degree variable. Agent 3 should assess incremental predictive value within time-ordered training and validation folds, rather than retain only features with large marginal correlations.

### Figure 3 Ordered risk-level distribution

![Figure 3](../figures/fig_eda_class_imbalance.png)

**Observation.** Across receptor rows, the four future-level counts are 19,986, 55,480, 22,484 and 13,617. Level 1 accounts for approximately half of the receptor samples. Level 3 represents 12.2% overall and is not globally scarce in this panel, although its local support differs across cells. Port_Klang has Level 2 and Level 3 proportions of 9.73% and 1.94%, respectively. This does not support the specification's suggested expectation that Port_Klang should necessarily contain more high-level samples.

**Physical interpretation.** The differences describe model grid-cell estimates and cannot establish differences in measured personal exposure. Including low-concentration source and corridor cells in a pooled training population would change the class prior relevant to KL receptors. The four categories are project experimental risk levels with concentration thresholds of 15, 35 and 55 µg/m³. They are not official hourly AQI health standards.

**Implication for modelling.** Main model comparisons should continue to train, validate and evaluate on receptor rows. The all-node stacked panel provides context without expanding the main training population. Downstream reports should disclose class counts after the eligibility filter, particularly when comparing high-risk recall. Figure 3 describes the original target table; it is not a count of all rows accepted for model fitting.

### Figure 4 Wind direction, speed and concentration

![Figure 4](../figures/fig_eda_wind_rose_pollution.png)

**Observation.** The training wind summary contains 73,189 non-calm receptor rows. The sector centred on 67.5° has the highest sector mean, 38.11 µg/m³, whereas the 225° southwest sector has a mean of 29.36 µg/m³. The highest sector mean therefore occurs under east-northeasterly flow, not southwesterly flow. The first polar panel separates wind-speed frequencies; the second colors directional frequency bars by mean PM2.5.

**Physical interpretation.** A directional association may reflect season, local emissions, receptor composition or circulation in the parent model. It cannot identify a geographic emission source. Calm hours are excluded because their directions are unreliable. Here, direction denotes where the wind comes from, with north at zero and angles increasing clockwise.

**Implication for modelling.** The wind-vector and geometric aggregation features are implemented as hypotheses to test. Their value should be determined by validation ablations and quality-sensitive analyses, without assuming that southwest flow must increase pollution. Regional fire attribution would require additional evidence beyond this single-source project.

### Figure 5 Coastal and inland lead–lag association

![Figure 5](../figures/fig_eda_spatial_crosscorr.png)

**Observation.** Positive lag k is defined as corr[coast(t−k), inland(t)], so a positive peak would indicate coastal lead. Both raw curves peak at −1h, with coefficients of 0.614 for Port_Klang–KL_City and 0.614 for Port_Klang–Sepang. After subtracting month/hour climatologies fitted only on training data, both peaks remain at −1h, with coefficients of 0.612 and 0.599. Thus the descriptive curves do not support the proposed positive coastal lead.

**Physical interpretation.** Removing a training climatology reduces shared calendar structure but does not remove all confounding or autocorrelation. Adjacent cells may share atmospheric forcing and model dynamics. A broad correlation maximum, including one at a negative lag, is insufficient evidence for transport direction, causal transmission or a stable travel time.

**Implication for modelling.** The specified one- and two-hour spatial lags are retained without choosing them from these all-split curves. No transport-speed estimate is inferred. Cross-correlations use exact timestamp joins over −12h to +12h; gaps remain missing rather than compressing the time axis. Stability across years and independently measured pollution episodes remains to be investigated.

## 2.4.1 Temporal Lag, Rolling & Monsoon Upwind Advection Features

### Temporal information available at issue time

The output `data/feat_spatiotemporal.parquet` preserves all input rows, their order, original column names, values and dtypes, and adds 56 columns. Its resulting shape is 966,677 by 91. New continuous columns use float32 precision; original columns retain their existing precision. No imputation, scaling, feature selection or learned model is fitted by Agent 2.

PM2.5, PM10, NO2 and relative humidity each receive lags at 1, 2, 3, 6, 12 and 24 hours. PM2.5 also receives rolling mean, sample standard deviation (ddof=1) and range over 3, 6 and 12 hours. The rolling family applies to PM2.5; the other three variables receive the specified lag family. The three-hour change is x(t)−x(t−3), and the six-hour trend is an ordinary least-squares slope on equally spaced hours, expressed in µg/m³ per hour.

Rolling windows end at t and include the current value, under the explicit assumption that the value at issue time has become available. A window requires all its hourly samples. Calculations operate separately within each node and split, on an internal complete hourly index. Missing hours are never filled, and the original table is not expanded. Consequently, a missing hour produces an unavailable exact lag or incomplete rolling window, rather than a misleading previous-row value. Validation and test histories restart at their split boundaries, following v2.0's strict boundary rule.

Hour, day of week and month receive sine/cosine encodings with periods of 24, 7 and 12. Month is zero-based before encoding. Time calculations use Asia/Kuala_Lumpur; timezone-aware inputs are converted to that zone, while timezone-naive input is interpreted according to the upstream local-time contract.

### Wind vectors and directional spatial aggregation

Wind speed is read from `wind_speed_ms`, cross-checked against `wind_speed_kmh / 3.6` when both exist. The km/h value is converted only when a named m/s column is absent. A legacy `wind_speed_10m` field requires an explicit unit argument, avoiding a silent double conversion. Meteorological wind-from direction θ produces eastward and northward components:

u = −s sin(θπ/180), v = −s cos(θπ/180).

Vectors are set to zero for calm speeds below 0.5 m/s or an existing calm-wind flag. The implementation computes great-circle distance D in kilometres and an east/north unit vector e at source j pointing toward destination i. Self-links and co-located points are excluded. The specified weight is:

w(j→i,t) = max[0, vector(j,t) · e(j→i)] exp[−D(j,i)²/(2σ²)], σ = 15 km.

Projection values at floating-point zero and weights at or below 10⁻¹² m/s are discarded before normalization. This numerical floor prevents a practically disconnected remote node from becoming the sole apparent neighbor. For k equal to one or two hours, both the source concentration and source wind are taken at exactly t−k, and the source must belong to the destination row's split. The weighted mean excludes unavailable source values from both numerator and denominator. A zero denominator produces NaN, not zero concentration or a fallback drawn from another period.

`spatial_pm25_gradient` is current destination PM2.5 minus the contemporaneous directional neighbor mean. This choice makes its time reference explicit; it is not current concentration minus a lagged neighbor value. Weight sums and contributing-neighbor counts accompany both lagged spatial means. They distinguish lack of spatial support from an actual low concentration.

The 15 km bandwidth is narrow relative to the expanded regional network. One-hour advection is available for 86.15% of KL_City rows, 72.87% of Port_Klang rows and 95.41% of Sepang rows. These are numerical support rates, not evidence of useful inter-island transport. Distant Sumatra and Borneo cells receive negligible weight at this scale, and the aggregation has no travel-time model. Any broader bandwidth or longer transport lag requires a separately reported validation experiment. No bandwidth search was conducted here.

### Availability flags and downstream use

Upstream linear interpolation uses future endpoints and was performed before splitting. Agent 2 conservatively treats rows flagged `qc_interpolated` or `is_missing` as unavailable for new concentration histories and wind/spatial contributions. Original interpolated values and upstream fitted baselines remain unchanged. The eligibility flag alone cannot make every retained upstream column safe: downstream code should use the explicit `model_feature_columns` list in the feature audit rather than all numeric columns. That list excludes labels, diagnostic eligibility flags, raw wind direction and upstream baseline/anomaly columns.

An additional exact t+3h audit checks target alignment, class thresholds, source quality and split boundaries. It adds flags rather than changing or deleting labels. Eligibility requires clean current input, complete requested temporal histories and a verified future target within the same split. The accepted receptor counts are 77,753 training, 16,656 validation and 16,647 test rows. This conservative policy also rejects endpoint labels whose future row cannot be verified from the supplied cleaned table. Target-audit flags depend on future label availability and must never be used as predictor features or as conditions for issuing a live forecast.

Missing spatial features are expected. A downstream estimator must handle them explicitly or fit an imputer only on its training fold. Within-fold feature transforms, climatologies and selection must be rebuilt for each time-series validation or held-out-node experiment. A label horizon also requires purging training examples whose target time reaches a later validation fold. Agent 2's top-level split flags do not implement future CV folds or establish an absence of all upstream leakage.

### Verification and limitations

The implementation is checked against synthetic examples with known slopes, distances, wind signs, gaps and time delays. Tests perturb future and held-out values to verify issue-time feature invariance, check exact unit conversion, isolate node/split boundaries, and distinguish an absent neighbor from a clean concentration. The original Agent 1 checker is also run without rewriting its manifest. Verification results and environment versions are recorded separately in `models/metrics_stage2_verification.json`.

These checks establish the documented transformations on the available panel. They do not validate the parent atmospheric models, the accuracy of the original retrieval, the independent nature of hourly values or operational forecast availability. The upstream flags for PM2.5 greater than PM10 and censored concentrations are preserved for later sensitivity analysis. Claims in earlier repository prose that particular high-value periods are confirmed real haze events have not been independently verified here. No model improvement, causal transport effect, calibrated health benefit or measured exposure reduction is claimed.

### Sources and reproducible evidence

- Uploaded specification: *HazeGuard-KL_技术规范_v2.0.docx*, SHA256 `e15c706d08d86ce6939938c55666d16b8d4c15aea7ebf479b12504921ee7bd04`.
- Repository evidence: `src/config.py`, `src/01_data_pipeline.py`, `docs/node_discovery.json`, `data/manifest.json`, `models/gate0_report.json` and `docs/handover/01_data_and_baseline.md` at the stated upstream revision.
- [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api), consulted 8 November 2026. Underlying CAMS data and Open-Meteo require attribution under their stated licences.
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), consulted 8 November 2026. This documents default model selection and the distinction between reanalysis and station observations.
