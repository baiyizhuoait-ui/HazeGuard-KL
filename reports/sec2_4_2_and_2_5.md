# Window granulation and feature selection

## 2.4.2 Window Granulation

At issue time t the model sees the hourly PM2.5 values of the last few hours. A single lag reports one past value. It does not say how high the recent level was, how much it varied, or whether it was rising, and a haze pulse is mainly a change of this kind. We therefore summarise each trailing window with an interval granule (lower bound, core, upper bound) and a trend granule (least-squares slope), using windows of 6 and 12 hours. An earlier plan with fuzzy C-means prototypes was dropped because it falls outside the course scope and overlaps with rolling statistics.

For a grid cell and issue time t, with hourly PM2.5 x in µg/m³, the trailing window is

$$\mathbf{x}_W(t) = \left(x_{t-W+1}, \dots, x_t\right), \qquad W \in \{6, 12\},$$

and the five granule values are

$$\text{low} = \min_k x_k, \quad \text{core} = \frac{1}{W}\sum_{k=1}^{W} x_k, \quad \text{up} = \max_k x_k, \quad \text{width} = \text{up} - \text{low},$$

$$\text{slope} = \frac{\sum_{k=1}^{W} (k-\bar{k})\, x_k}{\sum_{k=1}^{W} (k-\bar{k})^2}, \qquad \bar{k} = \frac{W+1}{2}.$$

This gives ten columns named `gran_{low,core,up,width,slope}_{6,12}h`. The slope is in µg/m³ per hour and the others in µg/m³.

Each window uses hour t and earlier hours only. Windows are built per grid cell and split on a complete hourly grid, and hours flagged as interpolated or missing are left empty. A window with any empty hour gives missing values. This happens in 0.155% of panel rows for 6 hours and 0.27% for 12 hours, and in no eligible receptor row. The statistics have no fitted parameters, so nothing is estimated on the training split.

Five of the ten columns are identical, value by value, to stage 2 columns: the two cores equal `pm2_5_mean_6h` and `pm2_5_mean_12h`, the two widths equal `pm2_5_range_6h` and `pm2_5_range_12h`, and the 6-hour slope equals `pm2_5_slope_6h`. This confirms that both modules build windows the same way. Only the other five columns (`gran_low_6h`, `gran_up_6h`, `gran_low_12h`, `gran_up_12h`, `gran_slope_12h`) enter feature selection.

### Figure 6 Interval and trend granules during a haze pulse

![Figure 6](../figures/fig_granulation_bands.png)

The figure shows the KL_City grid cell for five days on either side of 07:00 on 7 March 2025, the hour with the highest 12-hour core among eligible receptor training rows. When the pulse enters the window, the 12-hour band widens at once, since the upper bound jumps while the lower bound still holds the earlier clean hours. It stays wide for about half a day after the peak, until the peak hour leaves the window. The 6-hour band and slope react faster: the 6-hour slope peaks during the build-up and turns negative soon after the peak.

## 2.5 Feature Selection and Ablation

### Candidates and fitting rows

The candidates are the 63 columns of the stage 2 whitelist `model_feature_columns` (12 original variables and 51 stage 2 predictors) plus the five new granulation columns, 68 in total. Climatological anomalies, audit flags, weight diagnostics, identifiers and targets are excluded. Selection uses only the 77,753 eligible receptor rows of the training split, sorted by time.

The near-field upwind features are missing when no valid upwind neighbour lies within 150 km. Among eligible receptor rows this affects 9.76% of training, 9.74% of validation and 4.52% of test rows. We filled these cells with fixed rules before selection (Table 1). The rules have no parameters and do not treat a missing value as zero concentration.

**Table 1.** Filling rules for missing spatial values.

| Column | Filled with | Rows filled (train / val / test) |
|---|---|---|
| `upwind_pm25_advection_{1,2,3}h` | the cell's own `pm2_5_lag_{1,2,3}` | 7,584 to 7,586 / 1,623 / 753 |
| `spatial_pm25_gradient` | 0 | 7,587 / 1,623 / 753 |
| `far_source_pm25_mean_48h` | `far_source_pm25_mean_24h` at the same hour | 70 / 72 / 72 |

### Selection procedure

Three filters were applied to the training rows:

1. mutual information with the four-level target, keeping the top 30;
2. a quasi-constant filter (no column removed), then greedy Spearman pruning at |ρ| > 0.92, which kept 47 columns;
3. permutation importance on an internal holdout (the last 20% of training time, from 19 December 2024, 15,606 rows), with a default random forest fitted on the 62,138 earlier rows (leaving a three-hour gap) and scored by macro-F1, keeping the top 30.

Their union of 64 columns was reduced to 25 by recursive feature elimination with a default random forest. The union removed only four candidates, so the elimination step did most of the work. The Top-25 (listed in `models/selected_features.json`) contains seven original variables, twelve stage 2 temporal features, two spatial features and four granulation features. None of the advection columns `upwind_pm25_advection_1h`, `_2h` or `_3h` was selected.

### Ablation

We trained a default random forest on five nested feature sets with seeds 42, 43 and 44 and evaluated on the validation split (16,656 eligible receptor rows, 4,842 high risk). Metrics come from `src/utils_metrics.py`. High-risk recall is the share of true Level 2 or 3 hours predicted as Level 2 or 3. M0 predicts the current level.

**Table 2.** Feature ablation on the validation split (mean ± standard deviation over three seeds).

| Feature set | Columns | Macro-F1 | High-risk recall | MAOE | Avg. cost |
|---|---:|---|---|---|---|
| M0 persistence | n/a | 0.6095 | 0.7051 | 0.3811 | 0.9266 |
| 1. Original variables | 12 | 0.6666 ± 0.0012 | 0.6597 ± 0.0029 | 0.3053 | 0.8547 |
| 2. + Lag and rolling | 53 | 0.7125 ± 0.0013 | 0.7642 ± 0.0022 | 0.2566 | 0.6419 |
| 3. + Spatial | 63 | 0.7103 ± 0.0006 | 0.7555 ± 0.0010 | 0.2579 | 0.6545 |
| 4. + Granulation | 68 | 0.7129 ± 0.0018 | 0.7623 ± 0.0006 | 0.2561 | 0.6419 |
| 5. Selected Top-25 | 25 | 0.7107 ± 0.0013 | 0.7664 ± 0.0019 | 0.2597 | 0.6497 |

### Figure 7 Feature ablation on the validation split

![Figure 7](../figures/fig_feature_ablation.png)

Dots are means over three seeds, bars one standard deviation, and the dashed line is M0.

### Findings

**Lags and rolling statistics give almost all of the gain.** They raise macro-F1 by 0.046 (0.6666 to 0.7125) and high-risk recall by 0.10 (0.6597 to 0.7642). With the original variables alone, high-risk recall stays below M0 (0.6597 against 0.7051).

**Spatial features did not help.** Adding them lowers macro-F1 from 0.7125 to 0.7103 and high-risk recall from 0.7642 to 0.7555, drops larger than the seed standard deviations. No advection column was selected. Two facts explain this. The forecast horizon is three hours, and the Sumatran and Bornean source cells lie outside the 150 km near-field radius, so they only describe the regional background. And in about one row in ten the filled advection columns simply repeat the cell's own lags. The value of spatial information is better judged in the cross-cell generalisation study.

**Granulation adds little.** Set 4 is only 0.0004 above set 2 in macro-F1 (0.7129 against 0.7125) with the same average cost. Four granulation columns were still selected, but most of their information is already in the rolling features.

**Selection keeps performance with fewer columns.** The Top-25 reaches the highest high-risk recall (0.7664) with 25 of 68 columns, while its macro-F1 and cost are slightly worse than set 4.

**Limitations.** The standard deviations only reflect the forest seed, so differences below about 0.01 should not be read as firm rankings. Because the Top-25 was chosen on the whole training split, later cross-validation inside that split will be slightly optimistic. No test data were used here.

**Sources.** `models/selected_features.json` (selection and filling), `models/metrics_stage3.json` (ablation, per-seed values and M0), `models/metrics_stage2_features.json` (missing rates, 150 km radius); code in `src/03_granulation_features.py` and `src/03_feature_selection.py`.
