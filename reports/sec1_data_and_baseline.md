# Section 1 — Introduction

## 1.1 Background

The Klang Valley has two kinds of pollution at the same time. One is the local daily
cycle from traffic and industry. It repeats in much the same way every day, so it is
fairly easy to anticipate. The other is transboundary biomass-burning haze, carried from
Sumatra and Kalimantan by the south-west monsoon between June and October. It can raise
ground-level PM2.5 from about 20 µg/m³ to over 100 µg/m³ within a few hours.

The two need different responses. Most people deal with the first by habit. The second is
what closes schools, cancels outdoor events and sends asthma patients to hospital, and it
is also the harder one to anticipate from the number most Malaysians look at.

## 1.2 Problem Statement

The Department of Environment (DOE) publishes the Air Pollutant Index (API), which is a
**24-hour moving average**. For regulatory reporting and long-term health studies this is
a reasonable choice: it smooths out short spikes and gives a stable number that can be
compared over time. For decisions within the next few hours it does not work well, for
two reasons.

**A 24-hour average lags behind the air.** If a haze front arrives at 14:00 and lifts PM2.5 from
25 to 95 µg/m³, it adds only about 70 µg/m³ to the 24-hour mean, because most of the
denominator is clean air. The index therefore crosses the "unhealthy" threshold several
hours after the air has already become unhealthy.

**The index reports a level, not a trend.** A current reading only answers "how bad is it
now". At 07:00 the question is usually a different one: what will it be like at 10:00.
That is when I have to decide whether to cycle to campus. Answering it needs a
three-hour-ahead estimate, and it needs some statement of how reliable the estimate is.

There is a third issue. The four bands (good, moderate, unhealthy, very unhealthy) are
ordered, and the two directions of error cost very different amounts. Telling someone the
air is moderate when it is in fact very unhealthy is much worse than the opposite mistake.
A model that treats the four levels as four unrelated categories throws away both the
ordering and the cost difference.

## 1.3 Objectives

This work develops and evaluates a three-hour-ahead PM2.5 risk nowcasting system for the
Klang Valley. Specifically, it aims to:

1. **O1**: build a leakage-free, chronologically split hourly panel of atmospheric
   composition and meteorology for the Klang Valley and its upwind source regions, covering
   fifty-one months.
2. **O2**: subtract the climatological background so that the model has to learn *deviation
   from normal* instead of *which month it is*.
3. **O3**: compare seven model families under one evaluation contract. These are a
   decision-tree floor, nominal random forests with and without class reweighting, a
   boosting baseline, a regression-then-threshold formulation, an ordinal decomposition,
   and a cost-sensitive ordinal model whose thresholds are tuned by grid search.
4. **O4**: measure what happens when the model meets an unseen node (leave-one-node-out)
   and an unseen year (temporal extrapolation), instead of reporting one in-distribution
   score.
5. **O5**: package the pipeline so that another group can rerun it, including the negative
   results.

## 1.4 Scope and Honest Framing

Two limitations belong here rather than in the conclusion, because both change how the
numbers should be read.

**The pollutant fields are model output, not monitor readings.** The air-quality variables
come from the Copernicus Atmosphere Monitoring Service (CAMS) global composition product,
distributed by Open-Meteo, at a native resolution of 0.4 degrees (about 45 km). They are
grid-cell estimates from a chemistry and transport model, not readings from a DOE station.
That is adequate for studying *regional transport*, which happens over hundreds of
kilometres, and not adequate for claims about exposure at street level.

**The spatial unit is a grid cell, not a monitoring station.** Because 45 km exceeds the
extent of the Klang Valley itself, the eight administrative sites originally proposed for
this project collapse onto three distinct grid cells, with numerically identical series
within each. Rather than report eight "stations" that are in truth three, this work
re-identifies the spatial unit explicitly, extends the domain to include the upwind
source regions, and verifies cell distinctness by direct comparison of the retrieved
series (Section 2.1.2). The resulting panel contains twenty-six genuinely distinct cells.

---

# Section 2.1 — Data Acquisition and Quality Control

## 2.1.1 Sources

Two Open-Meteo endpoints were used, both free of authentication and therefore
reproducible by a marker without credentials:

| Endpoint | Variables | Native product | Resolution |
|---|---|---|---|
| `air-quality-api.open-meteo.com/v1/air-quality` | PM2.5, PM10, NO₂, SO₂, O₃, CO | CAMS global composition | 0.4° (~45 km) |
| `archive-api.open-meteo.com/v1/archive` | 2 m temperature, relative humidity, 10 m wind speed / direction, precipitation | ERA5 reanalysis | ~9 km |

The retrieval window is **1 August 2022 to 31 October 2026**, hourly, in
`Asia/Kuala_Lumpur` (UTC+8). The start date is not arbitrary: the CAMS global archive
begins on 1 August 2022, and the first ~80 hours of that month return null values across
all pollutants. Those leading rows are trimmed rather than interpolated (Section 2.1.4),
because a data-availability edge is not a measurement gap.

Total retrieved volume: **969,072 station-hours** across 26 nodes.

## 2.1.2 Spatial Unit: Grid-Cell Resolvability

This subsection documents a design decision that materially changes the project, and the
evidence for it.

Thirty-three candidate coordinates were placed across Peninsular Malaysia, Sumatra and
Kalimantan. For each, one month of hourly PM2.5 was retrieved (July 2025) and the series
were compared pairwise for exact equality. The result:

| Candidate group | Distinct cells | Example |
|---|---|---|
| 8 Klang Valley sites | **3** | `KL_City` ≡ `Batu_Muda` ≡ `Petaling_Jaya` ≡ `Shah_Alam` |
| Peninsular corridor (12 sites) | 9 | `Malacca` ≡ `Muar`; `Ipoh` ≡ `Cameron_Highland` |
| Sumatra (8 sites) | 8 | all distinct |
| Kalimantan (5 sites) | 5 | all distinct |
| **Total** | **26** of 33 | — |

Within a cell the hourly series are *bit-identical*, confirming that the API returns a
single grid value per cell rather than an interpolated field. Meteorological variables,
by contrast, differ at every one of the 8 original sites, because ERA5's ~9 km grid is
finer than the inter-site spacing.

Two consequences follow, and the design changed because of them:

1. **The upwind-transport feature requires regional, not municipal, separation.** Computing
   an advection index between two coordinates that share a grid cell is meaningless. The
   "neighbour" contribution is the target's own value. The domain is therefore extended to
   include Sumatra and Kalimantan source cells, which is physically the correct scale for
   transboundary haze anyway.
2. **Leave-one-out generalisation requires distinct cells.** Holding out one of eight
   Klang Valley "stations" would leak the answer, since three siblings in the same cell
   carry an identical label sequence. With 26 distinct cells, the experiment recovers its
   meaning.

The final panel:

| Role | Definition | Nodes | n |
|---|---|---|---|
| **Receptor** | Prediction target; deployment sites in the Klang Valley | `KL_City`, `Port_Klang`, `Sepang` | 3 |
| **Corridor** | Both target and advection source | Peninsular cities (Seremban, Malacca, Johor Bahru, Kuantan, Temerloh, Kota Bharu, Kuala Terengganu, Penang, Alor Setar, Ipoh) | 10 |
| **Source** | Advection features only | Sumatra (8) + Kalimantan (5) | 13 |

## 2.1.3 Variables and Unit Normalisation

| Variable | Unit as delivered | Unit used | Note |
|---|---|---|---|
| `pm2_5`, `pm10`, `no2`, `so2`, `o3`, `co` | µg/m³ | µg/m³ | CO is **not** ppm on this endpoint |
| `wind_speed_10m` | km/h | **m/s** (÷ 3.6) | raw km/h retained as `wind_speed_kmh` for audit |
| `temperature_2m` | °C | °C | — |
| `relative_humidity_2m` | % | % | — |
| `precipitation` | mm | mm | — |
| `wind_direction_10m` | degrees, 0 = N, clockwise | degrees + sin/cos | circular quantity; see below |

Two unit traps were found and handled explicitly. The wind-speed default of km/h would
inflate the advection weights by a factor of 3.6 while raising no error, and CO arriving
in µg/m³ rather than ppm would silently mis-scale any ratio computed against it.

Wind direction is a circular quantity: 359° and 1° are two degrees apart but numerically
358 apart. Passing raw degrees to a tree invites spurious splits at the 0/360 boundary, so
`wd_sin` and `wd_cos` are derived. This is unit handling rather than feature engineering: no
temporal lag or spatial aggregation is involved.

## 2.1.4 Quality Control Procedure

Applied per node, in order:

1. **Deduplication** on `(station_id, timestamp)`, first occurrence retained. *Rows removed: 0.*
2. **Hourly grid reconstruction** via `reindex` on a complete `pd.date_range`. Without this,
   `shift(1)` means "the previous record", not "one hour ago", and every lag silently
   becomes wrong whenever a row is missing. *Rows added: 0* (all nodes already complete).
3. **Leading-block trimming** (the CAMS archive edge). *Rows trimmed: 2,080*
   (80 hours × 26 nodes).
4. **Physical bounds** → NaN. Thresholds: PM2.5 [0, 900], PM10 [0, 1200], NO₂/SO₂/O₃
   [0, 500], CO [0, 20000], T [−20, 50] °C, RH [0, 100] %, wind [0, 60] m/s,
   precipitation [0, 300] mm. *Violations found: 0 across all variables.*
5. **Short-gap interpolation**: linear, maximum 3 consecutive hours, interior only.
   *Cells filled: 656.* Longer gaps are left as NaN and flagged, never extrapolated.
6. **Missing flag**: `is_missing = 1` if any variable remains null after step 5.
   *Rows flagged: 277* (0.03 %).
7. **Target construction and row removal**: rows without a valid `t+3h` target are dropped.
   *Rows dropped: 226.*

Net: 969,072 → **966,677 rows** (99.75 % retention).

## 2.1.5 A Discontinuity at the Model-Run Boundary

The air-quality fields are not a continuous physical measurement; they are assembled from
successive forecast runs. The CAMS global composition forecast is issued twice per day, at
00 and 12 UTC, with a five-day horizon on a 0.4° grid, and single-level fields are published
at hourly resolution. When a long hourly series is built from these runs, each new run
replaces the previous run's later forecast hours at the cycle boundary. The two values come
from different initial states and do not agree exactly, so the joined series contains a step
that belongs to the production system rather than to the atmosphere.

In local time (UTC+8) these boundaries fall at 08:00 and 20:00. The seam is visible in the
first difference of PM2.5. Across all 26 nodes, the mean absolute hour-to-hour change at
08:00 and 20:00 is 5.82 µg/m³ against 1.69 µg/m³ at all other hours, a factor of 3.45.
At the reference receptor node KL_City the effect is stronger: 15.17 against 4.72, a factor
of 3.22, and the mean signed change at 20:00 there is +17.93 µg/m³, so the series steps
upward rather than scattering in both directions.

Three checks separate this from ordinary weather or from the diurnal cycle.

1. **It survives removal of the seasonal and diurnal climatology.** Repeating the same
   statistic on `pm2_5_anomaly` at KL_City, which has the (month, hour) climatology
   subtracted, moves the ratio only from 3.22 to 3.02. A normal diurnal cycle would
   disappear here.
2. **It happens across the network at once.** At 20:00, 79 % of the 26 nodes change in the
   same direction, against 45 % at other hours. Local weather does not synchronise two
   hundred kilometres of coastline.
3. **Every pollutant jumps at the same two hours.** Relative to background hours, the
   absolute change is 3.23× for PM10, 2.35× for NO2, 2.32× for SO2, 1.88× for CO and
   1.81× for O3.

We also tested the alternative explanation that the hourly values are interpolated from a
three-hourly field. They are not. CAMS publishes single-level fields hourly, and the data
show no three-hour periodicity: the first difference is constant within a three-hour block
in only 1.2 % of blocks, and the mean absolute second difference grouped by hour modulo
three is 6.07, 5.17 and 6.04, i.e. flat. A linear interpolation from three-hourly nodes
would produce the opposite pattern.

**Consequences, stated rather than corrected.** The seam cannot be removed without an
independent reference series, which we do not have, so it is retained and declared here.
Two effects follow. First, any target whose three-hour lead crosses 20:00 inherits part of
the step, which adds error that no model can anticipate from the inputs available at issue
time; targets issued between 17:00 and 19:00 are the affected ones. Second, the anomaly
series used for baseline subtraction inherits the same step, so anomaly magnitudes near the
seam should be read with that in mind. Both are properties of the data source, not of the
pipeline, and `tools/check_run_seam.py` reproduces every number in this subsection.

## 2.1.6 Target Construction

Both targets are built with `groupby("station_id")["pm2_5"].shift(-3)`:

- `target_pm25_3h`: the PM2.5 concentration three hours ahead;
- `target_level_3h`: its ordinal band, using the thresholds (15, 35, 55) µg/m³.

The `groupby` is not incidental. A global `shift(-3)` would concatenate the tail of one
node to the head of the next, manufacturing labels that mix two locations. This is a
quiet failure: the code runs, the model trains, and only the metrics are wrong.

## 2.1.7 Gate 0: Feasibility Checks Before Modelling

Four gates are evaluated before any model is fitted. Two are assessed on **receptor nodes
only**, because pooling in the low-concentration source cells would dilute the target
distribution and misread their inter-annual variability as distribution drift.

| Gate | Criterion | Result | Verdict |
|---|---|---|---|
| G0-1 | ≥ 100,000 rows, ≥ 12,000 per node | 966,677 rows; min 36,952 | pass |
| G0-2 | Receptor Level 3 share ≥ 1 % | 12.21 % (panel: 3.37 %) | pass |
| G0-3 | Longest receptor gap ≤ 24 h | 5 h (panel max 37 h, Palangkaraya) | pass |
| G0-4 | Train/test Level 3 ratio ≤ 3× | 11.13 % vs 18.85 % (1.69×) | pass |

G0-3's receptor-only formulation is deliberate. A gap in a *source* node corrupts a
feature, which downstream code can mask; a gap in a *receptor* node corrupts the label
itself. The 37-hour gap at Palangkaraya (Central Kalimantan) is retained and disclosed
rather than patched, since interpolating 37 hours would fabricate a haze episode.

---

# Section 2.2 — Climatological Baseline Subtraction and Chronological Splitting

## 2.2.1 Why Subtract a Baseline

PM2.5 has a strong seasonal and daily pattern. A model given raw concentrations and a
month index can reach a decent accuracy by learning that "July is hazier than November",
without representing transport at all. It will also look fine in-sample and then fail on
an unusual event, which is the case the system is built for.

The baseline removes the predictable part so that the model is forced to explain the
residual:

$$b_{\text{proto}}(s, m, h) = \mathbb{E}\left[\text{PM}_{2.5} \mid \text{station}=s,\ \text{month}=m,\ \text{hour}=h\right]$$

$$a(t) = x(t) - b_{\text{proto}}(s, m(t), h(t))$$

The same operation is applied to 2 m temperature, yielding `temp_anomaly`, which captures
whether the boundary layer is unusually warm (convective, dispersive) or unusually cool
(stable, accumulating).

**The baseline is fitted on the training split only**, then broadcast to validation and
test. Computing it over the full panel would leak future information into training
features through the group means. This is the most common and least visible form of leakage in
time-series work. The implementation asserts the input frame is already filtered to
`split == 'train'`.

## 2.2.2 Chronological Splitting

Each node is split independently along its own time axis into 70 % train / 15 %
validation / 15 % test:

| Split | Approx. window | n rows | Permitted use |
|---|---|---|---|
| train | 2022-08 → 2025-06 | 676,884 | baseline fitting, model fitting, CV, hyperparameter search |
| val | 2025-06 → 2026-03 | 145,054 | model selection, threshold grid search, early stopping |
| test | 2026-03 → 2026-10 | 144,739 | **final reporting only — evaluated once** |

The split is defined on timestamps (span percentiles), not row counts, so that missing
rows cannot shift a boundary. No shuffling, no `KFold`, no random subsampling appears
anywhere in the pipeline. The only cross-validation used is `TimeSeriesSplit(n_splits=5)`
**inside the training block**, and it never sees validation or test data.

## 2.2.3 A Note on the Test Window

The test split happens to contain the severe haze episode of August–October 2026. At
`KL_City`, the Level 3 share reaches **46.9 % in September and 51.0 % in October 2026**,
with a monthly PM2.5 95th percentile of **151.8 µg/m³**.

We kept this rather than adjusting the split. It means the reported High-Risk Recall is
measured on a real event rather than on background noise, and it is the reason the
cost-sensitive models have something meaningful to improve. It also means the test set is
*harder* than the training distribution (Level 3: 18.85 % vs 11.13 %), which should be
stated whenever the test numbers are compared against published in-distribution results.

Receptor-level distributions:

| Split | Level 0 | Level 1 | Level 2 | Level 3 |
|---|---|---|---|---|
| train | 18.54 % | 50.29 % | 20.05 % | 11.13 % |
| test | 11.78 % | 47.18 % | 22.18 % | 18.85 % |

The distribution is workably balanced (no class falls below 11 %), so this project does
not face the extreme-rarity regime that motivates most cost-sensitive literature. The
asymmetry that matters here is in the **cost matrix**, not in the class frequencies.
We report the shares as they are; calling this "severe imbalance" would overstate it.

---

# Section 2.2.4 — Baseline Decision Tree

## 2.2.4.1 Configuration

The baseline uses current-hour observations only, with no lag or rolling
statistic, no spatial aggregation. Features are the six pollutants, five meteorological
variables, the two anomaly residuals, three calendar variables, and the wind sin/cos pair
which comes to eighteen columns in total.

Hyperparameters are selected by `TimeSeriesSplit(n_splits=5)` grid search scored on
macro-F1, over `max_depth ∈ {3, 5, 7, 9, 12, None}`, `min_samples_leaf ∈ {5, 20, 50, 100}`
and `criterion ∈ {gini, entropy}`, giving 48 configurations.

Selected: `criterion=gini, max_depth=9, min_samples_leaf=20`, giving 384 leaves and a
depth of 9. Cross-validated macro-F1: **0.6315**.

Overfitting check: training macro-F1 0.7017 vs validation 0.6867, a gap of **0.015**. The
model is not memorising the training set; the remaining error comes from the features,
which is what a baseline should show.

## 2.2.4.2 Results

Test split, receptor nodes, n = 16,647 (of which 6,848 high-risk). This is the same eligible
row set used by M2–M7, so the table can be read against the model comparison that follows.

| Metric | M0 Persistence | M1 Decision Tree | Δ (M1 − M0) |
|---|---|---|---|
| Macro-F1 | 0.6231 | **0.6658** | +0.0427 |
| Balanced Accuracy | 0.6232 | **0.6473** | +0.0241 |
| High-Risk Recall (L2 ∪ L3) | **0.7757** | 0.7675 | **−0.0082** |
| MAOE | 0.3925 | **0.3199** | −0.0726 |
| Severe Jumping Error Rate | 0.0374 | **0.0210** | −0.0164 |
| Average Asymmetric Cost | 0.9978 | **0.9047** | −0.0931 |

**Definition of High-Risk Recall.** We report the *union* recall,
Pr(ŷ ∈ {2,3} | y ∈ {2,3}), which asks whether a dangerous hour received *any* high-risk
warning. A true Level 2 predicted as Level 3 counts as a hit: it over-warns, it does not
under-warn. The stricter per-class version, Pr(ŷ = y | y ∈ {2,3}), is 0.5707 for M1 and is
reported as a diagnostic (`diag_High_Risk_Exact_Recall`) rather than as a headline number,
because penalising over-warning contradicts the asymmetric cost structure this project is
built on. The two definitions differ by 0.20 absolute, so every table states which one is used.

**M0 is not a straw man.** A persistence forecast sets the t+3h level equal to the current
level and involves no learning whatsoever. It is included because the natural competitor for
any added complexity is not an empty model but the persistent behaviour of the pollutant
itself.

Per receptor node:

| Node | Macro-F1 | High-Risk Recall (union) | Avg. Asymmetric Cost |
|---|---|---|---|
| KL_City | 0.6007 | **0.8078** | 1.0791 |
| Port_Klang | 0.6741 | 0.6724 | 0.6264 |
| Sepang | 0.6152 | 0.7615 | 0.9659 |

## 2.2.4.3 Interpretation

Three findings from the baseline set the agenda for the rest of the project.

**The tree is essentially a persistence-plus-diurnal model.** `pm2_5` alone accounts for
roughly two-thirds of Gini importance, with `hour` second. Every meteorological variable
other than `no2`, `so2` and `o3` contributes almost nothing. In other words, the baseline
has learned that "tomorrow's air resembles today's air, adjusted for the time of day".
That works because PM2.5 is autocorrelated. So any added complexity has to be compared
against persistence, not against an empty model.

**Aggregate metrics improve, but the metric that matters does not.** Macro-F1 rises by
0.04, MAOE falls by 18 %, and average asymmetric cost falls by 11 %; on every summary
statistic the tree looks like an improvement. Yet its union recall (0.7675) is *lower* than
that of pure persistence (0.7757). Training a model and thereby losing ground on "should we
raise an alarm" is the single most important result of this section: it shows that
aggregate accuracy is the wrong objective for a warning system, and it is the empirical
justification for the ordinal decomposition (M6) and the cost-sensitive threshold search
(M7) that follow.

The confusion matrix explains the mechanism. 1,358 true Level 2 cases are predicted as
Level 1 and 809 true Level 3 cases as Level 2, while false alarms in the opposite direction
are rare (5 Level 2 predicted from Level 0). The tree suppresses the boundary to accommodate
Level 1, which holds roughly half the test set. That is hedging toward the majority, not
random error.

**Cross-cell performance is uneven, and the pattern is physical.** `Port_Klang` scores
highest on macro-F1 but lowest on high-risk recall; it is the cleanest receptor (Level 3
share 1.9 %), so its minority class is least represented in training. `KL_City` shows the
opposite: lowest macro-F1, highest high-risk recall, because it carries the widest
dynamic range. A single pooled number would have hidden both behaviours.

These three observations motivate the ordinal decomposition (which attacks the downward
hedging directly by modelling cumulative probabilities), the class reweighting variants,
and the cost-sensitive threshold search. They are not added as extras.
