# Section 1 — Introduction

## 1.1 Background

Malaysia's Klang Valley experiences two superimposed pollution regimes. The first is a
local diurnal background driven by traffic and industry, which repeats with remarkable
regularity and is therefore relatively easy to anticipate. The second is transboundary
biomass-burning haze, transported by the south-west monsoon from Sumatra and Kalimantan
between June and October, which can raise ground-level PM2.5 from roughly 20 µg/m³ to
above 100 µg/m³ within a few hours.

The two regimes demand different responses. A commuter can adapt to the first by habit.
The second is the one that closes schools, cancels outdoor events, and drives asthma
admissions — and it is precisely the one that is hardest to anticipate from the metric
most Malaysians actually consult.

## 1.2 Problem Statement

The Department of Environment (DOE) publishes the Air Pollutant Index (API), which is
derived from a **24-hour moving average**. That design is defensible for regulatory
reporting and long-term health epidemiology: it smooths out transient spikes and
produces a stable, comparable number. It is, however, structurally unsuitable for
short-horizon decisions, for two reasons.

**First, a 24-hour average is a lagging filter, not a leading one.** A haze front that
arrives at 14:00 and lifts PM2.5 from 25 to 95 µg/m³ contributes only 70 µg/m³ to the
24-hour mean, spread across a denominator that is mostly clean air. The index therefore
crosses the "unhealthy" threshold hours after the air has already become unhealthy. The
information needed to act has been averaged away.

**Second, the index reports a level, not a trajectory.** Even a perfectly current
reading answers "how bad is it now", which is the wrong question at 07:00 when the
decision is whether to cycle to campus at 10:00. The decision-relevant quantity is the
three-hour-ahead level, together with an honest statement of how uncertain that
projection is.

A third, subtler problem is that the index is a **scalar with an ordinal meaning**. The
four bands (good / moderate / unhealthy / very unhealthy) are ordered, and the cost of
misclassification is violently asymmetric: telling someone the air is moderate when it
is in fact very unhealthy is not the same magnitude of error as the reverse. Any model
that treats the four levels as four interchangeable categories has discarded both the
ordering and the asymmetry before it has seen a single sample.

## 1.3 Objectives

This work develops and evaluates a three-hour-ahead PM2.5 risk nowcasting system for the
Klang Valley. Specifically, it aims to:

1. **O1** — Construct a leakage-free, chronologically split panel of ground-truth-referenced
   atmospheric composition and meteorology covering the Klang Valley and its upwind
   source regions, at hourly resolution over fifty-one months.
2. **O2** — Separate the climatological background from pollution anomalies, so that a
   model is forced to learn *deviation from normal* rather than *which month it is*.
3. **O3** — Compare six model families under a single evaluation contract: a decision-tree
   floor, nominal random forests with and without class reweighting, a
   regression-then-threshold formulation, a boosting baseline, an ordinal decomposition,
   and a cost-sensitive ordinal model whose decision thresholds are tuned by grid search.
4. **O4** — Quantify what happens when the model meets an unseen station (leave-one-node-out)
   and an unseen year (temporal extrapolation), rather than reporting a single
   in-distribution score.
5. **O5** — Deliver the whole pipeline as a reproducible, inspectable artifact suitable for
   classroom demonstration, including the negative results.

## 1.4 Scope and Honest Framing

Two limitations are stated here rather than buried in a conclusions section, because they
shape every number that follows.

**The pollutant fields are model products, not regulatory monitors.** The air-quality
variables are taken from the Copernicus Atmosphere Monitoring Service (CAMS) global
composition product distributed by Open-Meteo, at a native resolution of 0.4°
(approximately 45 km). Values are therefore grid-cell estimates from a
physics-and-chemistry assimilation system, not readings from a DOE station. This is
adequate for studying *regional transport*, whose characteristic scale is hundreds of
kilometres, and inadequate for making claims about street-level exposure.

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

Two consequences follow, and both were designed around rather than papered over:

1. **The upwind-transport feature requires regional, not municipal, separation.** Computing
   an advection index between two coordinates that share a grid cell is meaningless — the
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
`wd_sin` and `wd_cos` are derived. This is unit handling, not feature engineering — no
temporal lag or spatial aggregation is involved.

## 2.1.4 Quality Control Procedure

Applied per node, in order:

1. **Deduplication** on `(station_id, timestamp)`, first occurrence retained. *Rows removed: 0.*
2. **Hourly grid reconstruction** via `reindex` on a complete `pd.date_range`. Without this,
   `shift(1)` means "the previous record", not "one hour ago", and every lag silently
   becomes wrong whenever a row is missing. *Rows added: 0* (all nodes already complete).
3. **Leading-block trimming** — the CAMS archive edge. *Rows trimmed: 2,080*
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

## 2.1.5 Target Construction

Both targets are built with `groupby("station_id")["pm2_5"].shift(-3)`:

- `target_pm25_3h` — the PM2.5 concentration three hours ahead;
- `target_level_3h` — its ordinal band, via thresholds (15, 35, 55) µg/m³.

The `groupby` is not incidental. A global `shift(-3)` would concatenate the tail of one
node to the head of the next, manufacturing labels that mix two locations. This is a
silent failure: the code runs, the model trains, and only the metrics are quietly wrong.

## 2.1.6 Gate 0: Feasibility Checks Before Modelling

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

PM2.5 has an overwhelming seasonal and diurnal structure. A model given raw concentrations
and a month index can achieve respectable accuracy by learning "July is hazier than
November" and never represent transport at all. Worse, it will look competent in-sample
and collapse on an anomalous event — precisely the case the system exists for.

The baseline removes the predictable part so that the model is forced to explain the
residual:

$$b_{\text{proto}}(s, m, h) = \mathbb{E}\left[\text{PM}_{2.5} \mid \text{station}=s,\ \text{month}=m,\ \text{hour}=h\right]$$

$$a(t) = x(t) - b_{\text{proto}}(s, m(t), h(t))$$

The same operation is applied to 2 m temperature, yielding `temp_anomaly`, which captures
whether the boundary layer is unusually warm (convective, dispersive) or unusually cool
(stable, accumulating).

**The baseline is fitted on the training split only**, then broadcast to validation and
test. Computing it over the full panel would leak future information into training
features through the group means — the most common and least visible form of leakage in
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

This is not a defect to be smoothed away. It means the reported High-Risk Recall is
measured on a real event rather than on background noise, and it is the reason the
cost-sensitive models have something meaningful to improve. It also means the test set is
*harder* than the training distribution (Level 3: 18.85 % vs 11.13 %), which should be
stated whenever the test numbers are compared against published in-distribution results.

Receptor-level distributions:

| Split | Level 0 | Level 1 | Level 2 | Level 3 |
|---|---|---|---|---|
| train | 18.54 % | 50.29 % | 20.05 % | 11.13 % |
| test | 11.78 % | 47.18 % | 22.18 % | 18.85 % |

The distribution is workably balanced — no class falls below 11 % — so this project does
not face the extreme-rarity regime that motivates most cost-sensitive literature. The
asymmetry that matters here is in the **cost matrix**, not in the class frequencies.
Stating this plainly is preferable to invoking "severe imbalance" as a reflex.

---

# Section 2.2.4 — Baseline Decision Tree

## 2.2.4.1 Configuration

The baseline is deliberately austere: current-hour observations only, no lag, no rolling
statistic, no spatial aggregation. Features are the six pollutants, five meteorological
variables, the two anomaly residuals, three calendar variables, and the wind sin/cos pair
— eighteen columns in total.

Hyperparameters are selected by `TimeSeriesSplit(n_splits=5)` grid search scored on
macro-F1, over `max_depth ∈ {3, 5, 7, 9, 12, None}`, `min_samples_leaf ∈ {5, 20, 50, 100}`
and `criterion ∈ {gini, entropy}` — 48 configurations.

Selected: `criterion=gini, max_depth=9, min_samples_leaf=20`, giving 384 leaves and a
depth of 9. Cross-validated macro-F1: **0.6315**.

Overfitting check: training macro-F1 0.7017 vs validation 0.6867, a gap of **0.015**. The
model is not memorising; the remaining error is structural, which is exactly what a floor
should look like.

## 2.2.4.2 Results

Test split, receptor nodes, n = 16,728 (of which 6,865 high-risk):

| Metric | M0 Persistence | M1 Decision Tree | Δ (M1 − M0) |
|---|---|---|---|
| Macro-F1 | 0.6222 | **0.6620** | +0.0398 |
| Balanced Accuracy | 0.6222 | **0.6443** | +0.0221 |
| High-Risk Recall (L2 ∪ L3) | **0.7752** | 0.7693 | **−0.0059** |
| MAOE | 0.3930 | **0.3220** | −0.0710 |
| Severe Jumping Error Rate | 0.0375 | **0.0218** | −0.0157 |
| Average Asymmetric Cost | 0.9986 | **0.8905** | −0.1081 |

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
has learned "tomorrow's air resembles today's air, adjusted for the time of day" — and
that is a strong prior precisely because PM2.5 is autocorrelated. The burden of proof for
any added complexity is therefore not "beat a naive model" but "beat persistence".

**Aggregate metrics improve, but the metric that matters does not.** Macro-F1 rises by
0.04, MAOE falls by 18 %, and average asymmetric cost falls by 11 %; on every summary
statistic the tree looks like an improvement. Yet its union recall (0.7693) is *lower* than
that of pure persistence (0.7752). Training a model and thereby losing ground on "should we
raise an alarm" is the single most important result of this section: it shows that
aggregate accuracy is the wrong objective for a warning system, and it is the empirical
justification for the ordinal decomposition (M6) and the cost-sensitive threshold search
(M7) that follow.

The confusion matrix explains the mechanism. 1,348 true Level 2 cases are predicted as
Level 1 and 758 true Level 3 cases as Level 2, while false alarms in the opposite direction
are rare (9 Level 2 predicted from Level 0). The tree suppresses the boundary to accommodate
Level 1, which holds roughly half the test set. That is hedging toward the majority, not
random error.

**Cross-cell performance is uneven, and the pattern is physical.** `Port_Klang` scores
highest on macro-F1 but lowest on high-risk recall; it is the cleanest receptor (Level 3
share 1.9 %), so its minority class is least represented in training. `KL_City` shows the
opposite: lowest macro-F1, highest high-risk recall, because it carries the widest
dynamic range. A single pooled number would have hidden both behaviours.

These three observations motivate the ordinal decomposition (which attacks the downward
hedging directly by modelling cumulative probabilities), the class reweighting variants,
and the cost-sensitive threshold search — rather than being introduced as unmotivated
extras.
