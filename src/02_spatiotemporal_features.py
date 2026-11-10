"""Stage 2 features for the confirmed 26-cell panel.

All histories use exact hourly timestamps within (station_id, split). Rolling
windows end at issue time t and include t. No interpolation, backfill, learned
imputation or resplitting is performed. Original columns and row order survive.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from stage2_storage import digest, feature_artifact, write_feature_parts

LAG_COLUMNS = ("pm2_5", "pm10", "no2", "relative_humidity_2m")
LAGS = (1, 2, 3, 6, 12, 24)
WINDOWS = (3, 6, 12)
TIMEZONE = "Asia/Kuala_Lumpur"
CALM_MS = 0.5
WEIGHT_EPS = 1e-12  # m/s; avoid normalizing effectively zero transport strength
NEAR_RADIUS_KM = 150.0
NEAR_SIGMA_KM = 60.0
NEAR_LAGS = (1, 2, 3)
FAR_REGIONS = ("Sumatra", "Borneo")
FAR_WINDOWS = (12, 24, 48)
FAR_LAG = 24
FAR_COLUMNS = [f"far_source_pm25_mean_{h}h" for h in FAR_WINDOWS] + [f"far_source_pm25_lag_{FAR_LAG}h"]
NEAR_DIAGNOSTIC_COLUMNS = ([f"upwind_weight_sum_{h}h" for h in NEAR_LAGS]
                           + [f"upwind_neighbor_count_{h}h" for h in NEAR_LAGS])
NEAR_COLUMNS = ([f"upwind_pm25_advection_{h}h" for h in NEAR_LAGS] + ["spatial_pm25_gradient"]
                + NEAR_DIAGNOSTIC_COLUMNS)
AUDIT_COLUMNS = ("audit_target_within_split_3h", "audit_target_aligned_3h", "audit_target_safe_3h",
                 "audit_source_clean", "audit_history_complete", "audit_model_eligible")


def git_revision():
    """Support source archives without Git, without discovering a parent repository."""
    if (ROOT / ".git").exists():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    provenance = ROOT / "SOURCE_COMMIT.txt"
    if provenance.exists():
        revision = provenance.read_text(encoding="utf-8").strip()
        if re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", revision):
            return revision.lower()
    return None


def default_input():
    # The confirmed 26-cell panel is the only default. Legacy input is explicit.
    return ROOT / "data/haze_panel_clean.parquet"


def local_time(values):
    ts = pd.to_datetime(values, errors="raise")
    if ts.isna().any():
        raise ValueError("timestamp contains NaT")
    ts = ts.dt.tz_localize(TIMEZONE) if ts.dt.tz is None else ts.dt.tz_convert(TIMEZONE)
    if not ts.eq(ts.dt.floor("h")).all():
        raise ValueError("timestamps must fall on exact hours")
    return ts


def clean_rows(df):
    """Conservative row-level exclusion of retrospective interpolation/missingness."""
    ok = pd.Series(True, index=df.index)
    for col in ("qc_interpolated", "is_missing"):
        if col in df:
            ok &= df[col].eq(0)
    return ok


def wind_speed_ms(df, raw_unit=None):
    """Never infer units from magnitude; the unqualified v2.0 column needs a unit."""
    if "wind_speed_ms" in df:
        speed = df.wind_speed_ms.astype(float)
        if "wind_speed_kmh" in df:
            if not np.allclose(speed, df.wind_speed_kmh / 3.6,
                               rtol=1e-6, atol=1e-7, equal_nan=True):
                raise ValueError("wind_speed_ms disagrees with wind_speed_kmh / 3.6")
    elif "wind_speed_kmh" in df:
        speed = df.wind_speed_kmh.astype(float) / 3.6
    elif "wind_speed_10m" in df and raw_unit in ("km/h", "m/s"):
        speed = df.wind_speed_10m.astype(float) / (3.6 if raw_unit == "km/h" else 1)
    else:
        raise ValueError("Need wind_speed_ms, wind_speed_kmh, or explicit --wind-speed-unit")
    if ((speed.dropna() < 0) | ~np.isfinite(speed.dropna())).any():
        raise ValueError("wind speed must be finite and nonnegative, or missing")
    return speed


def wind_vectors(df, raw_unit=None):
    speed = wind_speed_ms(df, raw_unit)
    direction = df.wind_direction_10m.astype(float)
    if not direction.dropna().between(0, 360).all():
        raise ValueError("wind direction must lie in [0, 360] degrees")
    angle = np.deg2rad(direction)
    u, v = -speed * np.sin(angle), -speed * np.cos(angle)
    calm = speed.lt(CALM_MS)
    if "qc_calm_wind" in df:
        calm |= df.qc_calm_wind.eq(1)
    # A calm flag cannot supply a missing speed. Known zero speed needs no direction.
    calm &= speed.notna()
    u, v = u.mask(calm, 0.0), v.mask(calm, 0.0)
    return u.where(clean_rows(df)), v.where(clean_rows(df))


def geometry(coords):
    """Matrices [source j, destination i]: km distance and EN unit bearing at j."""
    lat, lon = np.deg2rad(np.asarray(coords, dtype=float)).T
    dlat = lat[None, :] - lat[:, None]
    dlon = lon[None, :] - lon[:, None]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    distance = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    east = np.sin(dlon) * np.cos(lat[None, :])
    north = (np.cos(lat[:, None]) * np.sin(lat[None, :])
             - np.sin(lat[:, None]) * np.cos(lat[None, :]) * np.cos(dlon))
    norm = np.hypot(east, north)
    e = np.divide(east, norm, out=np.zeros_like(east), where=norm > 1e-12)
    n = np.divide(north, norm, out=np.zeros_like(north), where=norm > 1e-12)
    return distance, e, n


def validate_panel(df):
    required = {"station_id", "timestamp", "split", "lat", "lon",
                "wind_direction_10m", *LAG_COLUMNS}
    if required - set(df):
        raise ValueError(f"Missing columns: {sorted(required - set(df))}")
    if df.empty or df[["station_id", "split", "lat", "lon"]].isna().any().any():
        raise ValueError("empty panel or missing identifiers/coordinates")
    if not df.split.isin(["train", "val", "test"]).all():
        raise ValueError("split must be train, val or test")
    if not df.lat.between(-90, 90).all() or not df.lon.between(-180, 180).all():
        raise ValueError("coordinates out of range")
    if np.isinf(df[list(LAG_COLUMNS)].to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("lag inputs must be finite or missing")
    ts = local_time(df.timestamp)
    keys = pd.DataFrame({"station_id": df.station_id, "timestamp": ts})
    if keys.duplicated().any():
        raise ValueError("duplicate (station_id, timestamp)")
    for _, sub in df.assign(_time=ts).groupby("station_id", observed=True):
        if (sub[["lat", "lon"]].nunique() != 1).any():
            raise ValueError("coordinates change within a node")
        ranks = sub.sort_values("_time").split.astype(str).map({"train": 0, "val": 1, "test": 2})
        if (ranks.diff().dropna() < 0).any():
            raise ValueError("split intervals are interleaved or reversed")
    return ts


def temporal_features(work):
    names = [f"{c}_lag_{h}" for c in LAG_COLUMNS for h in LAGS]
    names += [f"pm2_5_{s}_{h}h" for h in WINDOWS for s in ("mean", "std", "range")]
    names += ["pm2_5_change_3h", "pm2_5_slope_6h"]
    out = pd.DataFrame(np.nan, index=work.index, columns=names)
    for _, sub in work.groupby(["station_id", "split"], observed=True, sort=False):
        sub = sub.sort_values("_time")
        # Reindex BEFORE shifting: missing timestamps never turn a row lag into an hour lag.
        hourly = pd.date_range(sub._time.min(), sub._time.max(), freq="h")
        values = sub.set_index("_time")[list(LAG_COLUMNS)].astype(float)
        values.loc[~clean_rows(sub).to_numpy(), :] = np.nan
        values = values.reindex(hourly)
        result = {}
        for c in LAG_COLUMNS:
            for h in LAGS:
                result[f"{c}_lag_{h}"] = values[c].shift(h)
        x = values.pm2_5
        for h in WINDOWS:
            r = x.rolling(h, min_periods=h)
            result[f"pm2_5_mean_{h}h"] = r.mean()
            result[f"pm2_5_std_{h}h"] = r.std(ddof=1)
            result[f"pm2_5_range_{h}h"] = r.max() - r.min()
        result["pm2_5_change_3h"] = x - x.shift(3)
        centered = np.arange(6) - 2.5
        result["pm2_5_slope_6h"] = x.rolling(6, min_periods=6).apply(
            lambda a: np.dot(a, centered) / np.dot(centered, centered), raw=True)
        out.loc[sub.index, :] = pd.DataFrame(result).reindex(sub._time).to_numpy()
    return out


def spatial_network(work, sigma_km, radius_km):
    if not np.isfinite(sigma_km) or sigma_km <= 0:
        raise ValueError("sigma_km must be positive and finite")
    nodes = work.groupby("station_id", observed=True)[["lat", "lon"]].first().sort_index()
    distance, east, north = geometry(nodes.to_numpy())
    candidates = distance >= 1e-6
    if radius_km is not None:
        if not np.isfinite(radius_km) or radius_km <= 0:
            raise ValueError("radius_km must be positive and finite")
        candidates &= distance <= radius_km
        if "region" in work:
            # Separate domestic near field from the two offshore source regions.
            region = (work.groupby("station_id", observed=True).region.first()
                      .reindex(nodes.index).astype(str).replace({"KL": "Peninsula"}).to_numpy())
            candidates &= region[:, None] == region[None, :]
    kernel = np.where(candidates, np.exp(-.5 * (distance / sigma_km) ** 2), 0.)
    return nodes.index, distance, east, north, candidates, kernel


def spatial_clock(sub, nodes):
    clock = pd.date_range(sub._time.min(), sub._time.max(), freq="h")
    def pivot(column):
        return (sub.pivot(index="_time", columns="station_id", values=column)
                .reindex(index=clock, columns=nodes).to_numpy(dtype=float, na_value=np.nan))
    return clock, pivot("_safe_pm25"), pivot("u_wind"), pivot("v_wind")


def spatial_features(work, sigma_km, radius_km=NEAR_RADIUS_KM):
    nodes, _, east, north, candidates, kernel = spatial_network(work, sigma_km, radius_km)
    out = pd.DataFrame(np.nan, index=work.index, columns=NEAR_COLUMNS)
    # Each split gets its own time grid. Source data must also belong to this split.
    for _, sub in work.groupby("split", observed=True, sort=False):
        clock, pm, u, v = spatial_clock(sub, nodes)
        observed = np.isfinite(pm) & np.isfinite(u) & np.isfinite(v)
        for i, sid in enumerate(nodes):
            idx = sub.index[sub.station_id.eq(sid)]
            if not len(idx):
                continue
            weights, valid = directional_weights(pm, u, v, east[:, i], north[:, i], kernel[:, i])
            denom = weights.sum(axis=1)
            numerator = (np.where(valid, pm, 0) * weights).sum(axis=1)
            aggregate = np.divide(numerator, denom, out=np.full(len(clock), np.nan), where=denom > 0)
            missing_inputs = candidates[:, i].any() & ~(observed & candidates[:, i]).any(axis=1)
            history = pd.DataFrame({"advection": aggregate, "weight_sum": np.where(missing_inputs, np.nan, denom),
                                    "neighbor_count": np.where(missing_inputs, np.nan, valid.sum(axis=1))}, index=clock)
            target_times = work.loc[idx, "_time"]
            for k in NEAR_LAGS:
                lag = history.shift(k).reindex(target_times)
                columns = [f"upwind_pm25_advection_{k}h", f"upwind_weight_sum_{k}h", f"upwind_neighbor_count_{k}h"]
                out.loc[idx, columns] = lag.to_numpy()
            # Same-time target minus same-time weighted neighbors; all available at t by assumption.
            out.loc[idx, "spatial_pm25_gradient"] = (work.loc[idx, "_safe_pm25"].to_numpy()
                                                       - history.advection.reindex(target_times).to_numpy())
    return out


def directional_weights(pm, u, v, east, north, kernel):
    """The shared production/diagnostic weights; arrays are [hour, source]."""
    projection = u * east + v * north
    # Remove floating-point cos(pi/2) noise from exactly perpendicular winds.
    weights = np.where(projection > 1e-12, projection, 0) * kernel
    valid = np.isfinite(pm) & np.isfinite(weights) & (weights > WEIGHT_EPS)
    return np.where(valid, weights, 0), valid


def far_field_features(work):
    """Unweighted source-region PM2.5 history, strictly before the issue hour."""
    out = pd.DataFrame(np.nan, index=work.index, columns=FAR_COLUMNS)
    if "region" not in work:
        return out
    for _, sub in work.groupby("split", observed=True, sort=False):
        sources = sub.loc[sub.region.isin(FAR_REGIONS)]
        if sources.empty:
            continue
        clock = pd.date_range(sub._time.min(), sub._time.max(), freq="h")
        hourly = sources.groupby("_time")._safe_pm25.mean().reindex(clock)
        # Shift before rolling so t itself never contributes to the background.
        past = hourly.shift(1)
        result = {f"far_source_pm25_mean_{w}h": past.rolling(w, min_periods=w).mean() for w in FAR_WINDOWS}
        result[f"far_source_pm25_lag_{FAR_LAG}h"] = hourly.shift(FAR_LAG)
        out.loc[sub.index, :] = pd.DataFrame(result).reindex(sub._time).to_numpy()
    return out


def numeric_summary(values):
    """Finite-only JSON summary; unavailable statistics are null rather than zero."""
    values = np.asarray(values, dtype=float).ravel()
    finite = values[np.isfinite(values)]
    result = {"n": int(len(values)), "n_finite": int(len(finite))}
    for name, q in (("min", 0), ("p25", .25), ("median", .5), ("p75", .75),
                    ("p95", .95), ("max", 1)):
        result[name] = float(np.quantile(finite, q)) if len(finite) else None
    result["mean"] = float(finite.mean()) if len(finite) else None
    return result


SUPPORT_REASONS = ("supported", "no_geometric_neighbors", "missing_source_inputs", "calm_sources_only",
                   "no_positive_projection", "below_weight_floor")


def support_reasons(pm, u, v, east, north, candidates, denominator):
    """Separate absence of geometry, observations and directional support."""
    observed = np.isfinite(pm) & np.isfinite(u) & np.isfinite(v) & candidates
    noncalm = observed & ((u != 0) | (v != 0))
    positive = observed & (u * east + v * north > WEIGHT_EPS)
    return np.select(
        [denominator > 0, np.full(len(pm), not candidates.any()), ~observed.any(axis=1),
         ~noncalm.any(axis=1), ~positive.any(axis=1)],
        SUPPORT_REASONS[:-1], default=SUPPORT_REASONS[-1],
    )


def support_summary(reasons):
    counts = {reason: int((reasons == reason).sum()) for reason in SUPPORT_REASONS}
    return support_count_summary(counts)


def support_count_summary(counts):
    rows = sum(counts.values())
    unsupported = rows - counts["supported"]
    comparable = rows - counts["missing_source_inputs"]
    return {"rows": rows, "zero_weight_rows": unsupported,
            "zero_weight_fraction": unsupported / rows if rows else None,
            "missing_source_input_fraction": counts["missing_source_inputs"] / rows if rows else None,
            "computable_rows": comparable,
            "zero_weight_fraction_among_computable": ((unsupported - counts["missing_source_inputs"]) / comparable
                                                       if comparable else None),
            "support_reason_counts": counts}


def spatial_training_diagnostics(train, sigma_km=NEAR_SIGMA_KM, raw_wind_unit=None, *, radius_km=NEAR_RADIUS_KM):
    """Diagnose the fixed spatial design without using held-out rows or labels.

    Coordinates are the available sampling coordinates, not verified CAMS cell
    centres. Weight summaries use exactly the production threshold and QC mask.
    """
    if train.empty or "split" not in train or not train.split.eq("train").all():
        raise ValueError("spatial diagnostics require nonempty train-only input")
    return spatial_split_diagnostics(train, sigma_km, raw_wind_unit, radius_km=radius_km)


def spatial_split_diagnostics(frame, sigma_km, raw_wind_unit=None, *, radius_km=NEAR_RADIUS_KM):
    if frame.empty or frame.split.nunique() != 1:
        raise ValueError("spatial diagnostics require one nonempty split")
    work = frame.reset_index(drop=True).copy()
    work["_time"] = validate_panel(work)
    work["u_wind"], work["v_wind"] = wind_vectors(work, raw_wind_unit)
    work["_safe_pm25"] = work.pm2_5.where(clean_rows(work))
    nodes, distance, east, north, candidates, kernel = spatial_network(work, sigma_km, radius_km)
    pairs = np.triu_indices(len(nodes), 1)
    duplicate_pairs = [{"node_a": str(nodes[i]), "node_b": str(nodes[j])}
                       for i, j in zip(*pairs) if distance[i, j] < 1e-6]
    others = distance.copy()
    np.fill_diagonal(others, np.inf)
    distinct = np.where(distance >= 1e-6, distance, np.inf)
    nearest = others.min(axis=1)
    nearest_distinct = distinct.min(axis=1)
    clock, pm, u, v = spatial_clock(work, nodes)
    by_node, combined, all_reasons = {}, [], np.empty(len(work), dtype="<U24")
    for i, sid in enumerate(nodes):
        weights, valid = directional_weights(pm, u, v, east[:, i], north[:, i], kernel[:, i])
        denom = weights.sum(axis=1)
        neighbors = valid.sum(axis=1)
        normalized = np.divide(weights, denom[:, None], out=np.zeros_like(weights),
                               where=denom[:, None] > 0)
        dominant = normalized.max(axis=1)
        dominant[denom == 0] = np.nan
        concentration = (normalized ** 2).sum(axis=1)
        effective = np.divide(1., concentration, out=np.full(len(clock), np.nan), where=concentration > 0)
        target = work.loc[work.station_id.eq(sid)]
        positions = clock.get_indexer(target._time)
        d, n, m, e = (x[positions] for x in (denom, neighbors, dominant, effective))
        reasons = support_reasons(pm, u, v, east[:, i], north[:, i], candidates[:, i], denom)
        all_reasons[target.index] = reasons[positions]
        unavailable = np.isin(reasons[positions], ["no_geometric_neighbors", "missing_source_inputs"])
        lag_stats = {}
        for k in NEAR_LAGS:
            history = pd.Series(denom, index=clock).shift(k).reindex(target._time)
            missing_input = pd.Series(reasons, index=clock).shift(k).reindex(target._time).eq("missing_source_inputs")
            lag_stats[str(k)] = {"positive_weight_rows": int(history.gt(0).sum()),
                                 "zero_weight_rows": int((history.eq(0) & ~missing_input).sum()),
                                 "missing_source_input_rows": int(missing_input.sum()),
                                 "missing_history_rows": int(history.isna().sum()),
                                 "coverage_fraction": float(history.gt(0).mean())}
        by_node[str(sid)] = {
            **support_summary(reasons[positions]),
            "geometric_neighbor_count": int(candidates[:, i].sum()),
            "geometric_neighbors": [str(node) for node in nodes[candidates[:, i]]],
            "all_other_source_observations_unavailable_rows": int(unavailable.sum()),
            "weight_sum_ms": numeric_summary(d), "positive_weight_neighbor_count": numeric_summary(n),
            "max_normalized_weight_when_available": numeric_summary(m),
            "effective_neighbor_count_when_available": numeric_summary(e),
            "one_neighbor_fraction_when_available": float((n[d > 0] == 1).mean()) if (d > 0).any() else None,
            "advection_availability_by_lag_h": lag_stats,
        }
        combined.append(np.column_stack([d, n, m, e]))
    all_rows = np.concatenate(combined, axis=0)
    speed = wind_speed_ms(work, raw_wind_unit)
    missing_wind = work.u_wind.isna() | work.v_wind.isna()
    calm_wind = work.u_wind.eq(0) & work.v_wind.eq(0)
    speed_band = np.select([missing_wind, calm_wind, speed.lt(2), speed.lt(4)],
                           ["unavailable", "calm", "noncalm_below_2_m_s", "2_to_below_4_m_s"],
                           default="at_least_4_m_s")
    return {
        "scope": f"{work.split.iloc[0]} target rows; target labels are excluded",
        "sigma_km": float(sigma_km), "radius_km": radius_km, "minimum_weight_ms": WEIGHT_EPS,
        "candidate_rule": ("Distance only, no hard radius (legacy baseline)" if radius_km is None else
                           "Within hard radius; KL and Peninsula form one group, other named regions are separate"),
        "rows": len(work), "node_count": len(nodes),
        "time_min": work._time.min().isoformat(), "time_max": work._time.max().isoformat(),
        "coordinate_basis": "Available sampling coordinates; exact atmospheric-model cell centres are not verified.",
        "coordinate_pair_count": len(pairs[0]), "duplicate_coordinate_pairs": duplicate_pairs,
        "pair_distance_km": numeric_summary(distance[pairs]),
        "nearest_neighbor_distance_km": numeric_summary(nearest),
        "nearest_distinct_neighbor_distance_km": numeric_summary(nearest_distinct),
        "nearest_neighbor_by_node": {
            str(sid): {"nearest_km": float(nearest[i]) if np.isfinite(nearest[i]) else None,
                       "nearest_distinct_km": float(nearest_distinct[i]) if np.isfinite(nearest_distinct[i]) else None}
            for i, sid in enumerate(nodes)},
        "coordinate_pairs_within_sigma": int((distance[pairs] <= sigma_km).sum()),
        "coordinate_pairs_within_3sigma": int((distance[pairs] <= 3 * sigma_km).sum()),
        "coordinate_pairs_within_radius": (int((distance[pairs] <= radius_km).sum())
                                           if radius_km is not None else None),
        "coordinate_pairs_in_neighborhood": int(candidates[pairs].sum()),
        "neighborhood_pairs_within_sigma": int((candidates & (distance <= sigma_km))[pairs].sum()),
        "source_quality": {"excluded_qc_rows": int((~clean_rows(work)).sum()),
                           "missing_pm25_rows": int(work.pm2_5.isna().sum()),
                           "missing_wind_speed_rows": int(speed.isna().sum()),
                           "missing_wind_direction_rows": int(work.wind_direction_10m.isna().sum()),
                           "unavailable_wind_vector_rows": int((work.u_wind.isna() | work.v_wind.isna()).sum())},
        "overall": {**support_summary(all_reasons),
                    "weight_sum_ms": numeric_summary(all_rows[:, 0]),
                    "positive_weight_neighbor_count": numeric_summary(all_rows[:, 1]),
                    "max_normalized_weight_when_available": numeric_summary(all_rows[:, 2]),
                    "effective_neighbor_count_when_available": numeric_summary(all_rows[:, 3])},
        "by_node": by_node,
        "by_target_wind_speed": {band: support_summary(all_reasons[speed_band == band])
                                 for band in ("unavailable", "calm", "noncalm_below_2_m_s",
                                              "2_to_below_4_m_s", "at_least_4_m_s")},
        "definitions": {"effective_neighbor_count": "1 / sum(normalized_weight_j ** 2); null at zero weight",
                        "positive_weight_neighbor_count": "Number of QC-valid source weights > minimum_weight_ms",
                        "zero_weight": "All target rows without a positive usable weight, including unavailable inputs. "
                                       "The same unfiltered denominator is used for legacy and current comparisons.",
                        "unavailable_inputs": "A geometric source exists but no candidate has finite, QC-valid wind and PM2.5.",
                        "weight_concentration": "Maximum normalized weight; values near 1 imply one-source dominance",
                        "model_series_similarity": "See training-only spatial_similarity in metrics_stage2_eda.json"},
        "limitations": ["The fixed near-field scale and sensitivity values are evaluated on training rows only.",
                         "Directional weights are associations and do not establish physical transport or travel times."]}


def label_audit(work):
    """Do not change upstream labels; expose exact t+3h and split-boundary validity."""
    table = work.set_index(["station_id", "_time"])
    future = pd.MultiIndex.from_arrays([work.station_id, work._time + pd.Timedelta(hours=3)])
    nxt = table.reindex(future).reset_index(drop=True)
    same = nxt.split.astype(str).to_numpy() == work.split.astype(str).to_numpy()
    aligned = np.zeros(len(work), dtype=bool)
    if {"target_pm25_3h", "target_level_3h"} <= set(work):
        target = work.target_pm25_3h.to_numpy()
        aligned = np.isclose(target, nxt.pm2_5.to_numpy(), atol=1e-6, rtol=1e-7)
        aligned &= work.target_level_3h.to_numpy() == np.searchsorted([15, 35, 55], target, side="right")
    safe = same & aligned & clean_rows(nxt).to_numpy()
    return pd.DataFrame({"audit_target_within_split_3h": same,
                         "audit_target_aligned_3h": aligned,
                         "audit_target_safe_3h": safe}, index=work.index)


def build_features(df, sigma_km=NEAR_SIGMA_KM, raw_wind_unit=None):
    ts = validate_panel(df.reset_index(drop=True))
    work = df.reset_index(drop=True).copy()
    work["_time"] = ts
    u, v = wind_vectors(work, raw_wind_unit)
    work["u_wind"], work["v_wind"] = u, v
    work["_safe_pm25"] = work.pm2_5.where(clean_rows(work))
    temporal = temporal_features(work)
    added = temporal.copy()
    for name, values, period in (("hour", ts.dt.hour, 24), ("dayofweek", ts.dt.dayofweek, 7),
                                 ("month", ts.dt.month - 1, 12)):
        added[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        added[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
    added["u_wind"], added["v_wind"] = u, v
    added = pd.concat([added, spatial_features(work, sigma_km), far_field_features(work), label_audit(work)], axis=1)
    added["audit_source_clean"] = clean_rows(work)
    added["audit_history_complete"] = temporal.notna().all(axis=1)
    added["audit_model_eligible"] = (added.audit_target_safe_3h & added.audit_source_clean
                                     & added.audit_history_complete)
    overlap = set(df.columns) & set(added.columns)
    if overlap:
        raise ValueError(f"Refusing to overwrite existing columns: {sorted(overlap)}")
    # Single precision for added continuous features only; original values/dtypes stay exact.
    for c in added.select_dtypes(include="floating"):
        added[c] = added[c].astype("float32")
    added.index = df.index
    return pd.concat([df.copy(), added], axis=1)


def combine_spatial_support(by_split):
    """Combine coverage counts, retaining the same denominator for both models."""
    def combine(parts):
        return support_count_summary({reason: sum(p["support_reason_counts"][reason] for p in parts)
                                      for reason in SUPPORT_REASONS})

    def combine_key(group):
        keys = sorted({key for split in by_split.values() for key in split[group]})
        return {key: combine([split[group][key] for split in by_split.values() if key in split[group]])
                for key in keys}

    return {"overall": combine([split["overall"] for split in by_split.values()]),
            "by_node": combine_key("by_node"), "by_target_wind_speed": combine_key("by_target_wind_speed"),
            "by_split": {name: combine([split["overall"]]) for name, split in by_split.items()}}


def far_field_summary(out):
    sources = out.loc[out.region.isin(FAR_REGIONS)] if "region" in out else out.iloc[:0]
    by_region = {}
    for region in FAR_REGIONS:
        rows = sources.loc[sources.region.eq(region)] if len(sources) else sources
        valid = rows.loc[clean_rows(rows) & np.isfinite(rows.pm2_5)]
        by_region[region] = {"registered_nodes": int(rows.station_id.nunique()), "rows": len(rows),
                             "qc_valid_rows": len(valid), "distinct_hours_with_valid_data": int(valid.timestamp.nunique())}
    return {"definition": "Unweighted mean across QC-valid Sumatra and Borneo PM2.5 cells each hour; "
                          "strictly past rolling windows require every hour in the same split.",
            "windows_hours": FAR_WINDOWS, "lag_hours": FAR_LAG, "by_region": by_region,
            "coverage_by_split": {str(split): {c: float(g[c].notna().mean()) for c in FAR_COLUMNS}
                                  for split, g in out.groupby("split", observed=True)},
            "interpretation": "Regional background, without a wind weight or assumed travel time."}


def spatial_column_checks(out):
    train = out.loc[out.split.eq("train"), NEAR_COLUMNS + FAR_COLUMNS]
    cols = list(train)
    observed = {column: int(train[column].notna().sum()) for column in cols}
    unique = {column: int(train[column].nunique()) for column in cols}
    return {"scope": "train-only; equality compares values and null positions",
            "observed_counts": observed, "distinct_observed_values": unique,
            "all_missing": [column for column in cols if not observed[column]],
            "constant_when_observed": [column for column in cols if unique[column] == 1],
            "exact_duplicate_pairs": [[left, right] for i, left in enumerate(cols) for right in cols[i + 1:]
                                      if train[left].equals(train[right])]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=default_input())
    ap.add_argument("--output", type=Path, default=ROOT / "data/feat_spatiotemporal.parquet")
    ap.add_argument("--audit", type=Path, default=ROOT / "models/metrics_stage2_features.json")
    ap.add_argument("--sigma-km", type=float, default=NEAR_SIGMA_KM)
    ap.add_argument("--wind-speed-unit", choices=["km/h", "m/s"])
    args = ap.parse_args()
    paths = [args.input.resolve(), args.output.resolve(), args.audit.resolve()]
    if len(set(paths)) != len(paths) or any(paths[1] in path.parents for path in (paths[0], paths[2])):
        ap.error("input is read-only; input, output and audit must be different paths")
    before = digest(args.input)
    df = pd.read_parquet(args.input)
    out = build_features(df, args.sigma_km, args.wind_speed_unit)
    storage = write_feature_parts(out, args.output)
    assert before == digest(args.input), "input changed during feature construction"
    extra = [c for c in out if c not in df]
    rec = out.role.eq("receptor") if "role" in out else np.ones(len(out), dtype=bool)
    unit = ("wind_speed_ms (m/s), cross-checked against km/h when present" if "wind_speed_ms" in df
            else f"km/h converted once, or explicit raw unit {args.wind_speed_unit}")
    gate_path = ROOT / "models/gate0_report.json"
    family_counts = {
        "lag": len(LAG_COLUMNS) * len(LAGS),
        "rolling": len(WINDOWS) * 3,
        "change_and_slope": 2,
        "cyclical": 6,
        "wind_vector": 2,
        "near_field_and_availability": len(NEAR_COLUMNS),
        "far_field_background": len(FAR_COLUMNS),
        "audit_only": len(AUDIT_COLUMNS),
    }
    assert sum(family_counts.values()) == len(extra)
    timestamp = local_time(out.timestamp)
    gaps = (out.assign(_time=timestamp).sort_values(["station_id", "_time"])
            .groupby("station_id", observed=True)._time.diff().dropna())
    near_by_split, legacy_by_split = {}, {}
    for split, rows in df.groupby("split", observed=True):
        near_by_split[str(split)] = spatial_split_diagnostics(rows, args.sigma_km, args.wind_speed_unit)
        legacy_by_split[str(split)] = spatial_split_diagnostics(rows, 15., args.wind_speed_unit, radius_km=None)
    train = df.loc[df.split.eq("train")]
    sensitivity = {}
    for sigma in (50., 80.):
        check = spatial_training_diagnostics(train, sigma, args.wind_speed_unit)
        sensitivity[str(int(sigma))] = {"overall": check["overall"],
                                        "coordinate_pairs_within_sigma": check["coordinate_pairs_within_sigma"],
                                        "receptors": {str(sid): check["by_node"][str(sid)] for sid in
                                                      out.loc[rec, "station_id"].unique() if str(sid) in check["by_node"]}}
    predictor_cols = [c for c in extra if c not in AUDIT_COLUMNS and c not in NEAR_DIAGNOSTIC_COLUMNS]
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "git_commit": git_revision(),
        "source_sha256": digest(Path(__file__)),
        "input_sha256": before, **storage,
        "rows": len(out), "input_columns": len(df.columns), "output_columns": len(out.columns),
        "new_columns": extra, "new_column_count": len(extra), "new_predictor_count": len(predictor_cols),
        "new_diagnostic_columns": NEAR_DIAGNOSTIC_COLUMNS,
        "feature_family_counts": family_counts,
        "integrity": {"node_count": int(out.station_id.nunique()),
                      "duplicate_primary_keys": int(out.duplicated(["station_id", "timestamp"]).sum()),
                      "original_columns_values_dtypes_and_row_order_preserved": True,
                      "timestamp_min": timestamp.min().isoformat(),
                      "timestamp_max": timestamp.max().isoformat(),
                      "non_hourly_consecutive_intervals": int(gaps.ne(pd.Timedelta(hours=1)).sum()),
                      "split_counts": {str(k): int(v) for k, v in out.split.value_counts().items()},
                      "infinite_new_predictor_values": int(np.isinf(out[predictor_cols]).sum().sum())},
        "model_feature_columns": [c for c in ["pm2_5", "pm10", "no2", "so2", "o3", "co",
                                  "temperature_2m", "relative_humidity_2m", "wind_speed_ms", "precipitation",
                                  "lat", "lon"] if c in df] + predictor_cols,
        "model_exclusions": ["target_*", "split", "station_id", "role", "region", "timestamp", "audit_*",
                             "wind_direction_10m", "wind_speed_kmh", "b_proto_*", "*_anomaly", *NEAR_DIAGNOSTIC_COLUMNS],
        "nan_counts": {c: int(out[c].isna().sum()) for c in extra},
        "nan_fractions": {c: float(out[c].isna().mean()) for c in extra},
        "parameters": {"sigma_km": args.sigma_km, "radius_km": NEAR_RADIUS_KM, "near_lags": NEAR_LAGS,
                       "far_windows": FAR_WINDOWS, "far_lag": FAR_LAG, "far_regions": FAR_REGIONS,
                       "calm_ms": CALM_MS, "wind_unit": unit,
                       "lag_hours": LAGS, "rolling_hours": WINDOWS, "rolling_ddof": 1,
                       "rolling_includes_current": True, "rolling_variable": "pm2_5",
                       "minimum_weight_ms": WEIGHT_EPS},
        "gate0_disclosed": json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else None,
        "eligibility_by_receptor_split": {
            str(s): {"total": len(g), "eligible": int(g.audit_model_eligible.sum())}
            for s, g in out.loc[rec].groupby("split", observed=True)},
        "flags_false": {c: int((~out[c]).sum()) for c in extra if out[c].dtype == bool},
        "spatial_coverage_by_receptor": {
            str(s): {f"advection_{k}h_fraction": float(g[f"upwind_pm25_advection_{k}h"].notna().mean())
                     for k in NEAR_LAGS} for s, g in out.loc[rec].groupby("station_id", observed=True)},
        "all_receptor_spatial_missing_fractions": {
            str(split): {c: float(g[c].isna().mean()) for c in NEAR_COLUMNS + FAR_COLUMNS}
            for split, g in out.loc[rec].groupby("split", observed=True)},
        "eligible_receptor_spatial_missing_fractions": {
            str(split): {c: float(g[c].isna().mean()) for c in NEAR_COLUMNS + FAR_COLUMNS}
            for split, g in out.loc[rec & out.audit_model_eligible].groupby("split", observed=True)},
        "spatial_training_diagnostics": near_by_split["train"],
        "legacy_spatial_training_diagnostics": legacy_by_split["train"],
        "spatial_diagnostics_all_rows": {"near_field": combine_spatial_support(near_by_split),
                                         "legacy_sigma15": combine_spatial_support(legacy_by_split)},
        "spatial_sensitivity_training": sensitivity,
        "far_field_background": far_field_summary(out),
        "spatial_column_checks": spatial_column_checks(out),
        "limitations": ["CAMS grid estimates, not ground observations; weather archive is retrospective.",
                         "The legacy 15 km calculation is included as a coverage comparison only.",
                         "Original interpolated values and fitted baselines remain upstream-owned.",
                         "Use eligibility flags; refit upstream transforms within each CV/LOSO fold.",
                         "Weights encode a directional association, not travel time or source attribution."]}
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(out):,} x {len(out.columns)}; {storage['output_bytes'] / 1e6:.2f} MB; input preserved")


if __name__ == "__main__":
    main()
