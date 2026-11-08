"""Agent 2 features, compatible with the current panel and the v2.0 input.

All histories use exact hourly timestamps within (station_id, split). Rolling
windows end at issue time t and include t. No interpolation, backfill, learned
imputation or resplitting is performed. Original columns and row order survive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAG_COLUMNS = ("pm2_5", "pm10", "no2", "relative_humidity_2m")
LAGS = (1, 2, 3, 6, 12, 24)
WINDOWS = (3, 6, 12)
TIMEZONE = "Asia/Kuala_Lumpur"
CALM_MS = 0.5
WEIGHT_EPS = 1e-12  # m/s; avoid normalizing effectively zero transport strength


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   text=True).strip()


def default_input():
    current = ROOT / "data/haze_panel_clean.parquet"
    return current if current.exists() else ROOT / "data/kl_8stations_clean.parquet"


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


def spatial_features(work, sigma_km):
    if not np.isfinite(sigma_km) or sigma_km <= 0:
        raise ValueError("sigma_km must be positive and finite")
    nodes = work.groupby("station_id", observed=True)[["lat", "lon"]].first().sort_index()
    distance, east, north = geometry(nodes.to_numpy())
    kernel = np.exp(-0.5 * (distance / sigma_km) ** 2)
    # Self/co-located points cannot be independent upstream contributors.
    kernel[distance < 1e-6] = 0
    names = ["upwind_pm25_advection_1h", "upwind_pm25_advection_2h",
             "spatial_pm25_gradient", "upwind_weight_sum_1h", "upwind_weight_sum_2h",
             "upwind_neighbor_count_1h", "upwind_neighbor_count_2h"]
    out = pd.DataFrame(np.nan, index=work.index, columns=names)
    # Each split gets its own time grid. Source data must also belong to this split.
    for _, sub in work.groupby("split", observed=True, sort=False):
        clock = pd.date_range(sub._time.min(), sub._time.max(), freq="h")
        pivot = lambda c: sub.pivot(index="_time", columns="station_id", values=c).reindex(index=clock, columns=nodes.index)
        pm = pivot("_safe_pm25").to_numpy()
        u, v = pivot("u_wind").to_numpy(), pivot("v_wind").to_numpy()
        for i, sid in enumerate(nodes.index):
            idx = sub.index[sub.station_id.eq(sid)]
            if not len(idx):
                continue
            projection = u * east[:, i] + v * north[:, i]
            # Remove floating-point cos(pi/2) noise from exactly perpendicular winds.
            weights = np.where(projection > 1e-12, projection, 0) * kernel[:, i]
            valid = np.isfinite(pm) & np.isfinite(weights) & (weights > WEIGHT_EPS)
            weights = np.where(valid, weights, 0)
            denom = weights.sum(axis=1)
            numerator = (np.where(valid, pm, 0) * weights).sum(axis=1)
            aggregate = np.divide(numerator, denom, out=np.full(len(clock), np.nan), where=denom > 0)
            series = pd.Series(aggregate, index=clock)
            for k in (1, 2):
                t = work.loc[idx, "_time"]
                out.loc[idx, f"upwind_pm25_advection_{k}h"] = series.shift(k).reindex(t).to_numpy()
                out.loc[idx, f"upwind_weight_sum_{k}h"] = pd.Series(denom, index=clock).shift(k).reindex(t).to_numpy()
                out.loc[idx, f"upwind_neighbor_count_{k}h"] = pd.Series(valid.sum(axis=1), index=clock).shift(k).reindex(t).to_numpy()
            # Same-time target minus same-time weighted neighbors; all available at t by assumption.
            out.loc[idx, "spatial_pm25_gradient"] = (work.loc[idx, "_safe_pm25"].to_numpy()
                                                       - series.reindex(work.loc[idx, "_time"]).to_numpy())
    return out


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
    return pd.DataFrame({"agent2_target_within_split_3h": same,
                         "agent2_target_aligned_3h": aligned,
                         "agent2_target_safe_3h": safe}, index=work.index)


def build_features(df, sigma_km=15.0, raw_wind_unit=None):
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
    added = pd.concat([added, spatial_features(work, sigma_km), label_audit(work)], axis=1)
    added["agent2_current_source_clean"] = clean_rows(work)
    added["agent2_history_complete"] = temporal.notna().all(axis=1)
    added["agent2_model_eligible"] = (added.agent2_target_safe_3h & added.agent2_current_source_clean
                                      & added.agent2_history_complete)
    overlap = set(df.columns) & set(added.columns)
    if overlap:
        raise ValueError(f"Refusing to overwrite existing columns: {sorted(overlap)}")
    # Single precision for added continuous features only; original values/dtypes stay exact.
    for c in added.select_dtypes(include="floating"):
        added[c] = added[c].astype("float32")
    added.index = df.index
    return pd.concat([df.copy(), added], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=default_input())
    ap.add_argument("--output", type=Path, default=ROOT / "data/feat_spatiotemporal.parquet")
    ap.add_argument("--audit", type=Path, default=ROOT / "models/metrics_stage2_features.json")
    ap.add_argument("--sigma-km", type=float, default=15.0)
    ap.add_argument("--wind-speed-unit", choices=["km/h", "m/s"])
    args = ap.parse_args()
    if args.input.resolve() == args.output.resolve():
        ap.error("input is read-only; output must be a different path")
    before = digest(args.input)
    df = pd.read_parquet(args.input)
    out = build_features(df, args.sigma_km, args.wind_speed_unit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # ZSTD is lossless; new continuous features use the declared float32 precision.
    out.to_parquet(args.output, index=False, compression="zstd", compression_level=15,
                   row_group_size=40000)
    pd.testing.assert_frame_equal(pd.read_parquet(args.output)[df.columns], df)
    assert before == digest(args.input), "input changed during feature construction"
    extra = [c for c in out if c not in df]
    rec = out.role.eq("receptor") if "role" in out else np.ones(len(out), dtype=bool)
    unit = ("wind_speed_ms (m/s), cross-checked against km/h when present" if "wind_speed_ms" in df
            else f"km/h converted once, or explicit raw unit {args.wind_speed_unit}")
    gate_path = ROOT / "models/gate0_report.json"
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "git_commit": git_revision(),
        "source_sha256": digest(Path(__file__)),
        "input_sha256": before, "output_sha256": digest(args.output),
        "rows": len(out), "input_columns": len(df.columns), "output_columns": len(out.columns),
        "output_bytes": args.output.stat().st_size, "new_columns": extra,
        "model_feature_columns": [c for c in ["pm2_5", "pm10", "no2", "so2", "o3", "co",
                                  "temperature_2m", "relative_humidity_2m", "wind_speed_ms", "precipitation",
                                  "lat", "lon"] if c in df] + [c for c in extra if not c.startswith("agent2_")],
        "model_exclusions": ["target_*", "split", "station_id", "role", "region", "timestamp", "agent2_*",
                             "wind_direction_10m", "wind_speed_kmh", "b_proto_*", "*_anomaly"],
        "nan_counts": {c: int(out[c].isna().sum()) for c in extra},
        "parameters": {"sigma_km": args.sigma_km, "calm_ms": CALM_MS, "wind_unit": unit,
                       "lag_hours": LAGS, "rolling_hours": WINDOWS, "rolling_ddof": 1,
                       "rolling_includes_current": True, "rolling_variable": "pm2_5",
                       "minimum_weight_ms": WEIGHT_EPS},
        "gate0_disclosed": json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else None,
        "eligibility_by_receptor_split": {
            str(s): {"total": len(g), "eligible": int(g.agent2_model_eligible.sum())}
            for s, g in out.loc[rec].groupby("split", observed=True)},
        "flags_false": {c: int((~out[c]).sum()) for c in extra if out[c].dtype == bool},
        "spatial_coverage_by_receptor": {
            str(s): {f"advection_{k}h_fraction": float(g[f"upwind_pm25_advection_{k}h"].notna().mean())
                     for k in (1, 2)} for s, g in out.loc[rec].groupby("station_id", observed=True)},
        "limitations": ["CAMS grid estimates, not ground observations; weather archive is retrospective.",
                         "15 km is retained from v2.0 and is narrow relative to the expanded network.",
                         "Original interpolated values and fitted baselines remain upstream-owned.",
                         "Use eligibility flags; refit upstream transforms within each CV/LOSO fold.",
                         "Weights encode a directional association, not travel time or source attribution."]}
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(out):,} x {len(out.columns)}; {args.output.stat().st_size / 1e6:.2f} MB; input preserved")


if __name__ == "__main__":
    main()
