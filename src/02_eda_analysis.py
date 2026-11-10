"""Five reproducible 300 dpi EDA figures; test-period plots are descriptive only."""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("spatiotemporal_features", ROOT / "src/02_spatiotemporal_features.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
COLORS = ["#167d9a", "#dc7732", "#7356a5", "#4c956c", "#c55867", "#656b75", "#c0a236", "#4c5ac6"]
LEVEL_COLORS = ["#31858c", "#deb349", "#de8546", "#af4355"]
CCF_LAG_HOURS = tuple(range(-12, 13))
MAX_EDA_METRICS_BYTES = 100_000


def require_training(df):
    """Descriptive input/target associations must not use held-out samples."""
    if df.empty or not df.split.eq("train").all():
        raise ValueError("training summaries require nonempty train-only input")


def save(fig, directory, slug):
    fig.savefig(directory / f"fig_eda_{slug}.png", dpi=300, facecolor="white")
    plt.close(fig)


def block_diurnal_ci(daily, seed=42, n_boot=400, block_days=7):
    """Circular moving-block bootstrap of dates, preserving all 24 hours together."""
    x = np.asarray(daily, float)
    if len(x) < 2:
        return np.full((2, x.shape[1]), np.nan)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        starts = rng.integers(0, len(x), size=int(np.ceil(len(x) / block_days)))
        idx = ((starts[:, None] + np.arange(block_days)) % len(x)).ravel()[:len(x)]
        boot.append(np.nanmean(x[idx], axis=0))
    return np.nanpercentile(boot, [2.5, 97.5], axis=0)


def receptor_rows(df):
    return df[df.role.eq("receptor")] if "role" in df else df


def diurnal_seasonal(train, directory):
    require_training(train)
    rec = receptor_rows(train)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    peaks, nodes = {}, {}
    for i, (sid, sub) in enumerate(rec.groupby("station_id", observed=True)):
        daily = sub.pivot_table(index=sub._time.dt.date, columns=sub._time.dt.hour, values="pm2_5", aggfunc="mean").reindex(columns=range(24))
        daily.index = pd.to_datetime(daily.index)
        daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
        mean = daily.mean().to_numpy()
        lo, hi = block_diurnal_ci(daily)
        color = COLORS[i % len(COLORS)]
        axes[0].plot(range(24), mean, label=sid, color=color, lw=1.8)
        axes[0].fill_between(range(24), lo, hi, color=color, alpha=.13)
        candidates = [h for h in range(24) if mean[h] > mean[(h-1) % 24] and mean[h] > mean[(h+1) % 24]]
        peaks[sid] = candidates
        nodes[sid] = {"mean_by_hour": mean.tolist(), "ci_low": lo.tolist(), "ci_high": hi.tolist(),
                      "days": len(daily), "local_maxima_hours": candidates}
        for h in candidates:
            axes[0].scatter(h, mean[h], color=color, s=18, zorder=3)
    axes[0].set(xlabel="Local hour (UTC+8)", ylabel="PM2.5 (µg/m³)", xticks=[0, 4, 8, 12, 16, 20, 23],
                title="(a) Diurnal means and 95% block-bootstrap bands")
    axes[0].legend(fontsize=8, loc="best")
    month = rec._time.dt.month
    seasons = {"SW: Jun–Sep": rec.loc[month.isin([6, 7, 8, 9]), "pm2_5"],
               "NE: Nov–Mar": rec.loc[month.isin([11, 12, 1, 2, 3]), "pm2_5"]}
    arrays = [x.dropna().to_numpy() for x in seasons.values()]
    axes[1].boxplot(arrays, tick_labels=list(seasons), showfliers=False,
                    patch_artist=True, boxprops={"facecolor": "#c4dce2"}, medianprops={"color": "#9c482b"})
    axes[1].set(ylabel="PM2.5 (µg/m³)", title="(b) Receptor distribution by monsoon period")
    fig.suptitle("Training period · CAMS grid-cell estimates", fontsize=13, fontweight="bold")
    save(fig, directory, "diurnal_seasonal")
    return {"scope": "training receptor rows with interpolation/missing flags excluded",
            "bootstrap": {"seed": 42, "replicates": 400, "block_days": 7}, "nodes": nodes,
            "seasons": {k: {"n": len(v), "median": float(v.median()), "q25": float(v.quantile(.25)),
                             "q75": float(v.quantile(.75))} for k, v in seasons.items()}}


def correlations(train, directory):
    require_training(train)
    rec = receptor_rows(train).copy()
    u, v = F.wind_vectors(rec)
    rec["u_wind"], rec["v_wind"] = u, v
    columns = [c for c in ["pm2_5", "pm10", "no2", "so2", "o3", "co", "temperature_2m",
                           "relative_humidity_2m", "wind_speed_ms", "precipitation", "u_wind", "v_wind", "target_pm25_3h"] if c in rec]
    labels = {"temperature_2m": "Temperature", "relative_humidity_2m": "Humidity", "wind_speed_ms": "Wind speed",
              "target_pm25_3h": "PM2.5 t+3h", "precipitation": "Rain", "pm2_5": "PM2.5", "pm10": "PM10",
              "no2": "NO2", "so2": "SO2", "o3": "O3", "co": "CO", "u_wind": "Wind east", "v_wind": "Wind north"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    result = {"n_rows": len(rec), "columns": columns, "scope": "training receptors, exact t+3h safe labels only"}
    for method, ax in zip(("pearson", "spearman"), axes):
        corr = rec[columns].corr(method=method)
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set(xticks=range(len(columns)), yticks=range(len(columns)),
               xticklabels=[labels[c] for c in columns], yticklabels=[labels[c] for c in columns], title=method.title())
        ax.tick_params(axis="both", labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=55, ha="right")
        for i in range(len(columns)):
            for j in range(len(columns)):
                r = corr.iloc[i, j]
                if abs(r) > .3:
                    ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(r) > .65 else "#222222")
        result[method] = {c: {d: (None if pd.isna(r) else float(r)) for d, r in corr[c].items()} for c in columns}
    fig.colorbar(im, ax=axes, shrink=.75, label="Correlation coefficient")
    fig.suptitle("Training receptors · Pearson and Spearman associations", fontweight="bold")
    save(fig, directory, "correlation_matrix")
    return result


def risk_summary(df):
    """Observed target counts; missing classes retain their ordered zero counts."""
    counts = df.target_level_3h.value_counts().reindex(range(4), fill_value=0)
    total = int(counts.sum())
    return {"n_labeled": total, "counts": {str(k): int(v) for k, v in counts.items()},
            "proportions": {str(k): float(v / total) if total else None for k, v in counts.items()}}


def class_time_summary(df):
    rec = receptor_rows(df)
    years = pd.to_datetime(rec._time).dt.year
    return {"by_split": {s: risk_summary(rec.loc[rec.split.eq(s)]) for s in ("train", "val", "test")},
            "by_year": {str(y): risk_summary(rec.loc[years.eq(y)]) for y in sorted(years.unique())}}


def class_distribution(df, directory):
    rec = receptor_rows(df)
    counts = rec.target_level_3h.value_counts().reindex(range(4), fill_value=0)
    proportions = counts / counts.sum()
    all_counts = pd.crosstab(df.station_id, df.target_level_3h).reindex(columns=range(4), fill_value=0)
    all_prop = all_counts.div(all_counts.sum(axis=1), axis=0)
    time_summary = class_time_summary(df)
    fig = plt.figure(figsize=(15, 10), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.3])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[:, 1]), fig.add_subplot(grid[1, 0])]
    bars = axes[0].bar([f"Level {i}" for i in range(4)], proportions, color=LEVEL_COLORS)
    axes[0].set(yscale="log", ylabel="Fraction of receptor rows (log scale)", title="(a) Receptor target risk levels")
    for bar, n, p in zip(bars, counts, proportions):
        if n:
            axes[0].text(bar.get_x() + bar.get_width()/2, p*1.04, f"{p:.1%}\nn={n:,}", ha="center", fontsize=9)
        else:
            axes[0].text(bar.get_x() + bar.get_width()/2, .001, "n=0", ha="center", fontsize=9)
    axes[0].set_ylim(max(.0001, proportions[proportions > 0].min() / 2), 1)
    left = np.zeros(len(all_prop))
    for i in range(4):
        axes[1].barh(all_prop.index, all_prop[i], left=left, color=LEVEL_COLORS[i], label=f"Level {i}")
        left += all_prop[i].to_numpy()
    axes[1].tick_params(axis="y", labelsize=8)
    axes[1].set(xlabel="Fraction within each grid cell", xlim=(0, 1), title="(b) All 26 cells (sources are context only)" if len(all_prop)==26 else "(b) All cells")
    axes[1].invert_yaxis()
    axes[1].legend(ncol=4, fontsize=8, loc="lower right")
    time_items = [(s, v) for s, v in time_summary["by_split"].items()]
    time_items += [(y, v) for y, v in time_summary["by_year"].items()]
    bottom = np.zeros(len(time_items))
    for level in range(4):
        heights = [v["proportions"][str(level)] or 0 for _, v in time_items]
        axes[2].bar(range(len(time_items)), heights, bottom=bottom,
                    color=LEVEL_COLORS[level], label=f"Level {level}")
        bottom += heights
    axes[2].axvline(2.5, color="#777777", ls=":", lw=1)
    axes[2].set(xticks=range(len(time_items)), xticklabels=[s for s, _ in time_items], ylim=(0, 1.08),
                ylabel="Fraction within interval", title="(c) Receptors: split and calendar-year shifts")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].legend(ncol=4, fontsize=8, loc="upper center")
    fig.suptitle("All splits · descriptive target distribution; no feature or threshold selection", fontweight="bold", fontsize=12)
    save(fig, directory, "class_imbalance")
    return {"scope": "all splits, receptor total and all-node context; descriptive only",
            "receptor_counts": {str(k): int(v) for k, v in counts.items()},
            "receptor_proportions": {str(k): float(v) for k, v in proportions.items()},
            "node_proportions": {str(s): {str(k): float(v) for k, v in row.items()} for s, row in all_prop.iterrows()},
            "node_counts": {str(s): {str(k): int(v) for k, v in row.items()} for s, row in all_counts.iterrows()},
            "receptor_by_split": time_summary["by_split"], "receptor_by_year": time_summary["by_year"],
            "risk_definition": "Project experimental levels [<15, 15-<35, 35-<55, >=55]; not official hourly AQI."}


def wind_summary(frame):
    """Non-calm, quality-checked wind-FROM sectors on an explicit input period."""
    speed = F.wind_speed_ms(frame)
    valid = (speed.ge(F.CALM_MS) & frame.wind_direction_10m.notna()
             & frame.pm2_5.notna() & F.clean_rows(frame))
    if "qc_calm_wind" in frame:
        valid &= frame.qc_calm_wind.eq(0)
    data = frame.loc[valid]
    sector = np.floor(((data.wind_direction_10m % 360 + 11.25) % 360) / 22.5).astype(int)
    bins = [F.CALM_MS, 2, 4, 6, np.inf]
    speed_bin = pd.cut(speed[valid], bins=bins, right=False, labels=False)
    counts = pd.crosstab(sector, speed_bin).reindex(index=range(16), columns=range(4), fill_value=0)
    freq = counts / len(data) * 100 if len(data) else counts.astype(float)
    means = data.pm2_5.groupby(sector).mean().reindex(range(16))
    sw_flow = (data.wind_direction_10m % 360).ge(202.5) & (data.wind_direction_10m % 360).lt(247.5)
    return {"n_noncalm": len(data), "n_excluded_calm_or_missing": len(frame)-len(data),
            "sector_center_degrees": (np.arange(16)*22.5).tolist(),
            "sector_mean_pm25": [None if pd.isna(x) else float(x) for x in means],
            "sector_count": counts.sum(axis=1).astype(int).tolist(),
            "sector_frequency_percent": freq.sum(axis=1).tolist(),
            "speed_direction_counts": counts.astype(int).to_numpy().tolist(),
            "speed_direction_frequency_percent": freq.to_numpy().tolist(),
            "speed_bins_ms": ["0.5–2", "2–4", "4–6", ">=6"],
            "sw_flow_sector_degrees": [202.5, 247.5], "sw_flow_count": int(sw_flow.sum()),
            "sw_flow_fraction": float(sw_flow.mean()) if len(data) else None,
            "sw_flow_mean_pm25": float(data.loc[sw_flow, "pm2_5"].mean()) if sw_flow.any() else None}


def wind_rose(train, directory):
    require_training(train)
    rec = receptor_rows(train).copy()
    summary = wind_summary(rec)
    sw = rec.loc[rec._time.dt.month.isin([6, 7, 8, 9])]
    sw_summary = wind_summary(sw)
    nodes = {str(s): {"all_months": wind_summary(g),
                      "sw_monsoon": wind_summary(g.loc[g._time.dt.month.isin([6, 7, 8, 9])])}
             for s, g in train.groupby("station_id", observed=True, sort=True)}
    receptor_ids = sorted(rec.station_id.unique())
    # Retain one row per receptor; the current network has three receptor cells.
    extra_rows = int(np.ceil(len(receptor_ids) / 3))
    fig = plt.figure(figsize=(15, 5.2 * (1 + extra_rows)), layout="constrained")
    grid = fig.add_gridspec(1 + extra_rows, 3)
    axes = [fig.add_subplot(grid[0, i], projection="polar") for i in range(3)]
    angles = np.arange(16) * 2 * np.pi / 16
    bottom = np.zeros(16)
    for i, color in enumerate(["#d7e9ed", "#93c6d0", "#4c9fb1", "#17627b"]):
        heights = np.array(summary["speed_direction_frequency_percent"])[:, i]
        axes[0].bar(angles, heights, bottom=bottom, width=2*np.pi/16*.9, color=color,
                    edgecolor="white", linewidth=.4, label=["0.5–2", "2–4", "4–6", "≥6"][i]+" m/s")
        bottom += heights
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.08), ncol=2, fontsize=8)
    axes[0].set_title("All receptors: speed and direction", pad=20)
    panels = [(axes[1], summary, "All receptors: all months"),
              (axes[2], sw_summary, "All receptors: SW monsoon (Jun–Sep)")]
    for i, sid in enumerate(receptor_ids):
        ax = fig.add_subplot(grid[1 + i // 3, i % 3], projection="polar")
        axes.append(ax)
        panels.append((ax, nodes[str(sid)]["all_months"], f"{sid}: all training months"))
    finite = [v for _, stats, _ in panels for v in stats["sector_mean_pm25"] if v is not None]
    norm = Normalize(vmin=min(finite) if finite else 0, vmax=max(finite) if finite else 1)
    max_frequency = max([max(stats["sector_frequency_percent"]) for _, stats, _ in panels] + [1])
    for ax, stats, title in panels:
        means = np.array([np.nan if v is None else v for v in stats["sector_mean_pm25"]])
        ax.bar(angles, stats["sector_frequency_percent"], width=2*np.pi/16*.9,
               color=plt.cm.YlOrRd(norm(means)), edgecolor="white", linewidth=.4)
        ax.set_ylim(0, max_frequency * 1.05)
        ax.set_title(f"{title}\nn = {stats['n_noncalm']:,} non-calm hours", pad=20, fontsize=10)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="YlOrRd"), ax=[p[0] for p in panels],
                 shrink=.6, pad=.07, label="Sector mean PM2.5 (µg/m³), shared color scale")
    for ax in axes:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.arange(0, 360, 45), ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
        ax.set_rlabel_position(70)
    fig.suptitle("Training data · wind FROM direction · radial scale: non-calm hours (%)", fontsize=13, fontweight="bold")
    save(fig, directory, "wind_rose_pollution")
    return {"scope": "training only; receptor plots, all-node statistics; calm/flagged hours excluded",
            **summary, "sw_monsoon": sw_summary, "sw_monsoon_months": [6, 7, 8, 9], "by_node": nodes,
            "interpretation": "Calendar SW monsoon and wind-from SW sector are separate descriptive quantities; no source attribution."}


def lag_correlations(source, destination, lags=CCF_LAG_HOURS):
    """Positive k means source leads: corr(source(t-k), destination(t))."""
    clock = pd.date_range(min(source.index.min(), destination.index.min()),
                          max(source.index.max(), destination.index.max()), freq="h")
    a, b = source.reindex(clock), destination.reindex(clock)
    results = []
    for k in lags:
        paired = pd.concat([a.shift(k), b], axis=1).dropna()
        r = paired.iloc[:, 0].corr(paired.iloc[:, 1]) if len(paired) >= 3 else np.nan
        results.append({"lag_h": k, "r": None if pd.isna(r) else float(r), "n_pairs": len(paired)})
    return results


def deseasonalized(df):
    """Month/hour climatology fit to TRAIN only; used only for descriptive CCF."""
    train = df[df.split.eq("train")]
    keys = ["station_id", "_month", "_hour"]
    proto = train.groupby(keys, observed=True).pm2_5.mean().rename("_clim")
    lookup = pd.MultiIndex.from_frame(df[keys])
    return pd.Series(df.pm2_5.to_numpy() - proto.reindex(lookup).to_numpy(), index=df.index)


def geometry_pair_candidates(df, local_radius_km=150.0):
    """Predefine pairs using coordinates/roles alone, before reading pollution.

    For each receptor select its nearest other receptor and corridor within the
    fixed local radius, plus the nearest source from each registered source
    region. Distance, then node identifier, breaks ties. All network candidates
    are retained for auditing; correlations never rank or filter these pairs.
    """
    metadata = df[["station_id", "lat", "lon", "role", "region"]].drop_duplicates().copy()
    if metadata.station_id.duplicated().any():
        raise ValueError("node coordinates, roles or regions vary within the panel")
    metadata = metadata.sort_values("station_id").reset_index(drop=True)
    distance, _, _ = F.geometry(metadata[["lat", "lon"]].to_numpy())
    candidates = []
    for i, destination in metadata.iterrows():
        if destination.role != "receptor":
            continue
        for j, source in metadata.iterrows():
            if i == j or distance[j, i] < 1e-6:
                continue
            category = "regional_source" if source.role == "source" else f"local_{source.role}"
            eligible = source.role == "source" or distance[j, i] <= local_radius_km
            candidates.append({"source": str(source.station_id), "destination": str(destination.station_id),
                               "source_role": str(source.role), "source_region": str(source.region),
                               "distance_km": float(distance[j, i]), "category": category,
                               "eligible_by_geometry": bool(eligible)})
    groups = {}
    for pair in candidates:
        if not pair["eligible_by_geometry"]:
            continue
        key = (pair["destination"], pair["category"],
               pair["source_region"] if pair["source_role"] == "source" else "")
        groups.setdefault(key, []).append(pair)
    selected = [min(groups[k], key=lambda p: (p["distance_km"], p["source"])) for k in sorted(groups)]
    return {"rule": "Nearest other receptor and corridor within 150 km per receptor; nearest source per source region and receptor; distance then ID; no pollution ranking.",
            "local_radius_km": local_radius_km, "network_nodes": metadata.to_dict(orient="records"),
            "candidate_count": len(candidates), "candidates": candidates, "selected_pairs": selected}


def extended_training_peaks(train, pairs):
    """Check prespecified lag bands on the same geometry pairs using training hours."""
    require_training(train)
    lags = range(-48, 49)
    bands = {"local": (2, 4), "regional": (18, 36)}
    subset = train.loc[F.clean_rows(train), ["station_id", "split", "_time", "_month", "_hour", "pm2_5"]].copy()
    subset["_residual"] = deseasonalized(subset)
    by_grid = {str(grid): frame.set_index("_time")
               for grid, frame in subset.groupby("station_id", observed=True)}

    def peak_tuple(points):
        peak = max(points, key=lambda point: point["r"], default=None)
        return [peak["lag_h"], peak["r"], peak["n_pairs"]] if peak is not None else None

    records = []
    for index, pair in enumerate(pairs):
        group = "regional" if pair["category"] == "regional_source" else "local"
        lower, upper = bands[group]
        source, destination = by_grid.get(pair["source"]), by_grid.get(pair["destination"])
        for column, label in (("pm2_5", "raw"), ("_residual", "deseasonalized")):
            if source is None or destination is None:
                records.append([index, label, None, None])
                continue
            points = lag_correlations(source[column], destination[column], lags)
            finite = [point for point in points if point["r"] is not None]
            in_band = [point for point in finite if lower <= point["lag_h"] <= upper]
            records.append([index, label, peak_tuple(finite), peak_tuple(in_band)])

    return {"scope": "Clean train rows at both ends of each pair. Peak maximizes signed Pearson r. Positive lag means source leads.",
            "pair_index": "Zero-based position in pair_selection.selected_pairs.",
            "lag_range_hours": [lags.start, lags.stop - 1], "candidate_bands_hours": bands,
            "record_columns": ["pair_index", "series", "max_peak", "candidate_band_peak"],
            "peak_columns": ["lag_h", "r", "n_pairs"], "records": records}


def crosscorr(df, directory):
    selection = geometry_pair_candidates(df)
    pairs = selection["selected_pairs"]
    df = df.copy()
    df["_residual"] = deseasonalized(df)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True, layout="constrained")
    records = []
    color_positions = {"local": 0, "regional": 0}
    for pair in pairs:
        a, b = pair["source"], pair["destination"]
        group = "regional" if pair["category"] == "regional_source" else "local"
        row = 1 if group == "regional" else 0
        color = COLORS[color_positions[group] % len(COLORS)]
        color_positions[group] += 1
        for ax, column, label in zip(axes[row], ("pm2_5", "_residual"), ("raw", "deseasonalized")):
            src = df[df.station_id.eq(a)].set_index("_time")[column]
            dst = df[df.station_id.eq(b)].set_index("_time")[column]
            points = lag_correlations(src, dst)
            finite = [p for p in points if p["r"] is not None]
            peak = max(finite, key=lambda x: x["r"]) if finite else None
            peak_text = f"peak {peak['lag_h']:+d}h, r={peak['r']:.3f}" if peak else "no paired estimate"
            ax.plot([p["lag_h"] for p in points], [np.nan if p["r"] is None else p["r"] for p in points],
                    lw=1.7, color=color, label=f"{a} → {b} ({peak_text})")
            if peak:
                ax.scatter(peak["lag_h"], peak["r"], color=color, s=24, zorder=3)
            records.append({**pair, "series": label, "points": points, "peak": peak})
    for row, context in enumerate(("Local neighbors → receptors", "Regional source cells → receptors")):
        for col, series in enumerate(("Raw PM2.5", "Training-climatology residuals")):
            ax = axes[row, col]
            ax.axvline(0, color="#777777", ls="--", lw=.8)
            ax.set(xlim=(-12, 12), xticks=range(-12, 13, 3),
                   xlabel="Lag (hours); positive = source leads, negative = destination leads",
                   title=f"{context}\n{series}")
            if ax.lines and color_positions["regional" if row else "local"]:
                ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(.5, -.18), ncol=1)
            else:
                ax.text(.5, .5, "No pairs satisfy the geometry rule", transform=ax.transAxes, ha="center")
        axes[row, 0].set_ylabel("Pearson cross-correlation")
    fig.suptitle("26-cell network context · all splits, descriptive only\nPairs fixed by coordinates and roles; correlations do not establish physical transport", fontweight="bold", fontsize=13)
    save(fig, directory, "spatial_crosscorr")
    return {"lag_definition": "corr(source(t-k), destination(t)); positive k = source leads",
            "pair_selection": selection, "records": records, "residual_fit": "training month/hour means only",
            "extended_training_peaks": extended_training_peaks(df.loc[df.split.eq("train")], pairs),
            "interpretation": "Regional pairs provide network context; a ±12h correlation peak is not a validated transport time or an emissions attribution."}


def spatial_similarity(train):
    """Audit common-grid duplication on exactly timestamp-aligned training PM2.5."""
    require_training(train)
    values = train.pivot(index="_time", columns="station_id", values="pm2_5").sort_index()
    correlations = values.corr(method="pearson", min_periods=3)
    records = []
    for i, a in enumerate(values.columns):
        for b in values.columns[i + 1:]:
            overlap = values[[a, b]].dropna()
            r = correlations.loc[a, b]
            records.append({"node_a": str(a), "node_b": str(b), "n_pairs": len(overlap),
                            "pearson_r": None if pd.isna(r) else float(r),
                            "identical_on_overlap": bool(len(overlap) >= 3 and overlap[a].eq(overlap[b]).all()),
                            "complete_series_identical": bool(len(overlap) >= 3
                                and values[a].notna().equals(values[b].notna())
                                and overlap[a].eq(overlap[b]).all())})
    finite = [p for p in records if p["pearson_r"] is not None]
    return {"scope": "training rows only, quality flags excluded; exact hourly timestamp alignment",
            "nodes": len(values.columns), "pair_count": len(records),
            "exact_duplicate_pairs": [p for p in records if p["complete_series_identical"]],
            "high_correlation_threshold": .95,
            "high_correlation_pairs": [p for p in finite if abs(p["pearson_r"]) >= .95],
            "max_pair": max(finite, key=lambda p: abs(p["pearson_r"])) if finite else None,
            "pairs": records,
            "interpretation": "Distinct series do not establish independent ground observations or independent model errors."}


def compact_eda_metrics(metrics):
    """Store report statistics at their original precision within the JSON size budget."""
    result = {**metrics, "format_version": 2}
    wind = metrics["wind_rose"]
    node_fields = ("n_noncalm", "n_excluded_calm_or_missing", "sector_mean_pm25",
                   "speed_direction_counts", "sw_flow_count", "sw_flow_fraction", "sw_flow_mean_pm25")
    result["wind_rose"] = {
        **wind,
        "by_node": {node: {period: {field: stats[field] for field in node_fields}
                           for period, stats in periods.items()}
                    for node, periods in wind["by_node"].items()},
        "by_node_encoding": "Use the shared sector and speed bins. Sum each count row for the sector count. Percentages are counts times 100 divided by n_noncalm (zero if no observations).",
    }

    crosscorr_audit = metrics["crosscorr"]
    pair_selection = crosscorr_audit["pair_selection"]
    candidate_columns = ("source", "destination", "distance_km", "eligible_by_geometry")
    result["crosscorr"] = {
        **crosscorr_audit,
        "lag_hours": CCF_LAG_HOURS,
        "pair_selection": {
            **pair_selection,
            "candidate_columns": candidate_columns,
            "candidates": [[pair[column] for column in candidate_columns]
                           for pair in pair_selection["candidates"]],
        },
        "records": [{**{key: record[key] for key in ("source", "destination", "series", "peak")},
                     "points": {column: [point[column] for point in record["points"]]
                                for column in ("r", "n_pairs")}}
                    for record in crosscorr_audit["records"]],
        "records_encoding": "Point arrays follow lag_hours. Pair metadata are in pair_selection.selected_pairs.",
    }

    similarity = metrics["spatial_similarity"]
    result["spatial_similarity"] = {key: value for key, value in similarity.items() if key != "pairs"}
    result["spatial_similarity"]["identical_on_overlap_pairs"] = [
        pair for pair in similarity["pairs"] if pair["identical_on_overlap"]]
    result["spatial_similarity"]["pair_details"] = (
        "Full pair records can be recomputed with spatial_similarity on the clean training rows of the input panel.")
    return result


def write_eda_metrics(path, metrics):
    payload = json.dumps(compact_eda_metrics(metrics), ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8") + b"\n"
    if len(payload) > MAX_EDA_METRICS_BYTES:
        raise ValueError(f"EDA metrics require {len(payload):,} bytes. The limit is {MAX_EDA_METRICS_BYTES:,} bytes.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=F.default_input())
    ap.add_argument("--figures", type=Path, default=ROOT / "figures")
    ap.add_argument("--metrics", type=Path, default=ROOT / "models/metrics_stage2_eda.json")
    ap.add_argument("--wind-speed-unit", choices=["km/h", "m/s"])
    args = ap.parse_args()
    df = pd.read_parquet(args.input)
    # Derive in memory for legacy input, without renaming or writing the input.
    if "wind_speed_ms" not in df:
        df["wind_speed_ms"] = F.wind_speed_ms(df, args.wind_speed_unit)
    df["_time"] = F.validate_panel(df)
    df["_month"], df["_hour"] = df._time.dt.month, df._time.dt.hour
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlepad": 12, "savefig.dpi": 300})
    args.figures.mkdir(parents=True, exist_ok=True)
    clean = df.loc[F.clean_rows(df)].copy()
    train = clean[clean.split.eq("train")]
    audited = F.label_audit(df)
    corr_train = df.loc[df.split.eq("train") & F.clean_rows(df) & audited.audit_target_safe_3h]
    metrics = {"generated_at": datetime.now(timezone.utc).isoformat(), "git_commit": F.git_revision(),
               "source_sha256": {"eda": F.digest(Path(__file__)),
                                  "features": F.digest(ROOT / "src/02_spatiotemporal_features.py")},
               "input_sha256": F.digest(args.input), "rows": len(df), "nodes": df.station_id.nunique(),
               "figures_dpi": 300, "date_range": [str(df._time.min()), str(df._time.max())],
               "diurnal_seasonal": diurnal_seasonal(train, args.figures),
               "correlations": correlations(corr_train, args.figures),
               "class_distribution": class_distribution(df, args.figures),
               "wind_rose": wind_rose(train, args.figures),
               "crosscorr": crosscorr(clean, args.figures),
               "spatial_similarity": spatial_similarity(train)}
    metrics["figure_sha256"] = {
        f"fig_eda_{slug}.png": F.digest(args.figures / f"fig_eda_{slug}.png")
        for slug in ("diurnal_seasonal", "correlation_matrix", "class_imbalance",
                     "wind_rose_pollution", "spatial_crosscorr")}
    write_eda_metrics(args.metrics, metrics)
    print("Wrote five 300 dpi figures and reproducible EDA statistics")


if __name__ == "__main__":
    main()
