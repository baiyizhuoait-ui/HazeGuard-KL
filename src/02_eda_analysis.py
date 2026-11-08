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
spec = importlib.util.spec_from_file_location("agent2_features", ROOT / "src/02_spatiotemporal_features.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
COLORS = ["#167d9a", "#dc7732", "#7356a5", "#4c956c", "#c55867", "#656b75", "#c0a236", "#4c5ac6"]
LEVEL_COLORS = ["#31858c", "#deb349", "#de8546", "#af4355"]


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


def class_distribution(df, directory):
    rec = receptor_rows(df)
    counts = rec.target_level_3h.value_counts().reindex(range(4), fill_value=0)
    proportions = counts / counts.sum()
    all_counts = pd.crosstab(df.station_id, df.target_level_3h).reindex(columns=range(4), fill_value=0)
    all_prop = all_counts.div(all_counts.sum(axis=1), axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.8), gridspec_kw={"width_ratios": [1, 1.7]}, layout="constrained")
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
    fig.suptitle("All splits · descriptive target distribution; no feature or threshold selection", fontweight="bold", fontsize=12)
    save(fig, directory, "class_imbalance")
    return {"scope": "all splits, receptor total and all-node context; descriptive only",
            "receptor_counts": {str(k): int(v) for k, v in counts.items()},
            "receptor_proportions": {str(k): float(v) for k, v in proportions.items()},
            "node_proportions": {str(s): {str(k): float(v) for k, v in row.items()} for s, row in all_prop.iterrows()},
            "node_counts": {str(s): {str(k): int(v) for k, v in row.items()} for s, row in all_counts.iterrows()}}


def wind_rose(train, directory):
    require_training(train)
    rec = receptor_rows(train).copy()
    speed = F.wind_speed_ms(rec)
    valid = speed.ge(F.CALM_MS) & rec.wind_direction_10m.notna() & rec.pm2_5.notna()
    if "qc_calm_wind" in rec:
        valid &= rec.qc_calm_wind.eq(0)
    data = rec.loc[valid]
    sector = np.floor(((data.wind_direction_10m % 360 + 11.25) % 360) / 22.5).astype(int)
    bins = [F.CALM_MS, 2, 4, 6, np.inf]
    speed_bin = pd.cut(speed[valid], bins=bins, right=False, labels=False)
    counts = pd.crosstab(sector, speed_bin).reindex(index=range(16), columns=range(4), fill_value=0)
    freq = counts / len(data) * 100 if len(data) else counts.astype(float)
    means = data.pm2_5.groupby(sector).mean().reindex(range(16))
    fig = plt.figure(figsize=(12, 5.5), layout="constrained")
    axes = [fig.add_subplot(121, projection="polar"), fig.add_subplot(122, projection="polar")]
    angles = np.arange(16) * 2 * np.pi / 16
    bottom = np.zeros(16)
    for i, color in enumerate(["#d7e9ed", "#93c6d0", "#4c9fb1", "#17627b"]):
        axes[0].bar(angles, freq[i], bottom=bottom, width=2*np.pi/16*.9, color=color,
                    edgecolor="white", linewidth=.4, label=["0.5–2", "2–4", "4–6", "≥6"][i]+" m/s")
        bottom += freq[i].to_numpy()
    axes[0].legend(loc="lower left", bbox_to_anchor=(-.25, -.16), ncol=2, fontsize=8)
    axes[0].set_title("(a) Direction frequency by wind speed", pad=20)
    finite = means.dropna()
    norm = Normalize(vmin=float(finite.min()) if len(finite) else 0, vmax=float(finite.max()) if len(finite) else 1)
    axes[1].bar(angles, freq.sum(axis=1), width=2*np.pi/16*.9,
                color=plt.cm.YlOrRd(norm(means.fillna(0))), edgecolor="white", linewidth=.4)
    axes[1].set_title("(b) Frequency, colored by mean PM2.5", pad=20)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="YlOrRd"), ax=axes[1], shrink=.65, pad=.1, label="PM2.5 (µg/m³)")
    for ax in axes:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.arange(0, 360, 45), ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
        ax.set_rlabel_position(70)
    fig.suptitle("Training receptors · wind FROM direction · radial scale: non-calm hours (%)", fontsize=12, fontweight="bold")
    save(fig, directory, "wind_rose_pollution")
    return {"scope": "training receptors; calm/flagged hours excluded", "n_noncalm": len(data),
            "n_excluded_calm_or_missing": len(rec)-len(data), "sector_center_degrees": (np.arange(16)*22.5).tolist(),
            "sector_mean_pm25": [None if pd.isna(x) else float(x) for x in means],
            "sector_frequency_percent": freq.sum(axis=1).tolist(), "speed_bins_ms": ["0.5–2", "2–4", "4–6", ">=6"]}


def lag_correlations(source, destination, lags=range(-12, 13)):
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


def crosscorr(df, directory):
    available = set(df.station_id)
    # No duplicated aliases: Banting shares Sepang's CAMS cell in the current registry.
    pairs = [(a, b) for a in ("Port_Klang", "Banting") for b in ("Cheras_KL", "XMUM_Sepang")
             if a in available and b in available]
    requested = ["Port_Klang", "Banting", "Cheras_KL", "XMUM_Sepang"]
    mapping = {"requested_nodes": requested,
               "missing_nodes": [s for s in requested if s not in available],
               "note": "Available alternatives are comparisons, not verified aliases of missing coordinates."}
    if not pairs:
        pairs = [("Port_Klang", b) for b in ("KL_City", "Sepang") if {"Port_Klang", b} <= available]
    mapping["used_pairs"] = [list(p) for p in pairs]
    df = df.copy()
    df["_residual"] = deseasonalized(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
    records = []
    for i, (a, b) in enumerate(pairs):
        for ax, column, label in zip(axes, ("pm2_5", "_residual"), ("raw", "deseasonalized")):
            src = df[df.station_id.eq(a)].set_index("_time")[column]
            dst = df[df.station_id.eq(b)].set_index("_time")[column]
            points = lag_correlations(src, dst)
            finite = [p for p in points if p["r"] is not None]
            peak = max(finite, key=lambda x: x["r"]) if finite else None
            ax.plot([p["lag_h"] for p in points], [np.nan if p["r"] is None else p["r"] for p in points],
                    lw=2, color=COLORS[i], label=f"{a} → {b}")
            if peak:
                ax.scatter(peak["lag_h"], peak["r"], color=COLORS[i], zorder=3)
                ax.annotate(f"peak {peak['lag_h']:+d}h", (peak["lag_h"], peak["r"]),
                            xytext=(7, -18-i*12), textcoords="offset points", fontsize=8, color=COLORS[i])
            records.append({"source": a, "destination": b, "series": label, "points": points, "peak": peak})
    for ax, title in zip(axes, ("(a) Raw PM2.5", "(b) Training-climatology residuals")):
        ax.axvline(0, color="#777777", ls="--", lw=.8)
        ax.set(xlim=(-12, 12), xticks=range(-12, 13, 3), xlabel="Lag (hours); positive = coastal cell leads", title=title)
        ax.legend(fontsize=8, loc="lower center")
    axes[0].set_ylabel("Pearson cross-correlation")
    if not pairs:
        axes[0].text(.5, .5, "Required independent cell pairs unavailable", transform=axes[0].transAxes, ha="center")
    fig.suptitle("All splits · descriptive lead–lag association; no causal transport inference", fontweight="bold", fontsize=12)
    save(fig, directory, "spatial_crosscorr")
    return {"lag_definition": "corr(source(t-k), destination(t)); positive k = source leads",
            "mapping": mapping, "records": records, "residual_fit": "training month/hour means only"}


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
    corr_train = df.loc[df.split.eq("train") & F.clean_rows(df) & audited.agent2_target_safe_3h]
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
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print("Wrote five 300 dpi figures and reproducible EDA statistics")


if __name__ == "__main__":
    main()
