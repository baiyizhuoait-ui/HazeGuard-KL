#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HazeGuard-KL dashboard (module 07).

Run from the repository root:

    streamlit run app.py

Design rules
------------
1. Read-only. The app never trains a model, never searches a threshold on the
   test split and never writes a file. Every number on screen comes from a file
   under data/ or models/ that an upstream module has already saved.
2. One metric implementation. Anything that has to be recomputed (the
   validation-only threshold explorer) goes through src/utils_metrics.py.
3. The deployed decision rule is a single config line (DEFAULT_RULE below).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from config import (  # noqa: E402
    DEFAULT_COST_MATRIX,
    HORIZON_H,
    LEVEL_NAMES,
    LEVEL_THRESHOLDS,
    NODES,
)
from utils_metrics import evaluate_predictions  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA = ROOT / "data"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"

# Decision rules the dashboard can display. Each one maps to a column of saved
# predictions and to its row name in models/all_models_benchmark.csv.
RULES = {
    "M5_val_tuned": dict(label="M5 tuned cut points",
                         column="pred_m5_tuned", benchmark="M5_val_tuned"),
    "M5_fixed": dict(label="M5 fixed cut points",
                     column="pred_m5_fixed", benchmark="M5_fixed"),
    "M6": dict(label="M6 most probable level",
               column="pred_m6", benchmark="M6_Ordinal_RF"),
    "M7": dict(label="M7 cost-sensitive thresholds",
               column="pred_m7", benchmark="M7"),
}
# docs/spec_v3.md section 4.2 names M5 as the main model. Change this one line
# if the team decides to deploy M7 instead.
DEFAULT_RULE = "M5_val_tuned"

METRICS = {  # name -> True when higher is better
    "Macro_F1": True,
    "Balanced_Accuracy": True,
    "High_Risk_Recall": True,
    "MAOE": False,
    "Severe_Jumping_Error_Rate": False,
    "Avg_Asymmetric_Cost": False,
}

ADVICE = {
    0: "Normal outdoor activity.",
    1: "Sensitive groups should cut down long outdoor exertion.",
    2: "Reduce outdoor activity and keep windows closed.",
    3: "Avoid outdoor activity. Wear an N95 mask if you must go out.",
}

# Reserved status colours for the four ordered levels. They are always shown
# together with the level label, never as the only cue.
LEVEL_COLORS = ["#0ca30c", "#fab219", "#ec835a", "#d03b3b"]
INK, INK_2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
ROLE_MARKERS = {"receptor": "o", "corridor": "s", "source": "^"}


def level_label(level: int) -> str:
    return f"L{int(level)} {LEVEL_NAMES[int(level)]}"


# ---------------------------------------------------------------------------
# Loading (cached, read-only)
# ---------------------------------------------------------------------------
def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_thresholds(proba: np.ndarray, tau) -> np.ndarray:
    """Sequential ordinal decision on cumulative probabilities.

    Same rule as predict_with_thresholds in src/06_cost_thresholds.py, repeated
    here because that module name cannot be imported. self_check() below
    confirms this copy reproduces the saved validation cost.
    """
    proba = np.asarray(proba, dtype=float)
    q1, q2, q3 = proba[:, 1:].sum(axis=1), proba[:, 2:].sum(axis=1), proba[:, 3]
    pass1 = q1 >= tau[0]
    pass2 = pass1 & (q2 >= tau[1])
    pass3 = pass2 & (q3 >= tau[2])
    return pass1.astype(int) + pass2.astype(int) + pass3.astype(int)


@st.cache_data(show_spinner=False)
def load_thresholds() -> dict:
    return read_json(MODELS / "cost_thresholds.json")


@st.cache_data(show_spinner="Loading saved predictions")
def load_predictions() -> pd.DataFrame:
    """One row per (station, issue time) for the validation and test splits."""
    probs = np.load(MODELS / "ordinal_probs_val_test.npz")
    stage5 = np.load(MODELS / "stage5_predictions.npz")
    tau = load_thresholds()["thresholds"]
    frames = []
    for split in ("val", "test"):
        proba = probs[f"proba_{split}"]
        quant = stage5[f"quantiles_{split}"]
        n = len(proba)
        if len(quant) != n:
            raise ValueError(f"{split}: the two prediction files have different lengths")
        frame = pd.DataFrame({
            "split": split,
            "station_id": probs[f"station_id_{split}"].astype(str),
            "timestamp": pd.to_datetime(probs[f"ts_{split}"]),
            "y_true": probs[f"y_{split}"].astype(int),
            "p0": proba[:, 0], "p1": proba[:, 1], "p2": proba[:, 2], "p3": proba[:, 3],
            "reg": stage5[f"reg_{split}"],
            "q05": quant[:, 0], "q50": quant[:, 1], "q95": quant[:, 2],
            "pred_m5_fixed": stage5[f"pred_m5_fixed_{split}"].astype(int),
            "pred_m5_tuned": stage5[f"pred_m5_tuned_{split}"].astype(int),
            "pred_m6": stage5[f"pred_m6_{split}"].astype(int),
            "pred_m7": apply_thresholds(proba, tau),
        })
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["valid_time"] = out["timestamp"] + pd.Timedelta(hours=HORIZON_H)
    return out


def add_running_mean(station: pd.DataFrame) -> pd.DataFrame:
    """24 h running mean of observed PM2.5 for one station, sorted by time."""
    indexed = station.set_index("timestamp")["pm2_5"]
    mean = indexed.rolling("24h", min_periods=18).mean()
    return station.assign(pm2_5_mean_24h=mean.to_numpy())


@st.cache_data(show_spinner="Loading observations")
def load_observations() -> pd.DataFrame | None:
    """Current PM2.5 and the realised +3 h concentration for receptor rows."""
    path = DATA / "features_final.parquet"
    try:
        obs = pd.read_parquet(
            path, columns=["station_id", "timestamp", "pm2_5", "target_pm25_3h"])
    except Exception:  # missing file, missing parquet engine, renamed column
        return None
    obs["timestamp"] = _naive(obs["timestamp"])
    obs["station_id"] = obs["station_id"].astype(str)
    return obs


@st.cache_data(show_spinner="Loading the 26-node panel")
def load_panel() -> pd.DataFrame | None:
    path = DATA / "haze_panel_clean.parquet"
    try:
        panel = pd.read_parquet(path, columns=[
            "station_id", "timestamp", "pm2_5", "wind_speed_ms", "wind_direction_10m"])
    except Exception:
        return None
    panel["timestamp"] = _naive(panel["timestamp"])
    panel["station_id"] = panel["station_id"].astype(str)
    return panel


def _naive(series: pd.Series) -> pd.Series:
    series = pd.to_datetime(series)
    if getattr(series.dt, "tz", None) is not None:
        series = series.dt.tz_localize(None)
    return series


@st.cache_data(show_spinner=False)
def load_table(name: str) -> pd.DataFrame | None:
    path = MODELS / name
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_json_or_none(name: str) -> dict | None:
    path = MODELS / name
    return read_json(path) if path.exists() else None


def joined(preds: pd.DataFrame, obs: pd.DataFrame | None) -> pd.DataFrame:
    if obs is None:
        return preds.assign(pm2_5=np.nan, target_pm25_3h=np.nan)
    return preds.merge(obs, on=["station_id", "timestamp"], how="left")


def self_check(preds: pd.DataFrame) -> str | None:
    """Return a warning when this app's M7 rule disagrees with the saved cost."""
    saved = load_thresholds().get("validation_cost")
    val = preds[preds["split"] == "val"]
    mine = evaluate_predictions(val["y_true"], val["pred_m7"])["Avg_Asymmetric_Cost"]
    if saved is not None and abs(mine - float(saved)) > 5e-4:
        return (f"M7 check failed: this app gets validation cost {mine:.4f}, "
                f"models/cost_thresholds.json records {float(saved):.4f}.")
    return None


# ---------------------------------------------------------------------------
# Reliability note
# ---------------------------------------------------------------------------
def reliability_reference(preds: pd.DataFrame) -> dict:
    """Reference points taken from the validation split only."""
    val = preds[preds["split"] == "val"]
    top = val[["p0", "p1", "p2", "p3"]].max(axis=1)
    width = val["q95"] - val["q05"]
    return {"low_top": float(top.quantile(0.25)), "wide": float(width.quantile(0.75))}


def reliability_note(row: pd.Series, ref: dict) -> tuple[str, str]:
    top = float(max(row["p0"], row["p1"], row["p2"], row["p3"]))
    width = float(row["q95"] - row["q05"])
    crossed = [t for t in LEVEL_THRESHOLDS if row["q05"] < t <= row["q95"]]
    reasons = []
    if top < ref["low_top"]:
        reasons.append(f"the most probable level has only {top:.0%} probability")
    if width > ref["wide"]:
        reasons.append(f"the trees disagree by {width:.0f} µg/m³")
    if crossed:
        reasons.append("the tree spread crosses the "
                       + " and ".join(f"{t:g}" for t in crossed) + " µg/m³ boundary")
    if len(reasons) >= 2:
        return "Low confidence", "; ".join(reasons) + "."
    if reasons:
        return "Moderate confidence", reasons[0] + "."
    return "High confidence", (f"the most probable level has {top:.0%} probability and the "
                               f"tree spread ({width:.0f} µg/m³) stays inside one level.")


# ---------------------------------------------------------------------------
# Figures (matplotlib; each returns a Figure and draws nothing itself)
# ---------------------------------------------------------------------------
def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    return fig, ax


def _threshold_lines(ax, ymax: float) -> None:
    for level, value in enumerate(LEVEL_THRESHOLDS, start=1):
        if value < ymax:
            ax.axhline(value, color=AXIS, linewidth=0.8, linestyle=(0, (4, 3)))
            ax.annotate(f"L{level} from {value:g}", xy=(1.0, value),
                        xycoords=("axes fraction", "data"), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK_2)


def window_rows(station: pd.DataFrame, when: pd.Timestamp, hours: int) -> pd.DataFrame:
    start = when - pd.Timedelta(hours=hours)
    return station[(station["timestamp"] >= start) & (station["timestamp"] <= when)]


def fig_timeline(win: pd.DataFrame, when: pd.Timestamp, show_band: bool,
                 show_mean: bool = False):
    """Observed PM2.5 against the forecast that was issued three hours earlier."""
    fig, ax = _figure(9.6, 3.4)
    values = [win["reg"].max(), win["q95"].max() if show_band else np.nan,
              win["pm2_5"].max(), win["target_pm25_3h"].max()]
    ymax = float(np.nanmax(values)) * 1.12 if np.isfinite(np.nanmax(values)) else 60.0
    if show_band:
        ax.fill_between(win["valid_time"], win["q05"], win["q95"], color=ORANGE,
                        alpha=0.18, linewidth=0, label="Tree spread, 5th to 95th percentile")
    if win["pm2_5"].notna().any():
        ax.plot(win["timestamp"], win["pm2_5"], color=BLUE, linewidth=2,
                label="Observed PM2.5")
    if win["target_pm25_3h"].notna().any():
        future = win[win["valid_time"] > when]
        ax.plot(future["valid_time"], future["target_pm25_3h"], color=BLUE, linewidth=2,
                linestyle=(0, (1, 2)), label="What happened next (shown for replay)")
    ax.plot(win["valid_time"], win["reg"], color=ORANGE, linewidth=2,
            label=f"M5 forecast issued {HORIZON_H} h earlier")
    if show_mean and win["pm2_5_mean_24h"].notna().any():
        ax.plot(win["timestamp"], win["pm2_5_mean_24h"], color=INK_2, linewidth=1.5,
                linestyle=(0, (5, 3)), label="24 h running mean of observed PM2.5")
    ax.axvline(when, color=INK_2, linewidth=1)
    ax.annotate("now", xy=(when, 1.0), xycoords=("data", "axes fraction"),
                xytext=(0, 3), textcoords="offset points", ha="center",
                fontsize=8, color=INK_2)
    _threshold_lines(ax, ymax)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("PM2.5 (µg/m³)", fontsize=9, color=INK_2)
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    return fig


def fig_level_strip(win: pd.DataFrame, column: str):
    """Two rows of level tiles over the same valid times: forecast and outcome."""
    fig, ax = plt.subplots(figsize=(9.6, 1.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    grid = np.vstack([win[column].to_numpy(), win["y_true"].to_numpy()])
    ax.imshow(grid, aspect="auto", cmap=ListedColormap(LEVEL_COLORS), vmin=0, vmax=3,
              interpolation="nearest")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Forecast", "Outcome"], fontsize=9, color=INK_2)
    ticks = np.linspace(0, len(win) - 1, num=min(6, len(win))).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([win["valid_time"].iloc[i].strftime("%d %b %H:%M") for i in ticks],
                       fontsize=8, color=INK_2)
    ax.axhline(0.5, color=SURFACE, linewidth=2)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.legend(handles=[Patch(color=LEVEL_COLORS[k], label=level_label(k)) for k in range(4)],
              loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=4, fontsize=8,
              frameon=False)
    fig.tight_layout()
    return fig


def fig_probabilities(row: pd.Series):
    fig, ax = _figure(5.2, 3.0)
    probs = [float(row[f"p{k}"]) for k in range(4)]
    bars = ax.bar(range(4), probs, color=LEVEL_COLORS, width=0.62)
    for bar, value in zip(bars, probs):
        ax.annotate(f"{value:.0%}", xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=9, color=INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels([level_label(k) for k in range(4)], fontsize=9, color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("M6 probability", fontsize=9, color=INK_2)
    fig.tight_layout()
    return fig


def cumulative_table(row: pd.Series, tau) -> pd.DataFrame:
    """P(level >= k) next to the M7 threshold it is compared with."""
    probs = [float(row[f"p{k}"]) for k in range(4)]
    rows = []
    for k, t in enumerate(tau, start=1):
        cumulative = sum(probs[k:])
        rows.append({"Question": f"At least L{k}?",
                     "P(at least)": round(cumulative, 3),
                     "M7 τ": float(t),
                     "Pass": "yes" if cumulative >= float(t) else "no"})
    return pd.DataFrame(rows)


def fig_benchmark(table: pd.DataFrame, metric: str):
    higher = METRICS[metric]
    fig, ax = _figure(7.4, 0.42 * len(table) + 1.0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    values = table[metric].to_numpy(dtype=float)
    best = int(np.nanargmax(values) if higher else np.nanargmin(values))
    positions = np.arange(len(table))[::-1]
    ax.barh(positions, values, color=BLUE, height=0.6)
    for pos, value, idx in zip(positions, values, range(len(table))):
        note = "  best" if idx == best else ""
        ax.annotate(f"{value:.4f}{note}", xy=(value, pos), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_yticks(positions)
    ax.set_yticklabels(table["model"], fontsize=9, color=INK)
    ax.set_xlim(0, float(np.nanmax(values)) * 1.18)
    ax.set_xlabel(f"{metric}  ({'higher' if higher else 'lower'} is better)",
                  fontsize=9, color=INK_2)
    fig.tight_layout()
    return fig


def fig_map(snapshot: pd.DataFrame, when: pd.Timestamp):
    fig, ax = _figure(8.6, 4.6)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for role, marker in ROLE_MARKERS.items():
        part = snapshot[snapshot["role"] == role]
        ax.scatter(part["lon"], part["lat"], s=95 if role == "receptor" else 60,
                   marker=marker, c=[LEVEL_COLORS[k] for k in part["level"]],
                   edgecolors=SURFACE, linewidths=1.5, zorder=3)
    wind = snapshot.dropna(subset=["u", "v"])
    if len(wind):
        ax.quiver(wind["lon"], wind["lat"], wind["u"], wind["v"], color=INK_2,
                  width=0.0025, scale=90, zorder=2)
    # the three receptors sit within about 45 km, so labels are pulled out on leader lines
    receptors = snapshot[snapshot["role"] == "receptor"].sort_values("lat", ascending=False)
    offsets = [((-70, 34), "right"), ((-80, 0), "right"), ((-70, -34), "right")]
    for (_, node), (offset, align) in zip(receptors.iterrows(), offsets):
        ax.annotate(node["station_id"].replace("_", " "), xy=(node["lon"], node["lat"]),
                    xytext=offset, textcoords="offset points", ha=align, va="center",
                    fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.8,
                                    shrinkA=2, shrinkB=5))
    handles = [Patch(color=LEVEL_COLORS[k], label=level_label(k)) for k in range(4)]
    handles += [Line2D([], [], marker=m, linestyle="", color=INK_2, markersize=7,
                       label=role.capitalize()) for role, m in ROLE_MARKERS.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False, ncol=2)
    ax.set_xlabel("Longitude", fontsize=9, color=INK_2)
    ax.set_ylabel("Latitude", fontsize=9, color=INK_2)
    ax.set_title(f"Current PM2.5 level and wind at {when:%d %b %Y %H:%M}. "
                 "Arrows point downwind.", fontsize=9, color=INK_2, loc="left")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return fig


def map_snapshot(panel: pd.DataFrame, when: pd.Timestamp) -> pd.DataFrame:
    now = panel[panel["timestamp"] == when]
    nodes = pd.DataFrame([{"station_id": n["id"], "lat": n["lat"], "lon": n["lon"],
                           "role": n["role"]} for n in NODES])
    snap = nodes.merge(now, on="station_id", how="inner").dropna(subset=["pm2_5"])
    snap["level"] = np.searchsorted(LEVEL_THRESHOLDS, snap["pm2_5"], side="right")
    # wind_direction_10m is the direction the wind comes from, clockwise from north
    theta = np.deg2rad(snap["wind_direction_10m"].astype(float))
    speed = snap["wind_speed_ms"].astype(float)
    snap["u"], snap["v"] = -speed * np.sin(theta), -speed * np.cos(theta)
    return snap


def fig_loso(loso: pd.DataFrame):
    order = {"receptor": 0, "corridor": 1, "source": 2}
    data = loso.assign(_o=loso["role"].map(order)).sort_values(
        ["_o", "High_Risk_Recall"], ascending=[True, False], na_position="last")
    fig, ax = _figure(7.4, 0.3 * len(data) + 1.0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    positions = np.arange(len(data))[::-1]
    values = data["High_Risk_Recall"].to_numpy(dtype=float)
    ax.barh(positions, np.nan_to_num(values), color=BLUE, height=0.6)
    for pos, value, count in zip(positions, values, data["n_high_risk"]):
        text = "no high-risk samples" if np.isnan(value) else f"{value:.2f}  (n={int(count)})"
        ax.annotate(text, xy=(0 if np.isnan(value) else value, pos), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(positions)
    ax.set_yticklabels([f"{s.replace('_', ' ')}  ·  {r}"
                        for s, r in zip(data["station_id"], data["role"])],
                       fontsize=8, color=INK)
    ax.set_xlim(0, 1.25)
    ax.set_ylim(-0.7, len(data) - 0.3)
    ax.set_xlabel("High_Risk_Recall on the held-out node (higher is better)",
                  fontsize=9, color=INK_2)
    fig.tight_layout()
    return fig


def show(fig) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def missing(path: str) -> None:
    st.warning(f"`{path}` is not available, so this part is hidden.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp, str]:
    st.sidebar.header("Replay controls")
    split_label = st.sidebar.radio(
        "Period", ["Test (Mar to Oct 2026)", "Validation (Jul 2025 to Mar 2026)"], index=0)
    split = "test" if split_label.startswith("Test") else "val"
    stations = sorted(data["station_id"].unique())
    station_id = st.sidebar.selectbox("Receptor node", stations, index=0,
                                      format_func=lambda s: s.replace("_", " "))
    station = data[(data["split"] == split) & (data["station_id"] == station_id)]
    station = add_running_mean(station.sort_values("timestamp").reset_index(drop=True))

    first, last = station["timestamp"].iloc[0], station["timestamp"].iloc[-1]
    ranked = station["pm2_5"] if station["pm2_5"].notna().any() else station["reg"]
    peak = station["timestamp"].iloc[int(ranked.to_numpy().argmax())]
    day = st.sidebar.date_input("Date", value=peak.date(), min_value=first.date(),
                                max_value=last.date())
    hour = st.sidebar.slider("Hour (local time)", 0, 23, int(peak.hour))
    wanted = pd.Timestamp(day) + pd.Timedelta(hours=int(hour))
    earlier = station[station["timestamp"] <= wanted]
    when = (earlier["timestamp"].iloc[-1] if len(earlier) else first)
    if when != wanted:
        st.sidebar.caption(f"No eligible row at that hour. Showing {when:%d %b %Y %H:%M}.")

    rule = st.sidebar.selectbox("Decision rule", list(RULES), index=list(RULES).index(DEFAULT_RULE),
                                format_func=lambda k: RULES[k]["label"])
    st.sidebar.caption(
        "M5: regression forest cut into levels. M6: ordinal forest. "
        "M7: M6 probabilities with cost-sensitive thresholds.")
    st.sidebar.caption(
        "Replay only. The app reads saved predictions; it does not refit or retune anything.")
    return station, when, rule


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def tab_risk(station: pd.DataFrame, when: pd.Timestamp, rule: str, ref: dict,
             benchmark: pd.DataFrame | None) -> None:
    row = station[station["timestamp"] == when].iloc[0]
    column = RULES[rule]["column"]
    predicted = int(row[column])
    c1, c2, c3, c4 = st.columns(4)
    if pd.notna(row["pm2_5"]):
        current = int(np.searchsorted(LEVEL_THRESHOLDS, row["pm2_5"], side="right"))
        c1.metric("PM2.5 now", f"{row['pm2_5']:.0f} µg/m³")
        c1.caption(level_label(current))
    else:
        c1.metric("PM2.5 now", "n/a")
    c2.metric(f"Forecast level in {HORIZON_H} h", level_label(predicted))
    c2.caption(RULES[rule]["label"])
    c3.metric(f"Forecast PM2.5 in {HORIZON_H} h", f"{row['reg']:.0f} µg/m³")
    c3.caption(f"Tree spread {row['q05']:.0f} to {row['q95']:.0f} µg/m³")
    c4.metric("What happened (replay)", level_label(int(row["y_true"])))
    c4.caption("Forecast level was correct" if predicted == int(row["y_true"])
               else f"Forecast was off by {abs(predicted - int(row['y_true']))} level(s)")

    st.markdown(f"**Advice for {level_label(predicted)}:** {ADVICE[predicted]}")
    grade, why = reliability_note(row, ref)
    st.info(f"**{grade}.** This forecast was issued at {when:%d %b %Y %H:%M}: {why}")

    win = window_rows(station, when, 72)
    show(fig_timeline(win, when, show_band=False, show_mean=True))
    st.caption("The dashed line is a 24 h running mean of the same observations. It is a "
               "stand-in for an index built on a daily average, not the official API value.")
    st.caption("Level tiles for the same hours. Top row: the selected decision rule. "
               "Bottom row: the level that was actually reached.")
    show(fig_level_strip(win, column))

    if benchmark is not None:
        hit = benchmark[(benchmark["model"] == RULES[rule]["benchmark"])
                        & (benchmark["split"] == "test")]
        if len(hit):
            m = hit.iloc[0]
            st.caption(
                f"Fixed test result of this rule: High_Risk_Recall {m['High_Risk_Recall']:.4f}, "
                f"Macro_F1 {m['Macro_F1']:.4f}, Avg_Asymmetric_Cost "
                f"{m['Avg_Asymmetric_Cost']:.4f} (n = {int(m['n_samples']):,}).")

    panel = load_panel()
    if panel is None:
        missing("data/haze_panel_clean.parquet")
    else:
        snap = map_snapshot(panel, when)
        if len(snap):
            show(fig_map(snap, when))
            st.caption("Each point is a CAMS model grid cell, not a ground monitoring station.")


def tab_probability(station: pd.DataFrame, when: pd.Timestamp, preds: pd.DataFrame) -> None:
    row = station[station["timestamp"] == when].iloc[0]
    thresholds = load_thresholds()
    tau = thresholds["thresholds"]
    st.subheader("Four-level probability")
    left, right = st.columns(2)
    with left:
        show(fig_probabilities(row))
    with right:
        st.dataframe(cumulative_table(row, tau), use_container_width=True, hide_index=True)
        st.caption("M7 walks up the levels and stops at the first question that fails.")

    st.subheader("Concentration and tree spread")
    win = window_rows(station, when, 72)
    show(fig_timeline(win, when, show_band=True))
    stage5 = load_json_or_none("metrics_stage5.json")
    if stage5 and "interval" in stage5:
        test = stage5["interval"]["test"]
        st.warning(
            f"The band is the spread between the trees of the M5 forest. It is not a "
            f"calibrated prediction interval: on the test split it covered "
            f"{test['PICP']:.1%} of outcomes with a mean width of {test['MPIW']:.1f} µg/m³.")

    st.subheader("Threshold trade-off (validation split only)")
    st.caption("Move the three thresholds to see what the cost-sensitive rule trades away. "
               "This uses validation rows only; the test split is never touched here.")
    val = preds[preds["split"] == "val"]
    proba = val[["p0", "p1", "p2", "p3"]].to_numpy()
    s1, s2, s3 = st.columns(3)
    t1 = s1.slider("τ1  for level ≥ 1", 0.05, 0.95, float(tau[0]), 0.05)
    t2 = s2.slider("τ2  for level ≥ 2", 0.05, 0.95, float(tau[1]), 0.05)
    t3 = s3.slider("τ3  for level ≥ 3", 0.05, 0.95, float(tau[2]), 0.05)
    rows = {
        "Your thresholds": evaluate_predictions(val["y_true"], apply_thresholds(proba, (t1, t2, t3))),
        f"M7 selected ({tau[0]:.2f}, {tau[1]:.2f}, {tau[2]:.2f})":
            evaluate_predictions(val["y_true"], val["pred_m7"]),
        "M6 most probable level": evaluate_predictions(val["y_true"], val["pred_m6"]),
    }
    table = pd.DataFrame(rows).T[list(METRICS)]
    st.dataframe(table.style.format("{:.4f}"), use_container_width=True)


def tab_models(benchmark: pd.DataFrame | None) -> None:
    if benchmark is None:
        missing("models/all_models_benchmark.csv")
        return
    splits = sorted(benchmark["split"].unique())
    c1, c2 = st.columns(2)
    split = c1.selectbox("Split", splits, index=splits.index("test") if "test" in splits else 0)
    metric = c2.selectbox("Metric", list(METRICS), index=2)
    table = benchmark[benchmark["split"] == split].reset_index(drop=True)
    show(fig_benchmark(table, metric))

    sizes = sorted(table["n_samples"].unique())
    if len(sizes) > 1:
        odd = table[table["n_samples"] != table["n_samples"].mode().iloc[0]]
        st.warning("Not every model was scored on the same rows: "
                   + ", ".join(f"{m} uses n = {int(n):,}"
                               for m, n in zip(odd["model"], odd["n_samples"]))
                   + f"; the others use n = {int(table['n_samples'].mode().iloc[0]):,}.")
    if not table["model"].astype(str).str.startswith("M0").any():
        st.info("The persistence reference M0 is not in the benchmark file yet.")
    with st.expander("Table view", expanded=True):
        st.dataframe(table.style.format({m: "{:.4f}" for m in METRICS}),
                     use_container_width=True, hide_index=True)

    m7 = load_json_or_none("m7_test_metrics.json")
    if m7:
        st.subheader("Three decisions on the same M6 probabilities (test)")
        st.dataframe(pd.DataFrame(m7["metrics"]).T[list(METRICS)].style.format("{:.4f}"),
                     use_container_width=True)
    st.subheader("Asymmetric cost matrix")
    cost = pd.DataFrame(DEFAULT_COST_MATRIX,
                        index=[f"true {level_label(k)}" for k in range(4)],
                        columns=[f"pred {level_label(k)}" for k in range(4)])
    st.dataframe(cost, use_container_width=True)
    st.caption("Missing a high-risk hour (lower left) costs far more than a false alarm.")


def tab_explain() -> None:
    figures = [
        ("fig_perm_importance.png", "Permutation importance of the M6 ordinal forest (validation)."),
        ("fig_pdp_wind_humidity.png", "Partial dependence of the M1 baseline tree on wind speed and humidity."),
        ("fig_roc_high_risk.png", "ROC of the high-risk (L2 or L3) decision on validation."),
        ("fig_confusion_matrices_comparison.png", "Confusion matrices on validation."),
    ]
    for start in range(0, len(figures), 2):
        for column, (name, caption) in zip(st.columns(2), figures[start:start + 2]):
            path = FIGURES / name
            with column:
                if path.exists():
                    st.image(str(path), caption=caption, use_container_width=True)
                else:
                    missing(f"figures/{name}")

    st.subheader("Leave-one-node-out")
    loso = load_table("loso_results.csv")
    if loso is None:
        missing("models/loso_results.csv")
    else:
        st.caption(f"Model in this experiment: {', '.join(loso['model'].unique())}. "
                   "Nodes without any high-risk hour have no recall value.")
        show(fig_loso(loso))
        with st.expander("Table view"):
            st.dataframe(loso, use_container_width=True, hide_index=True)

    st.subheader("Seasonal slices of the 2026 hold-out")
    seasonal = load_table("seasonal_metrics.csv")
    if seasonal is None:
        missing("models/seasonal_metrics.csv")
    else:
        st.dataframe(seasonal, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="HazeGuard-KL", layout="wide")
    st.title("HazeGuard-KL")
    st.caption(f"Greater Kuala Lumpur PM2.5 risk level, {HORIZON_H} hours ahead. "
               "Inputs are CAMS model products from Open-Meteo, not ground measurements.")

    preds = load_predictions()
    obs = load_observations()
    data = joined(preds, obs)
    if obs is None:
        st.warning("`data/features_final.parquet` could not be read, so current PM2.5 and "
                   "the realised concentration are hidden. Forecast levels still work.")
    problem = self_check(preds)
    if problem:
        st.error(problem)

    station, when, rule = sidebar(data)
    benchmark = load_table("all_models_benchmark.csv")
    ref = reliability_reference(preds)

    t1, t2, t3, t4 = st.tabs([
        f"Risk now and in {HORIZON_H} h", "Probability and spread",
        "Model comparison", "Explanation and generalisation"])
    with t1:
        tab_risk(station, when, rule, ref, benchmark)
    with t2:
        tab_probability(station, when, preds)
    with t3:
        tab_models(benchmark)
    with t4:
        tab_explain()


if __name__ == "__main__":
    main()
