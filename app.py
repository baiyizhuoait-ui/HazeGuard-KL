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

import html as html_lib
import json
import sys
import time
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
LEVEL_CHIP_INK = ["#ffffff", "#0b0b0b", "#0b0b0b", "#ffffff"]
PALETTES = {
    "light": dict(INK="#0b0b0b", INK_2="#52514e", GRID="#e1e0d9", AXIS="#c3c2b7",
                  SURFACE="#fcfcfb", PAGE="#f5f5f2", CARD="#ffffff",
                  BORDER="rgba(11,11,11,0.10)", BLUE="#2a78d6", ORANGE="#eb6834"),
    "dark": dict(INK="#f0efec", INK_2="#c3c2b7", GRID="#2c2c2a", AXIS="#4a4a46",
                 SURFACE="#1a1a19", PAGE="#101010", CARD="#1a1a19",
                 BORDER="rgba(255,255,255,0.12)", BLUE="#3987e5", ORANGE="#d95926"),
}
INK = INK_2 = GRID = AXIS = SURFACE = PAGE = CARD = BORDER = BLUE = ORANGE = ""
ROLE_MARKERS = {"receptor": "o", "corridor": "s", "source": "^"}


def apply_palette(dark: bool) -> None:
    """Point the chart colours at the light or dark palette for this run."""
    globals().update(PALETTES["dark" if dark else "light"])
    matplotlib.rcParams.update({
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.edgecolor": AXIS, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


apply_palette(False)

# ---------------------------------------------------------------------------
# Language. Headings, labels and controls are bilingual; long method notes and
# the text inside charts stay in English, the language of the report.
# ---------------------------------------------------------------------------
ZH = {
    "Replay controls": "回放控制", "Period": "时段", "Receptor node": "受体格点",
    "Go to": "跳转方式", "High-risk episode": "高危事件", "Date and hour": "日期与小时",
    "Episode (highest peak first)": "事件（按峰值从高到低）", "Date": "日期",
    "Hour (local time)": "小时（当地时间）", "Decision rule": "判定规则",
    "Test (Mar to Oct 2026)": "测试段（2026 年 3–10 月）",
    "Validation (Jul 2025 to Mar 2026)": "验证段（2025 年 7 月–2026 年 3 月）",
    "Dark mode": "深色模式", "Language": "语言", "Display": "显示",
    "Risk now and in 3 h": "当前风险与 3 小时预测", "Probability and interval": "概率与区间",
    "Model comparison": "模型对比", "Explanation and generalisation": "解释与泛化",
    "Method and limits": "方法与局限",
    "PM2.5 now": "当前 PM2.5", "Forecast level in 3 h": "3 小时后预测等级",
    "Forecast PM2.5 in 3 h": "3 小时后预测浓度", "What happened (replay)": "实际结果（回放）",
    "Low confidence": "可信度低", "Moderate confidence": "可信度中等",
    "High confidence": "可信度高", "Forecast for": "预测时刻", "issued": "发布于",
    "Play": "播放", "Pause": "暂停", "Download this window (CSV)": "下载当前时段（CSV）",
    "Major episodes at this node": "该格点的重大事件", "Replay": "回放",
    "peak": "峰值", "Missed": "漏报", "Alert at onset": "起始时才报警",
    "Alert after onset": "起始后才报警",
    "What is unusual right now": "此刻有什么异常", "Feature": "特征", "Value now": "当前值",
    "Percentile in training history": "在训练期历史中的分位",
    "Early-warning record on the test split": "测试段预警记录",
    "High-risk onsets": "高危起始次数", "Alert before onset": "起始前已报警",
    "24 h mean lags by": "24 小时均值滞后", "Fresh alerts confirmed": "新发警报被证实",
    "Four-level probability": "四级概率",
    "Concentration and prediction interval": "浓度与预测区间",
    "Concentration and tree spread": "浓度与树间离散度",
    "Are the probabilities calibrated?": "概率是否校准？",
    "Threshold trade-off (validation split only)": "阈值权衡（仅验证段）",
    "M8 · v2 control for known defect KD-01": "M8 · 已知缺陷 KD-01 的 v2 对照",
    "Three decisions on the same M6 probabilities (test)": "同一 M6 概率上的三种决策（测试段）",
    "Where the concentration forecast is off (M5, test)": "浓度预测偏在哪里（M5，测试段）",
    "Asymmetric cost matrix": "非对称代价矩阵", "Leave-one-node-out": "留一格点验证",
    "Seasonal slices of the 2026 hold-out": "2026 年留出段的分季节结果",
    "How the system is put together": "系统是怎么搭起来的",
    "Known limits": "已知局限", "Limit": "局限", "What it affects": "影响", "Evidence": "依据",
}
ADVICE_ZH = {
    0: "可正常进行户外活动。", 1: "敏感人群应减少长时间户外运动。",
    2: "减少户外活动，关闭门窗。", 3: "避免户外活动；必须外出时佩戴 N95 口罩。",
}


def lang() -> str:
    return st.session_state.get("lang", "en")


def tr(text: str) -> str:
    return ZH.get(text, text) if lang() == "zh" else text


def L(en: str, zh: str) -> str:
    return zh if lang() == "zh" else en


def inject_css() -> None:
    st.markdown(f"""<style>
    #MainMenu, footer, [data-testid="stAppDeployButton"], [data-testid="stMainMenu"],
    [data-testid="stDecoration"] {{ display: none !important; }}
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background: {PAGE}; color: {INK}; }}
    [data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid {BORDER}; }}
    .block-container {{ padding-top: 2.2rem; max-width: 1320px; }}
    [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
    [data-testid="stMetricValue"], .stApp h1, .stApp h2, .stApp h3 {{ color: {INK}; }}
    [data-testid="stCaptionContainer"], [data-testid="stMetricLabel"] {{ color: {INK_2}; }}
    [data-testid="stMetric"] {{ background: {CARD}; border: 1px solid {BORDER};
        border-radius: 14px; padding: 16px 18px; }}
    [data-testid="stMetricValue"] {{ font-weight: 650; letter-spacing: -0.01em; }}
    .stApp h1 {{ font-weight: 750; letter-spacing: -0.02em; margin-bottom: 0; }}
    .hg-banner {{ display: flex; align-items: center; gap: 18px; background: {CARD};
        border: 1px solid {BORDER}; border-radius: 16px; padding: 18px 22px; margin: 6px 0 16px; }}
    .hg-chip {{ font-size: 1.35rem; font-weight: 750; padding: 12px 20px; border-radius: 12px;
        white-space: nowrap; }}
    .hg-banner-main {{ flex: 1; min-width: 0; }}
    .hg-banner-title {{ font-size: 1.05rem; font-weight: 650; color: {INK}; }}
    .hg-banner-sub {{ color: {INK_2}; margin-top: 4px; }}
    .hg-badge {{ border: 1px solid {BORDER}; border-radius: 999px; padding: 6px 14px;
        font-weight: 600; color: {INK}; white-space: nowrap; }}
    .hg-tag {{ display: inline-block; border-radius: 999px; padding: 2px 10px; font-size: 0.8rem;
        font-weight: 600; border: 1px solid {BORDER}; color: {INK}; }}
    </style>""", unsafe_allow_html=True)


def sync_native_theme(dark: bool) -> None:
    """Ask Streamlit itself to switch base theme so tables and widgets follow.
    Uses a non-public call; when it is unavailable the CSS above still applies."""
    changed = False
    try:
        want = "dark" if dark else "light"
        if st._config.get_option("theme.base") != want:
            st._config.set_option("theme.base", want)
            changed = True
    except Exception:
        changed = False
    if changed:
        st.rerun()


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
    calibrated = MODELS / "stage5_intervals_calibrated.npz"
    if calibrated.exists():
        cal = np.load(calibrated)
        rows = []
        for split in ("val", "test"):
            interval = cal[f"interval_{split}"]
            rows.append(pd.DataFrame({
                "station_id": cal[f"station_id_{split}"].astype(str),
                "timestamp": pd.to_datetime(cal[f"ts_{split}"]),
                "cal_lo": interval[:, 0], "cal_mid": interval[:, 1], "cal_hi": interval[:, 2]}))
        out = out.merge(pd.concat(rows, ignore_index=True), on=["station_id", "timestamp"],
                        how="left", validate="one_to_one")
    return out


def interval_label() -> tuple[str, str] | None:
    """Short legend label and the coverage note for member 5's calibrated interval."""
    record = load_json_or_none("metrics_stage5_interval_calibrated.json")
    if not record:
        return None
    nominal = record.get("nominal_coverage", 0.9)
    test = record["test"]["calibrated"]
    val = record["val"]["calibrated"]
    label = f"{nominal:.0%} target-coverage interval (calibrated on validation)"
    note = (f"Interval: {nominal:.0%} target coverage, calibrated on the validation split "
            f"({val['PICP']:.1%} there). On the test split it covered {test['PICP']:.1%} of "
            f"outcomes with a mean width of {test['MPIW']:.1f} µg/m³, so the 90% target is not "
            "met on unseen data. Method and limits: models/interval_calibration.json.")
    return label, note


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
        reasons.append(L(f"the most probable level has only {top:.0%} probability",
                         f"最可能的等级只有 {top:.0%} 的概率"))
    if width > ref["wide"]:
        reasons.append(L(f"the trees disagree by {width:.0f} µg/m³",
                         f"各棵树的预测相差 {width:.0f} µg/m³"))
    if crossed:
        cuts = " / ".join(f"{t:g}" for t in crossed)
        reasons.append(L(f"the tree spread crosses the {cuts} µg/m³ boundary",
                         f"树间离散范围跨过了 {cuts} µg/m³ 的等级分界"))
    if len(reasons) >= 2:
        return "Low confidence", "; ".join(reasons) + "."
    if reasons:
        return "Moderate confidence", reasons[0] + "."
    return "High confidence", L(
        f"the most probable level has {top:.0%} probability and the tree spread "
        f"({width:.0f} µg/m³) stays inside one level.",
        f"最可能的等级有 {top:.0%} 的概率，树间离散范围（{width:.0f} µg/m³）没有跨过等级分界。")


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


def run_boundary_hours() -> list[int]:
    """Local hours at which two CAMS forecast runs are joined (see models/seam_report.json)."""
    seam = load_json_or_none("seam_report.json")
    hours = (seam or {}).get("seam_hours_local") or [8, 20]
    return [int(h) for h in hours]


def fig_timeline(win: pd.DataFrame, when: pd.Timestamp, show_band: bool,
                 show_mean: bool = False):
    """Observed PM2.5 against the forecast that was issued three hours earlier."""
    fig, ax = _figure(9.6, 3.7)
    boundaries = run_boundary_hours()
    if len(win):
        span = pd.date_range(win["timestamp"].min().floor("h"),
                             win["valid_time"].max().ceil("h"), freq="h")
        marks = [t for t in span if t.hour in boundaries]
        for i, mark in enumerate(marks):
            ax.axvline(mark, color=AXIS, linewidth=0.9, linestyle=(0, (1, 3)), zorder=0,
                       label=("CAMS run boundary ("
                              + ", ".join(f"{h:02d}:00" for h in boundaries) + ")")
                       if i == 0 else None)
    values = [win["reg"].max(),
              (win["cal_hi"].max() if "cal_hi" in win else win["q95"].max()) if show_band else np.nan,
              win["pm2_5"].max(), win["target_pm25_3h"].max()]
    ymax = float(np.nanmax(values)) * 1.12 if np.isfinite(np.nanmax(values)) else 60.0
    if show_band:
        calibrated = interval_label() if "cal_lo" in win and win["cal_lo"].notna().any() else None
        lo, hi = ("cal_lo", "cal_hi") if calibrated else ("q05", "q95")
        ax.fill_between(win["valid_time"], win[lo], win[hi], color=ORANGE, alpha=0.18,
                        linewidth=0,
                        label=calibrated[0] if calibrated else "Tree spread, 5th to 95th percentile")
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
    ax.annotate("now", xy=(when, 0.0), xycoords=("data", "axes fraction"),
                xytext=(4, 4), textcoords="offset points", ha="left",
                fontsize=8, color=INK_2)
    _threshold_lines(ax, ymax)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("PM2.5 (µg/m³)", fontsize=9, color=INK_2)
    # legend sits above the plot so it can never cover the first hours of the curve
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.04), fontsize=8, frameon=False,
              ncol=3, borderaxespad=0.0, columnspacing=1.4, handlelength=1.8)
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
    # the three receptors sit within about 45 km, so they get a zoomed inset
    receptors = snapshot[snapshot["role"] == "receptor"]
    if len(receptors):
        lon0, lon1 = receptors["lon"].min() - 0.45, receptors["lon"].max() + 0.45
        lat0, lat1 = receptors["lat"].min() - 0.35, receptors["lat"].max() + 0.35
        inset = ax.inset_axes([0.02, 0.04, 0.30, 0.44])
        inset.set_facecolor(SURFACE)
        near = snapshot[snapshot["lon"].between(lon0, lon1) & snapshot["lat"].between(lat0, lat1)]
        for role, marker in ROLE_MARKERS.items():
            part = near[near["role"] == role]
            inset.scatter(part["lon"], part["lat"], s=110 if role == "receptor" else 70,
                          marker=marker, c=[LEVEL_COLORS[k] for k in part["level"]],
                          edgecolors=SURFACE, linewidths=1.5, zorder=3)
        for _, node in near.iterrows():
            inset.annotate(node["station_id"].replace("_", " "), xy=(node["lon"], node["lat"]),
                           xytext=(0, 9), textcoords="offset points", ha="center",
                           fontsize=7.5, color=INK)
        inset.set_xlim(lon0, lon1)
        inset.set_ylim(lat0, lat1)
        inset.set_xticks([])
        inset.set_yticks([])
        for side in inset.spines.values():
            side.set_color(AXIS)
        inset.set_title("Greater Kuala Lumpur", fontsize=8, color=INK_2, loc="left", pad=3)
        ax.indicate_inset_zoom(inset, edgecolor=INK_2, alpha=0.6, linewidth=0.8)
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


def fig_lead_counts(counts: dict):
    """How many high-risk onsets were flagged 3, 2, 1, 0 hours ahead, late, or not at all."""
    order = [str(h) for h in range(HORIZON_H, -1, -1)] + ["late", "missed"]
    labels = [f"{h} h ahead" for h in range(HORIZON_H, 0, -1)] + [
        "at onset", "after onset", "missed"]
    values = [int(counts.get(k, 0)) for k in order]
    fig, ax = _figure(7.4, 2.9)
    bars = ax.bar(range(len(values)), values, color=BLUE, width=0.62)
    for bar, value in zip(bars, values):
        ax.annotate(str(value), xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=9, color=INK)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=9, color=INK)
    ax.set_ylim(0, max(values + [1]) * 1.15)
    ax.set_ylabel("High-risk onsets", fontsize=9, color=INK_2)
    fig.tight_layout()
    return fig


def fig_onsets_by_hour(by_hour: dict, boundaries: list[int]):
    """High-risk onsets by local hour; run-boundary hours carry their own colour and legend."""
    hours = list(range(24))
    values = [int(by_hour.get(str(h), 0)) for h in hours]
    colours = [ORANGE if h in boundaries else BLUE for h in hours]
    fig, ax = _figure(9.6, 2.9)
    ax.bar(hours, values, color=colours, width=0.7)
    for hour, value in zip(hours, values):
        if value and (hour in boundaries or value >= 0.5 * max(values)):
            ax.annotate(str(value), xy=(hour, value), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}" for h in hours], fontsize=8, color=INK_2)
    ax.set_xlabel("Local hour of the onset", fontsize=9, color=INK_2)
    ax.set_ylabel("High-risk onsets", fontsize=9, color=INK_2)
    ax.set_ylim(0, max(values + [1]) * 1.15)
    ax.legend(handles=[Patch(color=ORANGE, label="CAMS run-boundary hour"),
                       Patch(color=BLUE, label="Other hours")],
              loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def fig_reliability(blocks: dict):
    """Observed frequency against mean predicted probability, one line per split."""
    fig, ax = _figure(4.6, 4.2)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle=(0, (4, 3)),
            label="Perfect calibration")
    names = {"val": ("Validation", BLUE, "o"), "test": ("Test", ORANGE, "s")}
    for split, (label, colour, marker) in names.items():
        rows = [b for b in blocks.get(split, {}).get("bins", []) if b["n"]]
        if rows:
            ax.plot([b["mean_predicted"] for b in rows], [b["observed_frequency"] for b in rows],
                    color=colour, linewidth=2, marker=marker, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Mean predicted probability", fontsize=9, color=INK_2)
    ax.set_ylabel("Observed frequency", fontsize=9, color=INK_2)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def fig_error_by_level(by_level: dict):
    """Mean error of the M5 concentration forecast for each true level."""
    levels = [k for k in ("0", "1", "2", "3") if k in by_level]
    values = [by_level[k]["mean_error"] for k in levels]
    fig, ax = _figure(5.6, 3.0)
    bars = ax.bar(range(len(levels)), values, color=BLUE, width=0.62)
    ax.axhline(0, color=AXIS, linewidth=1)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:+.1f}", xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3 if value >= 0 else -11), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([level_label(int(k)) for k in levels], fontsize=9, color=INK)
    low, high = min(values + [0]), max(values + [0])
    pad = max(high - low, 1.0) * 0.2
    ax.set_ylim(low - pad, high + pad)
    ax.set_ylabel("Forecast minus outcome (µg/m³)", fontsize=9, color=INK_2)
    fig.tight_layout()
    return fig


def banner(row: pd.Series, predicted: int, ref: dict, when: pd.Timestamp) -> None:
    grade, why = reliability_note(row, ref)
    valid = when + pd.Timedelta(hours=HORIZON_H)
    advice = ADVICE_ZH[predicted] if lang() == "zh" else ADVICE[predicted]
    st.markdown(f"""<div class="hg-banner" style="border-left: 8px solid {LEVEL_COLORS[predicted]}">
      <div class="hg-chip" style="background:{LEVEL_COLORS[predicted]};color:{LEVEL_CHIP_INK[predicted]}">
        {html_lib.escape(level_label(predicted))}</div>
      <div class="hg-banner-main">
        <div class="hg-banner-title">{tr("Forecast for")} {valid:%d %b %Y %H:%M}
          · {tr("issued")} {when:%H:%M} · {html_lib.escape(advice)}</div>
        <div class="hg-banner-sub">{html_lib.escape(why)}</div>
      </div>
      <div class="hg-badge">{tr(grade)}</div>
    </div>""", unsafe_allow_html=True)


def snap(station: pd.DataFrame, wanted: pd.Timestamp) -> pd.Timestamp:
    """Latest row at or before the wanted time; the first row when there is none."""
    earlier = station[station["timestamp"] <= wanted]
    return earlier["timestamp"].iloc[-1] if len(earlier) else station["timestamp"].iloc[0]


def playback_bar(station: pd.DataFrame, when: pd.Timestamp, rule: str) -> None:
    """Step and play controls plus the export of the window on screen."""
    cols = st.columns([1, 1, 1.2, 1, 1, 3.2])
    steps = [("−1 d", -24), ("−1 h", -1), None, ("+1 h", 1), ("+1 d", 24)]
    for col, step in zip(cols, steps):
        if step is None:
            playing = st.session_state.get("playing", False)
            if col.button(tr("Pause") if playing else tr("Play"), key="play",
                          use_container_width=True):
                st.session_state["playing"] = not playing
                st.rerun()
        elif col.button(step[0], key=f"step{step[1]}", use_container_width=True):
            st.session_state["cursor"] = snap(station, when + pd.Timedelta(hours=step[1]))
            st.session_state["playing"] = False
            st.rerun()
    win = window_rows(station, when, 72)
    keep = [c for c in ("station_id", "timestamp", "valid_time", "pm2_5", "pm2_5_mean_24h", "reg",
                        "cal_lo", "cal_hi", RULES[rule]["column"], "y_true") if c in win]
    cols[5].download_button(
        tr("Download this window (CSV)"), win[keep].to_csv(index=False).encode("utf-8"),
        file_name=f"hazeguard_{station['station_id'].iloc[0]}_{when:%Y%m%d_%H}.csv",
        mime="text/csv", use_container_width=True)


def autoplay(station: pd.DataFrame, when: pd.Timestamp) -> None:
    """Advance one hour per tick while Play is on; stops at the end of the period."""
    if not st.session_state.get("playing", False):
        return
    later = station[station["timestamp"] > when]
    if not len(later):
        st.session_state["playing"] = False
        return
    time.sleep(0.9)
    st.session_state["cursor"] = later["timestamp"].iloc[0]
    st.rerun()


def lead_tag(lead) -> str:
    if lead is None:
        return tr("Missed")
    if lead > 0:
        return L(f"Alert {lead} h ahead", f"提前 {lead} 小时报警")
    return tr("Alert at onset") if lead == 0 else tr("Alert after onset")


def fig_spark(series: pd.DataFrame, onset: pd.Timestamp):
    fig, ax = plt.subplots(figsize=(2.6, 0.85), dpi=150)
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)
    ax.plot(series["timestamp"], series["pm2_5"], color=BLUE, linewidth=1.6)
    ax.axhline(LEVEL_THRESHOLDS[1], color=AXIS, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.axvline(onset, color=INK_2, linewidth=0.8)
    ax.set_ylim(0, max(float(series["pm2_5"].max()) * 1.1, 60))
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
    return fig


def episode_cards(station: pd.DataFrame, events: list[dict], rule: str) -> None:
    """The six highest-peak major episodes at this node, each with a replay button."""
    major = sorted((e for e in events if e["peak_pm25"] >= LEVEL_THRESHOLDS[2]),
                   key=lambda e: -e["peak_pm25"])[:6]
    if not major or not station["pm2_5"].notna().any():
        return
    st.subheader(tr("Major episodes at this node"))
    for start in range(0, len(major), 3):
        for col, event in zip(st.columns(3), major[start:start + 3]):
            onset = pd.Timestamp(event["onset"])
            span = station[(station["timestamp"] >= onset - pd.Timedelta(hours=12))
                           & (station["timestamp"] <= onset + pd.Timedelta(
                               hours=min(event["duration_h"], 48) + 6))]
            with col.container(border=True):
                st.markdown(f"**{onset:%d %b %Y %H:%M}**")
                show(fig_spark(span, onset))
                st.caption(f"{tr('peak')} {event['peak_pm25']:.0f} µg/m³ · {event['duration_h']} h")
                st.markdown(f'<span class="hg-tag">{lead_tag(event["model_lead_h"][rule])}</span>',
                            unsafe_allow_html=True)
                if st.button(tr("Replay"), key=f"ep_{event['onset']}", use_container_width=True):
                    st.session_state["cursor"] = snap(station, onset - pd.Timedelta(hours=HORIZON_H))
                    st.session_state["playing"] = False
                    st.rerun()
    st.caption(L("The vertical line is the onset; the dashed line is the high-risk boundary. "
                 "Replay opens three hours before the onset.",
                 "竖线是事件起始时刻，虚线是高危分界。回放从起始前 3 小时开始。"))


@st.cache_data(show_spinner=False)
def load_feature_frame() -> tuple[pd.DataFrame, list[str]] | None:
    try:
        names = read_json(MODELS / "selected_features.json")["selected_features"]
        frame = pd.read_parquet(DATA / "features_final.parquet",
                                columns=["station_id", "timestamp", "split", *names])
    except Exception:
        return None
    frame["timestamp"] = _naive(frame["timestamp"])
    frame["station_id"] = frame["station_id"].astype(str)
    return frame, names


def unusual_now(station_id: str, when: pd.Timestamp) -> None:
    """Which model inputs sit in the tails of this node's training history right now.
    A description of the inputs, not an explanation of the model."""
    loaded = load_feature_frame()
    if loaded is None:
        return
    frame, names = loaded
    node = frame[frame["station_id"] == station_id]
    now = node[node["timestamp"] == when]
    history = node[node["split"] == "train"]
    if not len(now) or not len(history):
        return
    rows = []
    for name in names:
        if name.startswith(("hour_", "month_", "dow_")):
            continue                               # clock encodings are never "unusual"
        value = float(now[name].iloc[0])
        share = float((history[name] <= value).mean())
        if share >= 0.95 or share <= 0.05:
            rows.append({tr("Feature"): name, tr("Value now"): round(value, 2),
                         tr("Percentile in training history"): round(100 * share, 1),
                         "_d": abs(share - 0.5)})
    st.subheader(tr("What is unusual right now"))
    if not rows:
        st.caption(L("No input is in the top or bottom 5% of this node's training history.",
                     "当前没有输入特征落在该格点训练期历史的最高或最低 5%。"))
        return
    table = pd.DataFrame(rows).sort_values("_d", ascending=False).drop(columns="_d").head(6)
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(L("Inputs in the top or bottom 5% of this node's training-period values. This "
                 "describes the inputs; it does not say what the model did with them.",
                 "列出处于该格点训练期取值最高或最低 5% 的输入。这是对输入的描述，不代表模型如何使用它们。"))


def tab_method() -> None:
    st.subheader(tr("How the system is put together"))
    st.graphviz_chart(f"""digraph G {{
      rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded", color="{AXIS}",
      fontcolor="{INK}", fontname="Helvetica", fontsize=11]; edge [color="{INK_2}", arrowsize=0.7];
      src [label="Open-Meteo\nCAMS air quality + weather\n26 grid nodes, hourly"];
      p01 [label="01 Clean panel\nM0 persistence, M1 tree"];
      p02 [label="02 Lag, rolling and\nspatial features"];
      p03 [label="03 Top-25 features\nv1 (official) / v2 (control)"];
      v1 [label="M2 M3 M4 classifiers\nM5 regression to levels\nM6 ordinal forest\n(feature table v1)"];
      m7 [label="06 M7 cost-sensitive\nthresholds on M6"];
      m8 [label="M8 = M5 recipe\non feature table v2\n(KD-01 control)"];
      app [label="07 Dashboard\nearly-warning analysis"];
      src -> p01 -> p02 -> p03; p03 -> v1 -> m7 -> app; v1 -> app; p03 -> m8 -> app;
    }}""", use_container_width=True)
    st.caption(L("The dashboard only reads saved predictions and metrics. It never trains a "
                 "model and never tunes anything on the test split.",
                 "看板只读取已保存的预测与指标，不训练模型，也不在测试段上调任何参数。"))

    st.subheader(tr("Known limits"))
    rows = [{
        tr("Limit"): L("Model product, not measurements", "模式产品，不是实测"),
        tr("What it affects"): L("Every score is skill at predicting CAMS output for a grid cell.",
                                 "所有成绩都是预测 CAMS 格点输出的成绩。"),
        tr("Evidence"): "data/data_dictionary.json"}]
    seam = load_json_or_none("seam_report.json")
    if seam:
        hours = " / ".join(f"{h:02d}:00" for h in seam["seam_hours_local"])
        rows.append({
            tr("Limit"): L(f"Forecast-run boundary at {hours}", f"起报接缝（{hours}）"),
            tr("What it affects"): L(
                f"Hour-to-hour change is {seam['raw_seam_ratio']} times larger there; a jump "
                "across it cannot be foreseen from the inputs.",
                f"接缝处的逐小时变化是其他时段的 {seam['raw_seam_ratio']} 倍；跨接缝的跳变无法从输入预知。"),
            tr("Evidence"): "models/seam_report.json"})
    interval = load_json_or_none("metrics_stage5_interval_calibrated.json")
    if interval:
        got = interval["test"]["calibrated"]["PICP"]
        rows.append({
            tr("Limit"): L("Interval misses its target on test", "区间在测试段未达标"),
            tr("What it affects"): L(f"90% target, {got:.1%} achieved on the test split.",
                                     f"目标覆盖 90%，测试段实际 {got:.1%}。"),
            tr("Evidence"): "models/metrics_stage5_interval_calibrated.json"})
    stage7 = load_json_or_none("metrics_stage7.json")
    if stage7:
        level3 = stage7["error_anatomy"]["test"]["by_true_level"].get("3")
        if level3:
            rows.append({
                tr("Limit"): L("Concentration is under-predicted at peaks", "高峰时浓度被低估"),
                tr("What it affects"): L(
                    f"Mean error {level3['mean_error']:+.1f} µg/m³ when the outcome is level 3.",
                    f"实际为 L3 时平均误差 {level3['mean_error']:+.1f} µg/m³。"),
                tr("Evidence"): "models/metrics_stage7.json"})
    rows.append({
        tr("Limit"): L("KD-01: hour encoded with sine only (v1)", "KD-01：v1 的钟点只有正弦编码"),
        tr("What it affects"): L("24 hours collapse to 13 values. M8 on v2 measures the effect.",
                                 "24 个钟点只有 13 个取值；由 v2 上的 M8 量化影响。"),
        tr("Evidence"): "docs/known_defects.json, models/metrics_m8.json"})
    m8 = load_json_or_none("metrics_m8.json")
    if m8 and m8.get("environment"):
        env = m8["environment"]
        rows.append({
            tr("Limit"): L("M8 ran under different library versions", "M8 的运行环境与锁定版本不同"),
            tr("What it affects"): L(
                f"scikit-learn {env.get('sklearn')} instead of the pinned version; the measured "
                "difference is smaller than the seed-to-seed spread.",
                f"scikit-learn {env.get('sklearn')}，非锁定版本；实测差异小于种子间波动。"),
            tr("Evidence"): "models/runtime_environments.json"})
    rows.append({
        tr("Limit"): L("The 24 h mean is a stand-in", "24 小时均值只是替身"),
        tr("What it affects"): L("It is not the official API, so no lead over the official index "
                                 "is claimed.", "它不是官方 API，因此不声称领先官方指数多少小时。"),
        tr("Evidence"): "models/metrics_stage7.json (definitions)"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def show(fig) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def missing(path: str) -> None:
    st.warning(f"`{path}` is not available, so this part is hidden.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def preferences() -> bool:
    """Language and theme switches; returns True for dark mode."""
    st.sidebar.header(tr("Display"))
    choice = st.sidebar.radio("Language / 语言", ["English", "中文"], horizontal=True,
                              index=1 if lang() == "zh" else 0)
    wanted = "zh" if choice == "中文" else "en"
    if wanted != lang():
        st.session_state["lang"] = wanted
        st.rerun()
    return bool(st.sidebar.toggle(tr("Dark mode"), value=False, key="dark"))


def sidebar(data: pd.DataFrame, stage7: dict | None):
    st.sidebar.header(tr("Replay controls"))
    split_label = st.sidebar.radio(
        tr("Period"), ["Test (Mar to Oct 2026)", "Validation (Jul 2025 to Mar 2026)"], index=0,
        format_func=tr)
    split = "test" if split_label.startswith("Test") else "val"
    stations = sorted(data["station_id"].unique())
    station_id = st.sidebar.selectbox(tr("Receptor node"), stations, index=0,
                                      format_func=lambda s: s.replace("_", " "))
    station = data[(data["split"] == split) & (data["station_id"] == station_id)]
    station = add_running_mean(station.sort_values("timestamp").reset_index(drop=True))

    first, last = station["timestamp"].iloc[0], station["timestamp"].iloc[-1]
    ranked = station["pm2_5"] if station["pm2_5"].notna().any() else station["reg"]
    peak = station["timestamp"].iloc[int(ranked.to_numpy().argmax())]

    events = []
    if stage7:
        events = [e for e in stage7["early_warning"][split]["events"]
                  if e["station_id"] == station_id]
    wanted, picked = None, None
    if events:
        mode = st.sidebar.radio(tr("Go to"), ["High-risk episode", "Date and hour"], index=0,
                                format_func=tr)
        if mode == "High-risk episode":
            order = sorted(range(len(events)), key=lambda i: -events[i]["peak_pm25"])
            labels = {i: (f"{pd.Timestamp(events[i]['onset']):%d %b %Y %H:%M}  ·  "
                          f"{tr('peak')} {events[i]['peak_pm25']:.0f}  ·  {events[i]['duration_h']} h")
                      for i in order}
            chosen = st.sidebar.selectbox(tr("Episode (highest peak first)"), order, index=0,
                                          format_func=lambda i: labels[i])
            onset = pd.Timestamp(events[chosen]["onset"])
            wanted, picked = onset - pd.Timedelta(hours=HORIZON_H), ("episode", chosen)
    if wanted is None:
        day = st.sidebar.date_input(tr("Date"), value=peak.date(), min_value=first.date(),
                                    max_value=last.date())
        hour = st.sidebar.slider(tr("Hour (local time)"), 0, 23, int(peak.hour))
        wanted, picked = pd.Timestamp(day) + pd.Timedelta(hours=int(hour)), ("date", str(day), hour)

    # The sidebar sets the starting point; the step, play and replay buttons move a cursor
    # from there. A new sidebar choice resets the cursor.
    signature = (split, station_id, picked)
    if st.session_state.get("selection") != signature:
        st.session_state["selection"] = signature
        st.session_state["cursor"] = snap(station, wanted)
        st.session_state["playing"] = False
    when = st.session_state.get("cursor", snap(station, wanted))
    if not (station["timestamp"] == when).any():
        when = snap(station, wanted)
        st.session_state["cursor"] = when
    st.sidebar.caption(L(f"Showing {when:%d %b %Y %H:%M}.", f"当前显示 {when:%Y-%m-%d %H:%M}。"))

    rule = st.sidebar.selectbox(tr("Decision rule"), list(RULES),
                                index=list(RULES).index(DEFAULT_RULE),
                                format_func=lambda k: RULES[k]["label"])
    st.sidebar.caption(L(
        "M5: regression forest cut into levels. M6: ordinal forest. "
        "M7: M6 probabilities with cost-sensitive thresholds.",
        "M5：回归森林切档。M6：有序森林。M7：在 M6 概率上使用代价敏感阈值。"))
    st.sidebar.caption(L(
        "Replay only. The app reads saved predictions; it does not refit or retune anything.",
        "仅回放。看板读取已保存的预测，不重新训练或调参。"))
    return station, when, rule, events


def early_warning_section(stage7: dict | None, rule: str) -> None:
    st.subheader(tr("Early-warning record on the test split"))
    if not stage7:
        st.info("Run `python src/07_system_analysis.py` to produce "
                "`models/metrics_stage7.json`; this section reads it.")
        return
    summary = stage7["early_warning"]["test"]["summary"]
    mine = summary["rules"][rule]
    lag = summary["index_lag_h"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr("High-risk onsets"), f"{summary['n_onsets']}")
    c1.caption("Three receptor nodes")
    c2.metric(tr("Alert before onset"), f"{mine['share_alert_before_onset']:.0%}")
    c2.caption(f"{mine['share_full_horizon']:.0%} at the full {HORIZON_H} h")
    c3.metric(tr("24 h mean lags by"), f"{lag['median']:.0f} h" if lag else "n/a")
    c3.caption(f"Median over {lag['n']} onsets it reached" if lag else "No comparable onset")
    c4.metric(tr("Fresh alerts confirmed"),
              f"{mine['alert_precision']:.0%}" if mine["alert_precision"] is not None else "n/a")
    c4.caption(f"Of {mine['alert_onsets']} alert onsets")
    show(fig_lead_counts(mine["lead_counts"]))
    st.caption(
        f"Of {summary['n_compared']} onsets that started with the 24 h mean still below the "
        f"high-risk boundary, the mean never reached it within 48 h for "
        f"{summary['index_never_within_search']}. The 24 h mean is a stand-in for a "
        "daily-average index, not the official API.")
    breakdown = stage7["early_warning"]["test"].get("breakdown")
    if breakdown:
        boundaries = breakdown["run_boundary_hours_local"]
        st.markdown("**When the onsets happen, and how the record splits**")
        show(fig_onsets_by_hour(breakdown["onsets_by_hour"], boundaries))
        names = {"at_run_boundary": "Onset at a run-boundary hour",
                 "other_hours": "Onset at any other hour",
                 "major": "Major episodes (peak reaches level 3)",
                 "minor": "Minor episodes"}
        rows = []
        for key, label in names.items():
            block = breakdown[key]
            mine_here = block["rules"][rule]
            rows.append({"Onsets": label, "Count": block["n_onsets"],
                         "Alert before onset": mine_here["share_alert_before_onset"],
                         "Full horizon": mine_here["share_full_horizon"],
                         "Missed": mine_here["missed"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        seam = load_json_or_none("seam_report.json")
        note = ("Two CAMS forecast runs are joined at "
                + " and ".join(f"{h:02d}:00" for h in boundaries) + " local time.")
        if seam and "raw_seam_ratio" in seam:
            note += (f" At {seam.get('station', 'the reference node').replace('_', ' ')} the "
                     f"hour-to-hour PM2.5 change at those hours is {seam['raw_seam_ratio']} times "
                     "the change at other hours (models/seam_report.json).")
        st.caption(note)

    with st.expander("All decision rules and definitions"):
        rows = []
        for key, block in summary["rules"].items():
            adv = block["advantage_h"]
            rows.append({"Decision rule": RULES[key]["label"],
                         "Alert before onset": block["share_alert_before_onset"],
                         "Full horizon": block["share_full_horizon"],
                         "Missed": block["lead_counts"]["missed"],
                         "Median advantage over 24 h mean (h)": adv["median"] if adv else None,
                         "Alert onsets": block["alert_onsets"],
                         "Confirmed": block["alert_precision"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        for name in ("onset", "model_lead_h", "index_lag_h", "advantage_h", "alert_precision",
                     "run_boundary", "major"):
            if name in stage7["definitions"]:
                st.caption(f"**{name}**: {stage7['definitions'][name]}")


def tab_risk(station: pd.DataFrame, when: pd.Timestamp, rule: str, ref: dict,
             benchmark: pd.DataFrame | None, stage7: dict | None, events: list[dict]) -> None:
    row = station[station["timestamp"] == when].iloc[0]
    column = RULES[rule]["column"]
    predicted = int(row[column])
    banner(row, predicted, ref, when)
    c1, c2, c3, c4 = st.columns(4)
    if pd.notna(row["pm2_5"]):
        current = int(np.searchsorted(LEVEL_THRESHOLDS, row["pm2_5"], side="right"))
        c1.metric(tr("PM2.5 now"), f"{row['pm2_5']:.0f} µg/m³")
        c1.caption(level_label(current))
    else:
        c1.metric(tr("PM2.5 now"), "n/a")
    c2.metric(tr(f"Forecast level in {HORIZON_H} h"), level_label(predicted))
    c2.caption(RULES[rule]["label"])
    c3.metric(tr(f"Forecast PM2.5 in {HORIZON_H} h"), f"{row['reg']:.0f} µg/m³")
    if "cal_lo" in row and pd.notna(row["cal_lo"]):
        c3.caption(f"90% target interval {row['cal_lo']:.0f} to {row['cal_hi']:.0f} µg/m³")
    else:
        c3.caption(f"Tree spread {row['q05']:.0f} to {row['q95']:.0f} µg/m³")
    c4.metric(tr("What happened (replay)"), level_label(int(row["y_true"])))
    c4.caption("Forecast level was correct" if predicted == int(row["y_true"])
               else f"Forecast was off by {abs(predicted - int(row['y_true']))} level(s)")


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

    episode_cards(station, events, rule)
    unusual_now(str(row["station_id"]), when)
    early_warning_section(stage7, rule)


def tab_probability(station: pd.DataFrame, when: pd.Timestamp, preds: pd.DataFrame,
                    stage7: dict | None) -> None:
    row = station[station["timestamp"] == when].iloc[0]
    thresholds = load_thresholds()
    tau = thresholds["thresholds"]
    st.subheader(tr("Four-level probability"))
    left, right = st.columns(2)
    with left:
        show(fig_probabilities(row))
    with right:
        st.dataframe(cumulative_table(row, tau), use_container_width=True, hide_index=True)
        st.caption("M7 walks up the levels and stops at the first question that fails.")

    calibrated = interval_label()
    st.subheader(tr("Concentration and prediction interval" if calibrated
                    else "Concentration and tree spread"))
    win = window_rows(station, when, 72)
    show(fig_timeline(win, when, show_band=True))
    if calibrated:
        st.warning(calibrated[1])
    else:
        stage5 = load_json_or_none("metrics_stage5.json")
        if stage5 and "interval" in stage5:
            test = stage5["interval"]["test"]
            st.warning(
                f"The band is the spread between the trees of the M5 forest. It is not a "
                f"calibrated prediction interval: on the test split it covered "
                f"{test['PICP']:.1%} of outcomes with a mean width of {test['MPIW']:.1f} µg/m³.")

    if stage7:
        st.subheader(tr("Are the probabilities calibrated?"))
        level = st.radio("Question", [1, 2, 3], index=1, horizontal=True,
                         format_func=lambda k: f"Level at least L{k}")
        blocks = {split: stage7["calibration"][split][f"level_ge_{level}"]
                  for split in ("val", "test")}
        left, right = st.columns([1, 1.2])
        with left:
            show(fig_reliability(blocks))
        with right:
            st.dataframe(pd.DataFrame([
                {"Split": name, "ECE": blocks[key]["ece"], "Brier": blocks[key]["brier"],
                 "Mean predicted": blocks[key]["mean_predicted"],
                 "Observed rate": blocks[key]["base_rate"]}
                for key, name in (("val", "Validation"), ("test", "Test"))]),
                use_container_width=True, hide_index=True)
            st.caption("A point below the dashed line means the forest gave a higher "
                       "probability than the share of hours that turned out that way.")
            with st.expander("Bins"):
                st.dataframe(pd.DataFrame(blocks["val"]["bins"]).assign(split="val"),
                             use_container_width=True, hide_index=True)
                st.dataframe(pd.DataFrame(blocks["test"]["bins"]).assign(split="test"),
                             use_container_width=True, hide_index=True)

    st.subheader(tr("Threshold trade-off (validation split only)"))
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
        "M7 selected": evaluate_predictions(val["y_true"], val["pred_m7"]),
        "M6 default": evaluate_predictions(val["y_true"], val["pred_m6"]),
    }
    table = pd.DataFrame(rows).T[list(METRICS)]
    table.insert(0, "Decision", table.index)
    st.dataframe(table.style.format({m: "{:.4f}" for m in METRICS}),
                 use_container_width=True, hide_index=True)
    st.caption(f"M7 selected: τ = ({tau[0]:.2f}, {tau[1]:.2f}, {tau[2]:.2f}), the lowest "
               "Avg_Asymmetric_Cost on the validation grid. M6 default: the most probable level.")


def tab_models(benchmark: pd.DataFrame | None, stage7: dict | None) -> None:
    if benchmark is None:
        missing("models/all_models_benchmark.csv")
        return
    if "feature_set" not in benchmark.columns:
        benchmark = benchmark.assign(feature_set="v1")
    benchmark["feature_set"] = benchmark["feature_set"].fillna("v1").astype(str)
    main_table = benchmark[benchmark["feature_set"] == "v1"]
    st.caption("Main table: every model here uses feature table v1. The v2 control M8 is "
               "shown separately below and is never ranked against these models.")
    splits = sorted(main_table["split"].unique())
    c1, c2 = st.columns(2)
    split = c1.selectbox("Split", splits, index=splits.index("test") if "test" in splits else 0)
    metric = c2.selectbox("Metric", list(METRICS), index=2)
    table = main_table[main_table["split"] == split].reset_index(drop=True)
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

    m8_section()

    m7 = load_json_or_none("m7_test_metrics.json")
    if m7:
        st.subheader(tr("Three decisions on the same M6 probabilities (test)"))
        st.dataframe(pd.DataFrame(m7["metrics"]).T[list(METRICS)].style.format("{:.4f}"),
                     use_container_width=True)
    if stage7:
        st.subheader(tr("Where the concentration forecast is off (M5, test)"))
        anatomy = stage7["error_anatomy"]["test"]
        left, right = st.columns([1, 1.3])
        with left:
            show(fig_error_by_level(anatomy["by_true_level"]))
        with right:
            names = {"n": "Rows", "mean_error": "Mean error", "mae": "MAE",
                     "persistence_mae": "Persistence MAE",
                     "share_inside_tree_spread": "Inside tree spread"}
            trend = pd.DataFrame(anatomy["by_trend"]).T[list(names)].rename(columns=names)
            trend.insert(0, "PM2.5 over the next 3 h", trend.index)
            st.dataframe(trend, use_container_width=True, hide_index=True)
            st.caption(stage7["definitions"]["error"] + " " + stage7["definitions"]["trend"]
                       + " Persistence MAE is the error of repeating the current value.")

    st.subheader(tr("Asymmetric cost matrix"))
    cost = pd.DataFrame(DEFAULT_COST_MATRIX,
                        index=[f"true {level_label(k)}" for k in range(4)],
                        columns=[f"pred {level_label(k)}" for k in range(4)])
    st.dataframe(cost, use_container_width=True)
    st.caption("Missing a high-risk hour (lower left) costs far more than a false alarm.")


def m8_section() -> None:
    """M8 against its own v1 counterpart only: same recipe, different feature table."""
    st.subheader(tr("M8 · v2 control for known defect KD-01"))
    defects = ROOT / "docs" / "known_defects.json"
    kd = None
    if defects.exists():
        items = json.loads(defects.read_text(encoding="utf-8")).get("items", [])
        kd = next((i for i in items if i.get("id") == "KD-01"), None)
    if kd:
        st.caption(f"KD-01: {kd['description']}")
    m8 = load_json_or_none("metrics_m8.json")
    if not m8:
        st.info("M8 has not been run yet (`python src/07_m8_v2_control.py`).")
        return
    reference = m8["m5_reference"]
    rows = []
    for label, block in (("M5_val_tuned · feature table v1 (main table)",
                          reference["validation_mean_std"]),
                         ("M8 · feature table v2 (control)", m8["validation_mean_std"])):
        rows.append({"Validation, mean of 3 seeds": label,
                     **{k: block[k]["mean"] for k in METRICS}})
    rows.append({"Validation, mean of 3 seeds": "Change, v2 minus v1",
                 **m8["validation_delta_vs_M5_val_tuned"]})
    frame = pd.DataFrame(rows)
    st.dataframe(frame.style.format({k: "{:.4f}" for k in METRICS}),
                 use_container_width=True, hide_index=True)
    if m8.get("test_metrics_seed42"):
        test_rows = [
            {"Test, seed 42": "M5_val_tuned · v1", **{k: reference["test_seed42"][k]
                                                      for k in METRICS}},
            {"Test, seed 42": "M8 · v2 (control)", **{k: m8["test_metrics_seed42"][k]
                                                       for k in METRICS}},
            {"Test, seed 42": "Change, v2 minus v1", **m8["test_delta_vs_M5_val_tuned"]},
        ]
        st.dataframe(pd.DataFrame(test_rows).style.format({k: "{:.4f}" for k in METRICS}),
                     use_container_width=True, hide_index=True)
    else:
        st.caption("Test pass not run yet; validation only.")
    st.caption("M8 repeats the M5 recipe (same hyperparameters, same cut-point search on "
               "validation) on feature table v2, which pairs hour_sin with hour_cos. The only "
               "comparison made here is M8 against M5; the other models were trained on v1.")


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

    st.subheader(tr("Leave-one-node-out"))
    loso = load_table("loso_results.csv")
    if loso is None:
        missing("models/loso_results.csv")
    else:
        st.caption(f"Model in this experiment: {', '.join(loso['model'].unique())}. "
                   "Nodes without any high-risk hour have no recall value.")
        show(fig_loso(loso))
        with st.expander("Table view"):
            st.dataframe(loso, use_container_width=True, hide_index=True)

    st.subheader(tr("Seasonal slices of the 2026 hold-out"))
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
    dark = preferences()
    sync_native_theme(dark)
    apply_palette(dark)
    inject_css()
    st.title("HazeGuard-KL")
    st.caption(L(f"Greater Kuala Lumpur PM2.5 risk level, {HORIZON_H} hours ahead. "
                 "Inputs are CAMS model products from Open-Meteo, not ground measurements.",
                 f"大吉隆坡 PM2.5 风险等级，提前 {HORIZON_H} 小时。"
                 "输入为 Open-Meteo 提供的 CAMS 模式产品，不是地面实测。"))

    preds = load_predictions()
    obs = load_observations()
    data = joined(preds, obs)
    if obs is None:
        st.warning("`data/features_final.parquet` could not be read, so current PM2.5 and "
                   "the realised concentration are hidden. Forecast levels still work.")
    problem = self_check(preds)
    if problem:
        st.error(problem)

    stage7 = load_json_or_none("metrics_stage7.json")
    station, when, rule, events = sidebar(data, stage7)
    benchmark = load_table("all_models_benchmark.csv")
    ref = reliability_reference(preds)
    playback_bar(station, when, rule)

    t1, t2, t3, t4, t5 = st.tabs([
        tr(f"Risk now and in {HORIZON_H} h"), tr("Probability and interval"),
        tr("Model comparison"), tr("Explanation and generalisation"), tr("Method and limits")])
    with t1:
        tab_risk(station, when, rule, ref, benchmark, stage7, events)
    with t2:
        tab_probability(station, when, preds, stage7)
    with t3:
        tab_models(benchmark, stage7)
    with t4:
        tab_explain()
    with t5:
        tab_method()
    autoplay(station, when)


if __name__ == "__main__":
    main()
