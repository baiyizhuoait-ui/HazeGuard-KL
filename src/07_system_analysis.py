#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
System-level analysis for module 07. Run from the repository root:

    python src/07_system_analysis.py

It reads predictions that modules 05 and 06 already saved and writes
models/metrics_stage7.json. Nothing is trained, tuned or selected here, and the
six shared metrics in src/utils_metrics.py are not redefined. The three blocks
are descriptive:

1. early_warning   How many hours before a high-risk onset a correct alert was
                   available, against an index built on a 24 h running mean.
2. calibration     Whether the M6 cumulative probabilities match observed
                   frequencies (reliability bins, Brier score, ECE).
3. error_anatomy   Where the M5 concentration forecast is biased: by true level
                   and by whether PM2.5 was rising or falling.

Definitions are stored in the output file next to the numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import HORIZON_H, LEVEL_THRESHOLDS  # noqa: E402

MODELS = ROOT / "models"
DATA = ROOT / "data"
OUTPUT = MODELS / "metrics_stage7.json"

HIGH = float(LEVEL_THRESHOLDS[1])     # level 2 starts here (35 µg/m³)
QUIET_H = 6                           # hours below HIGH that separate two episodes
SEARCH_AFTER_H = 12                   # how long after onset a late alert still counts
INDEX_WINDOW_H = 24
INDEX_MIN_HOURS = 18
INDEX_SEARCH_H = 48
TREND_BAND = 10.0                     # µg/m³ change over the horizon that counts as a move
MAJOR = float(LEVEL_THRESHOLDS[2])    # an episode is "major" when its peak reaches level 3
DEFAULT_SEAM_HOURS = (8, 20)          # CAMS run boundaries in local time, see QN-002
RULE_COLUMNS = {
    "M5_val_tuned": "pred_m5_tuned",
    "M5_fixed": "pred_m5_fixed",
    "M6": "pred_m6",
    "M7": "pred_m7",
}

DEFINITIONS = {
    "onset": f"First hour with observed PM2.5 >= {HIGH:g} after at least {QUIET_H} "
             f"consecutive observed hours below {HIGH:g}, per receptor node.",
    "model_lead_h": f"Onset time minus the issue time of the earliest forecast that predicts "
                    f"level >= 2 for the onset hour or later. Only forecasts whose valid time "
                    f"is at or after the onset count, so the maximum is {HORIZON_H}. Negative "
                    f"means the alert came after the onset. 'missed' means no such forecast "
                    f"before the episode ended (or within {SEARCH_AFTER_H} h of the onset, "
                    f"whichever is earlier).",
    "index_lag_h": f"Hours after the onset until the {INDEX_WINDOW_H} h running mean of observed "
                   f"PM2.5 first reaches {HIGH:g} (needs {INDEX_MIN_HOURS} observed hours). "
                   f"'never' means it did not within {INDEX_SEARCH_H} h. Onsets where the "
                   f"running mean was already at or above {HIGH:g} one hour earlier are counted "
                   f"separately as index_already_elevated and left out of the comparison.",
    "advantage_h": "model_lead_h plus index_lag_h, for onsets where both exist.",
    "alert_precision": "Share of forecast alert onsets (a predicted level >= 2 after "
                       f"{QUIET_H} predicted hours below 2) with an observed high-risk hour "
                       "between one hour before and three hours after the predicted hour.",
    "index_note": "The running mean stands in for an index built on a daily average. It uses "
                  "the same CAMS series and the project's own 35 µg/m³ boundary. It is not the "
                  "official Malaysian API.",
    "run_boundary": "Onsets are split by whether the onset hour is a CAMS run boundary in "
                    "local time. The hours are read from models/seam_report.json "
                    "(seam_hours_local) when that file exists, otherwise 08:00 and 20:00.",
    "major": f"An episode is major when its peak observed PM2.5 reaches {MAJOR:g} (level 3).",
    "calibration": "Ten equal-width bins of the M6 cumulative probability P(level >= k). "
                   "ECE is the sample-weighted mean absolute gap between mean predicted "
                   "probability and observed frequency.",
    "error": "M5 concentration forecast minus realised concentration, µg/m³. Negative means "
             "the forecast was too low.",
    "trend": f"rising: realised concentration exceeds current PM2.5 by more than {TREND_BAND:g}; "
             f"falling: is lower by more than {TREND_BAND:g}; otherwise steady.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_thresholds(proba: np.ndarray, tau) -> np.ndarray:
    """Same sequential rule as predict_with_thresholds in src/06_cost_thresholds.py."""
    q1, q2, q3 = proba[:, 1:].sum(axis=1), proba[:, 2:].sum(axis=1), proba[:, 3]
    pass1 = q1 >= tau[0]
    pass2 = pass1 & (q2 >= tau[1])
    pass3 = pass2 & (q3 >= tau[2])
    return pass1.astype(int) + pass2.astype(int) + pass3.astype(int)


def load_frame() -> pd.DataFrame:
    probs = np.load(MODELS / "ordinal_probs_val_test.npz")
    stage5 = np.load(MODELS / "stage5_predictions.npz")
    tau = json.loads((MODELS / "cost_thresholds.json").read_text(encoding="utf-8"))["thresholds"]
    frames = []
    for split in ("val", "test"):
        proba = probs[f"proba_{split}"]
        quant = stage5[f"quantiles_{split}"]
        frames.append(pd.DataFrame({
            "split": split,
            "station_id": probs[f"station_id_{split}"].astype(str),
            "timestamp": pd.to_datetime(probs[f"ts_{split}"]),
            "y_true": probs[f"y_{split}"].astype(int),
            "p0": proba[:, 0], "p1": proba[:, 1], "p2": proba[:, 2], "p3": proba[:, 3],
            "reg": stage5[f"reg_{split}"],
            "q05": quant[:, 0], "q95": quant[:, 2],
            "pred_m5_fixed": stage5[f"pred_m5_fixed_{split}"].astype(int),
            "pred_m5_tuned": stage5[f"pred_m5_tuned_{split}"].astype(int),
            "pred_m6": stage5[f"pred_m6_{split}"].astype(int),
            "pred_m7": apply_thresholds(proba, tau),
        }))
    preds = pd.concat(frames, ignore_index=True)
    obs = pd.read_parquet(DATA / "features_final.parquet",
                          columns=["station_id", "timestamp", "pm2_5", "target_pm25_3h"])
    obs["timestamp"] = pd.to_datetime(obs["timestamp"])
    if getattr(obs["timestamp"].dt, "tz", None) is not None:
        obs["timestamp"] = obs["timestamp"].dt.tz_localize(None)
    obs["station_id"] = obs["station_id"].astype(str)
    frame = preds.merge(obs, on=["station_id", "timestamp"], how="left", validate="one_to_one")
    missing = int(frame["pm2_5"].isna().sum())
    if missing:
        raise ValueError(f"{missing} prediction rows have no match in features_final.parquet")
    return frame


# ---------------------------------------------------------------------------
# 1. Early warning
# ---------------------------------------------------------------------------
def hourly(station: pd.DataFrame) -> pd.DataFrame:
    """Reindex one station to a gap-free hourly grid; missing hours stay NaN."""
    station = station.set_index("timestamp").sort_index()
    grid = pd.date_range(station.index.min(), station.index.max(), freq="h")
    return station.reindex(grid)


def find_onsets(observed: np.ndarray) -> list[int]:
    onsets = []
    for i in range(QUIET_H, len(observed)):
        if not observed[i] >= HIGH:
            continue
        before = observed[i - QUIET_H:i]
        if np.isfinite(before).all() and (before < HIGH).all():
            onsets.append(i)
    return onsets


def station_events(grid: pd.DataFrame, station_id: str) -> list[dict]:
    observed = grid["pm2_5"].to_numpy(dtype=float)
    running = (grid["pm2_5"].rolling(INDEX_WINDOW_H, min_periods=INDEX_MIN_HOURS)
               .mean().to_numpy(dtype=float))
    times = grid.index
    n = len(grid)
    events = []
    for onset in find_onsets(observed):
        end = onset
        while end + 1 < n and observed[end + 1] >= HIGH:
            end += 1
        event = {
            "station_id": station_id,
            "onset": times[onset].isoformat(),
            "duration_h": int(end - onset + 1),
            "peak_pm25": round(float(np.nanmax(observed[onset:end + 1])), 1),
        }
        already = bool(np.isfinite(running[onset - 1]) and running[onset - 1] >= HIGH)
        event["index_already_elevated"] = already
        lag = None
        if not already:
            for step in range(0, INDEX_SEARCH_H + 1):
                j = onset + step
                if j < n and np.isfinite(running[j]) and running[j] >= HIGH:
                    lag = step
                    break
        event["index_lag_h"] = lag
        leads = {}
        for rule, column in RULE_COLUMNS.items():
            level = grid[column].to_numpy(dtype=float)   # NaN where no forecast was issued
            lead = None
            first = max(onset - HORIZON_H, 0)
            last = min(end, onset + SEARCH_AFTER_H, n - 1)   # late alerts count only while it lasts
            for issue in range(first, last + 1):
                if np.isfinite(level[issue]) and level[issue] >= 2:
                    lead = onset - issue
                    break
            leads[rule] = lead
        event["model_lead_h"] = leads
        events.append(event)
    return events


def alert_precision(grid: pd.DataFrame, column: str) -> dict:
    """How often a fresh forecast alert is followed by an observed high-risk hour."""
    observed = grid["pm2_5"].to_numpy(dtype=float)
    level = grid[column].to_numpy(dtype=float)
    n = len(grid)
    total = confirmed = 0
    for issue in range(QUIET_H, n):
        if not (np.isfinite(level[issue]) and level[issue] >= 2):
            continue
        before = level[issue - QUIET_H:issue]
        if not (np.isfinite(before).all() and (before < 2).all()):
            continue
        valid = issue + HORIZON_H
        window = observed[max(valid - 1, 0):min(valid + 3, n - 1) + 1]
        if len(window) == 0 or not np.isfinite(window).any():
            continue
        total += 1
        confirmed += int(np.nanmax(window) >= HIGH)
    return {"alerts": total, "confirmed": confirmed}


def quartiles(values) -> dict | None:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    return {"n": int(len(values)), "p25": float(q25), "median": float(q50), "p75": float(q75),
            "mean": round(float(values.mean()), 3)}


def seam_hours() -> list[int]:
    path = MODELS / "seam_report.json"
    if path.exists():
        hours = json.loads(path.read_text(encoding="utf-8")).get("seam_hours_local")
        if hours:
            return [int(h) for h in hours]
    return list(DEFAULT_SEAM_HOURS)


def group_stats(events: list[dict]) -> dict:
    """Lead-time record of one subset of onsets, for every decision rule."""
    out = {"n_onsets": len(events), "rules": {}}
    for rule in RULE_COLUMNS:
        leads = [e["model_lead_h"][rule] for e in events]
        n = len(leads)
        out["rules"][rule] = {
            "share_alert_before_onset": round(sum(v is not None and v > 0 for v in leads) / n, 4)
            if n else None,
            "share_full_horizon": round(sum(v == HORIZON_H for v in leads) / n, 4) if n else None,
            "missed": sum(v is None for v in leads),
        }
    return out


def onset_breakdown(events: list[dict], boundary_hours: list[int]) -> dict:
    hour_of = [pd.Timestamp(e["onset"]).hour for e in events]
    at_boundary = [e for e, h in zip(events, hour_of) if h in boundary_hours]
    elsewhere = [e for e, h in zip(events, hour_of) if h not in boundary_hours]
    major = [e for e in events if e["peak_pm25"] >= MAJOR]
    minor = [e for e in events if e["peak_pm25"] < MAJOR]
    return {
        "run_boundary_hours_local": boundary_hours,
        "onsets_by_hour": {str(h): hour_of.count(h) for h in range(24)},
        "at_run_boundary": group_stats(at_boundary),
        "other_hours": group_stats(elsewhere),
        "major": group_stats(major),
        "minor": group_stats(minor),
    }


def summarise_events(events: list[dict], precision: dict) -> dict:
    compared = [e for e in events if not e["index_already_elevated"]]
    lags = [e["index_lag_h"] for e in compared if e["index_lag_h"] is not None]
    summary = {
        "n_onsets": len(events),
        "n_index_already_elevated": len(events) - len(compared),
        "n_compared": len(compared),
        "index_lag_h": quartiles(lags),
        "index_never_within_search": sum(e["index_lag_h"] is None for e in compared),
        "rules": {},
    }
    for rule in RULE_COLUMNS:
        leads = [e["model_lead_h"][rule] for e in events]
        found = [v for v in leads if v is not None]
        counts = {str(h): sum(v == h for v in found) for h in range(HORIZON_H, -1, -1)}
        counts["late"] = sum(v < 0 for v in found)
        counts["missed"] = sum(v is None for v in leads)
        both = [e["model_lead_h"][rule] + e["index_lag_h"] for e in compared
                if e["model_lead_h"][rule] is not None and e["index_lag_h"] is not None]
        stats = precision[rule]
        summary["rules"][rule] = {
            "lead_counts": counts,
            "share_alert_before_onset": round(sum(v > 0 for v in found) / len(leads), 4)
            if leads else None,
            "share_full_horizon": round(sum(v == HORIZON_H for v in found) / len(leads), 4)
            if leads else None,
            "model_lead_h": quartiles(found),
            "advantage_h": quartiles(both),
            "alert_onsets": stats["alerts"],
            "alert_precision": round(stats["confirmed"] / stats["alerts"], 4)
            if stats["alerts"] else None,
        }
    return summary


def early_warning(frame: pd.DataFrame) -> dict:
    out = {}
    for split, part in frame.groupby("split"):
        events = []
        precision = {rule: {"alerts": 0, "confirmed": 0} for rule in RULE_COLUMNS}
        for station_id, station in part.groupby("station_id"):
            grid = hourly(station)
            events.extend(station_events(grid, station_id))
            for rule, column in RULE_COLUMNS.items():
                stats = alert_precision(grid, column)
                precision[rule]["alerts"] += stats["alerts"]
                precision[rule]["confirmed"] += stats["confirmed"]
        events.sort(key=lambda e: (e["onset"], e["station_id"]))
        out[split] = {"summary": summarise_events(events, precision),
                      "breakdown": onset_breakdown(events, seam_hours()),
                      "events": events}
    return out


# ---------------------------------------------------------------------------
# 2. Calibration
# ---------------------------------------------------------------------------
def reliability(probability: np.ndarray, outcome: np.ndarray, bins: int = 10) -> dict:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        mask = index == b
        count = int(mask.sum())
        if count == 0:
            rows.append({"lo": round(float(edges[b]), 2), "hi": round(float(edges[b + 1]), 2), "n": 0,
                         "mean_predicted": None, "observed_frequency": None})
            continue
        predicted = float(probability[mask].mean())
        frequency = float(outcome[mask].mean())
        ece += count / len(probability) * abs(predicted - frequency)
        rows.append({"lo": round(float(edges[b]), 2), "hi": round(float(edges[b + 1]), 2), "n": count,
                     "mean_predicted": round(predicted, 4),
                     "observed_frequency": round(frequency, 4)})
    return {
        "bins": rows,
        "ece": round(ece, 4),
        "brier": round(float(np.mean((probability - outcome) ** 2)), 4),
        "base_rate": round(float(outcome.mean()), 4),
        "mean_predicted": round(float(probability.mean()), 4),
    }


def calibration(frame: pd.DataFrame) -> dict:
    out = {}
    for split, part in frame.groupby("split"):
        proba = part[["p0", "p1", "p2", "p3"]].to_numpy()
        truth = part["y_true"].to_numpy()
        out[split] = {f"level_ge_{k}": reliability(proba[:, k:].sum(axis=1),
                                                   (truth >= k).astype(float))
                      for k in (1, 2, 3)}
    return out


# ---------------------------------------------------------------------------
# 3. Error anatomy of the concentration forecast
# ---------------------------------------------------------------------------
def error_block(part: pd.DataFrame) -> dict:
    error = part["reg"] - part["target_pm25_3h"]
    inside = (part["target_pm25_3h"] >= part["q05"]) & (part["target_pm25_3h"] <= part["q95"])
    persistence = part["pm2_5"] - part["target_pm25_3h"]
    return {
        "n": int(len(part)),
        "mean_error": round(float(error.mean()), 2),
        "mae": round(float(error.abs().mean()), 2),
        "rmse": round(float(np.sqrt((error ** 2).mean())), 2),
        "persistence_mae": round(float(persistence.abs().mean()), 2),
        "share_inside_tree_spread": round(float(inside.mean()), 4),
    }


def error_anatomy(frame: pd.DataFrame) -> dict:
    out = {}
    for split, part in frame.groupby("split"):
        change = part["target_pm25_3h"] - part["pm2_5"]
        trend = np.where(change > TREND_BAND, "rising",
                         np.where(change < -TREND_BAND, "falling", "steady"))
        block = {"overall": error_block(part), "by_true_level": {}, "by_trend": {},
                 "by_station": {}}
        for level in range(4):
            rows = part[part["y_true"] == level]
            if len(rows):
                block["by_true_level"][str(level)] = error_block(rows)
        for name in ("rising", "steady", "falling"):
            rows = part[trend == name]
            if len(rows):
                block["by_trend"][name] = error_block(rows)
        for station_id, rows in part.groupby("station_id"):
            block["by_station"][station_id] = error_block(rows)
        out[split] = block
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Module 07 system-level analysis")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()

    frame = load_frame()
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    result = {
        "module": "07 system analysis",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "code_sha256": sha256(Path(__file__)),
        "input_sha256": {name: sha256(path) for name, path in {
            "ordinal_probs_val_test": MODELS / "ordinal_probs_val_test.npz",
            "stage5_predictions": MODELS / "stage5_predictions.npz",
            "cost_thresholds": MODELS / "cost_thresholds.json",
            "features_final": DATA / "features_final.parquet",
            **({"seam_report": MODELS / "seam_report.json"}
               if (MODELS / "seam_report.json").exists() else {}),
        }.items()},
        "rows_by_split": {k: int(v) for k, v in frame["split"].value_counts().items()},
        "note": "Descriptive statistics of saved predictions. No model, threshold or feature "
                "was fitted or chosen with these numbers.",
        "definitions": DEFINITIONS,
        "early_warning": early_warning(frame),
        "calibration": calibration(frame),
        "error_anatomy": error_anatomy(frame),
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    for split in ("val", "test"):
        summary = result["early_warning"][split]["summary"]
        rule = summary["rules"]["M5_val_tuned"]
        print(f"[{split}] onsets {summary['n_onsets']}, compared {summary['n_compared']}, "
              f"index lag {summary['index_lag_h']}")
        print(f"        M5 tuned lead counts {rule['lead_counts']}, "
              f"advantage {rule['advantage_h']}, alert precision {rule['alert_precision']}")
        split_up = result["early_warning"][split]["breakdown"]
        for name in ("at_run_boundary", "other_hours", "major"):
            block = split_up[name]
            print(f"        {name}: {block['n_onsets']} onsets, M5 tuned "
                  f"{block['rules']['M5_val_tuned']}")
        cal = result["calibration"][split]["level_ge_2"]
        print(f"        P(level>=2): ECE {cal['ece']}, Brier {cal['brier']}")
        err = result["error_anatomy"][split]
        print("        mean error by true level: "
              + ", ".join(f"L{k} {v['mean_error']:+.1f}" for k, v in err["by_true_level"].items()))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
