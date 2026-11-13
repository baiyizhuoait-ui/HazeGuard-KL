#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run stage 04 RF depth and history-window sensitivity experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, log_loss


HERE = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = HERE / "src" / "04_rf_and_boosting.py"
SPEC = importlib.util.spec_from_file_location("stage4_main", MAIN_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load {MAIN_SCRIPT}")
stage4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage4)

DEPTHS = list(range(2, 26))
WINDOWS = (3, 6, 12, 24)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=stage4.FEATURE_PATH)
    parser.add_argument("--features-json", type=Path, default=stage4.SELECTION_PATH)
    parser.add_argument("--metrics-json", type=Path, default=stage4.METRICS_PATH)
    parser.add_argument("--figures-dir", type=Path, default=stage4.FIGURE_DIR)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--latency-repeats", type=int, default=1000)
    parser.add_argument(
        "--skip-contract-counts",
        action="store_true",
        help="Only for isolated smoke tests; formal runs must not use this flag.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"M2_Nominal_RF", "M3_Balanced_RF", "M4_Boosting"}
    if not required.issubset(payload.get("models", {})):
        raise ValueError(f"{path} does not contain all stage 04 model results")
    return payload


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(stage4.LABELS),
            average="macro",
            zero_division=0,
        )
    )


def depth_scan(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    base_params: Mapping[str, Any],
    n_jobs: int,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for depth in DEPTHS:
        params = dict(base_params)
        params["max_depth"] = depth
        model = RandomForestClassifier(
            **params,
            class_weight="balanced_subsample",
            random_state=stage4.C.RANDOM_STATE,
            n_jobs=n_jobs,
        ).fit(x_train, y_train)
        train_prediction = model.predict(x_train)
        val_prediction = model.predict(x_val)
        train_probability = stage4.aligned_probabilities(model, x_train)
        val_probability = stage4.aligned_probabilities(model, x_val)
        row = {
            "max_depth": depth,
            "train_macro_f1": round(macro_f1(y_train, train_prediction), 6),
            "val_macro_f1": round(macro_f1(y_val, val_prediction), 6),
            "train_log_loss": round(
                float(log_loss(y_train, train_probability, labels=list(stage4.LABELS))),
                6,
            ),
            "val_log_loss": round(
                float(log_loss(y_val, val_probability, labels=list(stage4.LABELS))),
                6,
            ),
        }
        rows.append(row)
        print(
            f"Depth {depth}/25: train={row['train_macro_f1']:.4f}, "
            f"val={row['val_macro_f1']:.4f}"
        )
    return rows


LOOKBACK_PATTERNS = (
    re.compile(r"(?:^|_)lag_(\d+)(?:h)?(?:$|_)", re.IGNORECASE),
    re.compile(
        r"(?:mean|std|range|slope|change)_(\d+)h(?:$|_)", re.IGNORECASE
    ),
    re.compile(
        r"gran_(?:low|core|up|width|slope)_(\d+)h(?:$|_)", re.IGNORECASE
    ),
    re.compile(
        r"far_source_pm25_(?:mean|lag)_(\d+)h(?:$|_)", re.IGNORECASE
    ),
    re.compile(r"advection_(\d+)h(?:$|_)", re.IGNORECASE),
)


def feature_lookback_hours(name: str) -> int:
    matches: list[int] = []
    for pattern in LOOKBACK_PATTERNS:
        matches.extend(int(value) for value in pattern.findall(name))
    return max(matches, default=0)


def window_feature_indices(features: Sequence[str], window_hours: int) -> list[int]:
    indices = [
        index
        for index, feature in enumerate(features)
        if feature_lookback_hours(feature) <= window_hours
    ]
    if not indices:
        raise ValueError(f"no features are available for the {window_hours}h window")
    return indices


def latency(
    model: RandomForestClassifier,
    x: np.ndarray,
    repeats: int,
    warmup: int = 100,
) -> tuple[float, float]:
    if repeats < 1:
        raise ValueError("latency-repeats must be positive")
    for index in range(warmup):
        row = x[index % len(x) : index % len(x) + 1]
        model.predict(row)
    timings = np.empty(repeats, dtype=float)
    for index in range(repeats):
        row = x[index % len(x) : index % len(x) + 1]
        started = time.perf_counter_ns()
        model.predict(row)
        timings[index] = (time.perf_counter_ns() - started) / 1_000_000.0
    return float(timings.mean()), float(np.percentile(timings, 95))


def window_scan(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    features: Sequence[str],
    params: Mapping[str, Any],
    n_jobs: int,
    latency_repeats: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window in WINDOWS:
        indices = window_feature_indices(features, window)
        model = RandomForestClassifier(
            **params,
            class_weight="balanced_subsample",
            random_state=stage4.C.RANDOM_STATE,
            n_jobs=n_jobs,
        ).fit(x_train[:, indices], y_train)
        prediction = model.predict(x_val[:, indices])
        mean_ms, p95_ms = latency(
            model, x_val[:, indices], repeats=latency_repeats
        )
        row = {
            "window_hours": window,
            "feature_count": len(indices),
            "feature_names": [features[index] for index in indices],
            "validation_macro_f1": round(macro_f1(y_val, prediction), 6),
            "latency_mean_ms": round(mean_ms, 6),
            "latency_p95_ms": round(p95_ms, 6),
        }
        rows.append(row)
        print(
            f"Window {window}h: features={len(indices)}, "
            f"val={row['validation_macro_f1']:.4f}, "
            f"latency={row['latency_mean_ms']:.3f}ms"
        )
    return rows


def plot_depth(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    depths = [row["max_depth"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    axes[0].plot(
        depths,
        [row["train_macro_f1"] for row in rows],
        label="Train",
        color="#1F4E79",
    )
    axes[0].plot(
        depths,
        [row["val_macro_f1"] for row in rows],
        label="Validation",
        color="#C55A11",
    )
    axes[0].set(
        xlabel="Maximum tree depth",
        ylabel="Macro-F1",
        title="Balanced RF complexity",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        depths,
        [row["train_log_loss"] for row in rows],
        label="Train",
        color="#1F4E79",
    )
    axes[1].plot(
        depths,
        [row["val_log_loss"] for row in rows],
        label="Validation",
        color="#C55A11",
    )
    axes[1].set(
        xlabel="Maximum tree depth",
        ylabel="Log-loss",
        title="Probability fit",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_windows(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    windows = [row["window_hours"] for row in rows]
    scores = [row["validation_macro_f1"] for row in rows]
    means = [row["latency_mean_ms"] for row in rows]
    p95s = [row["latency_p95_ms"] for row in rows]
    figure, score_axis = plt.subplots(figsize=(7.8, 4.8))
    latency_axis = score_axis.twinx()
    score_axis.plot(
        windows,
        scores,
        marker="o",
        linewidth=2,
        color="#1F4E79",
        label="Validation Macro-F1",
    )
    latency_axis.plot(
        windows,
        means,
        marker="s",
        color="#C55A11",
        label="Mean latency",
    )
    latency_axis.plot(
        windows,
        p95s,
        marker="^",
        linestyle="--",
        color="#ED7D31",
        label="P95 latency",
    )
    score_axis.set(
        xlabel="Maximum history window (hours)",
        ylabel="Validation Macro-F1",
    )
    latency_axis.set_ylabel("Single-sample inference latency (ms)")
    score_axis.set_xticks(windows)
    score_axis.grid(alpha=0.25)
    lines = score_axis.get_lines() + latency_axis.get_lines()
    score_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    payload = read_metrics(args.metrics_json)
    frame, features = stage4.load_and_validate(
        args.data,
        args.features_json,
        check_contract_counts=not args.skip_contract_counts,
    )
    train = stage4.split_frame(frame, "train")
    validation = stage4.split_frame(frame, "val")
    x_train = train.loc[:, features].to_numpy(dtype=float)
    y_train = train["target_level_3h"].to_numpy(dtype=int)
    x_val = validation.loc[:, features].to_numpy(dtype=float)
    y_val = validation["target_level_3h"].to_numpy(dtype=int)
    rf_params = dict(payload["selected_rf_params"])

    depth_rows = depth_scan(
        x_train, y_train, x_val, y_val, rf_params, args.n_jobs
    )
    window_rows = window_scan(
        x_train,
        y_train,
        x_val,
        y_val,
        features,
        rf_params,
        args.n_jobs,
        args.latency_repeats,
    )

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_depth(depth_rows, args.figures_dir / "fig_rf_complexity_curve.png")
    plot_windows(window_rows, args.figures_dir / "fig_window_sensitivity.png")

    payload["complexity_experiments"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_sha256": sha256(Path(__file__)),
        "model": "M3_Balanced_RF",
        "depth_scan": depth_rows,
        "window_sensitivity": window_rows,
        "latency_protocol": {
            "warmup_predictions": 100,
            "timed_single_sample_predictions": args.latency_repeats,
            "statistics": ["mean_ms", "p95_ms"],
        },
        "test_split_used": False,
    }
    args.metrics_json.write_text(
        json.dumps(stage4.json_value(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Updated {args.metrics_json}")


if __name__ == "__main__":
    main()
