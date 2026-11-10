#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train stage 04 nominal RF, balanced RF, and boosting models.

The training domain is the three receptor grid cells only. Hyperparameters are
searched on chronological train folds, selected on validation data, and the
test split is evaluated only after every choice is fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit

import config as C
from utils_metrics import DEFAULT_COST_MATRIX, evaluate_predictions


os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


HERE = Path(__file__).resolve().parents[1]
FEATURE_PATH = HERE / "data" / "features_final.parquet"
SELECTION_PATH = HERE / "models" / "selected_features.json"
MODEL_DIR = HERE / "models"
FIGURE_DIR = HERE / "figures"
METRICS_PATH = MODEL_DIR / "metrics_stage4.json"

SEEDS = (42, 43, 44)
LABELS = (0, 1, 2, 3)
METRIC_KEYS = (
    "Macro_F1",
    "Balanced_Accuracy",
    "High_Risk_Recall",
    "MAOE",
    "Severe_Jumping_Error_Rate",
    "Avg_Asymmetric_Cost",
    "n_samples",
    "n_high_risk",
)
RF_SPACE: dict[str, list[Any]] = {
    "n_estimators": [100, 200, 300],
    "max_depth": [6, 10, 14, 18, None],
    "min_samples_leaf": [2, 5, 10, 20],
    "max_features": ["sqrt", 0.5, 0.8],
}
BOOST_SPACE: dict[str, list[Any]] = {
    "learning_rate": [0.03, 0.06, 0.1],
    "max_iter": [200, 400],
    "max_leaf_nodes": [15, 31],
    "min_samples_leaf": [20, 50],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=FEATURE_PATH)
    parser.add_argument("--features-json", type=Path, default=SELECTION_PATH)
    parser.add_argument("--models-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--metrics-json", type=Path, default=METRICS_PATH)
    parser.add_argument(
        "--rf-candidates",
        type=int,
        default=12,
        help="Number of RF configurations sampled from the registered space.",
    )
    parser.add_argument(
        "--boost-candidates",
        type=int,
        default=24,
        help="Number of boosting configurations; 24 evaluates the full grid.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
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


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def selected_features(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("selected_features")
    if not isinstance(features, list) or not features:
        raise ValueError(f"{path} must contain a non-empty selected_features list")
    if not all(isinstance(name, str) and name for name in features):
        raise ValueError("selected_features contains an invalid column name")
    if len(features) != len(set(features)):
        raise ValueError("selected_features contains duplicates")
    return features


def load_and_validate(
    data_path: Path,
    selection_path: Path,
    check_contract_counts: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    features = selected_features(selection_path)
    frame = pd.read_parquet(data_path)
    required = set(features) | {
        "station_id",
        "timestamp",
        "role",
        "split",
        "target_level_3h",
        "pm2_5",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{data_path} is missing required columns: {missing}")

    frame = frame.loc[frame["role"].eq("receptor")].copy()
    if frame.empty:
        raise ValueError("no receptor rows found; check the feature table")

    unknown_splits = sorted(
        set(frame["split"].dropna().astype(str)) - {"train", "val", "test"}
    )
    if unknown_splits:
        raise ValueError(f"unexpected split labels: {unknown_splits}")

    if check_contract_counts:
        expected = {
            "train": C.PANEL_CONTRACT["receptor_train_eligible_rows"],
            "val": C.PANEL_CONTRACT["receptor_val_eligible_rows"],
            "test": C.PANEL_CONTRACT["receptor_test_eligible_rows"],
        }
        observed = {
            name: int(frame["split"].eq(name).sum()) for name in expected
        }
        if observed != expected:
            raise ValueError(
                f"receptor eligible rows = {observed}, contract expects {expected}; "
                "ask 1号 to register an intentional change before training"
            )

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values(
        ["timestamp", "station_id"], kind="mergesort"
    ).reset_index(drop=True)

    numeric_features = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
    values = numeric_features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = [
            name
            for name in features
            if not np.isfinite(numeric_features[name].to_numpy(dtype=float)).all()
        ]
        raise ValueError(f"Top-25 features contain NaN or infinity: {bad}")
    frame.loc[:, features] = numeric_features

    levels = pd.to_numeric(frame["target_level_3h"], errors="coerce")
    if levels.isna().any():
        raise ValueError("target_level_3h contains missing or non-numeric values")
    frame["target_level_3h"] = levels.astype(int)
    invalid_levels = sorted(set(frame["target_level_3h"]) - set(LABELS))
    if invalid_levels:
        raise ValueError(f"target_level_3h contains invalid levels: {invalid_levels}")
    return frame, features


def split_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    part = frame.loc[frame["split"].eq(name)].copy()
    if part.empty:
        raise ValueError(f"split {name!r} is empty")
    return part.reset_index(drop=True)


def chronological_splits(
    timestamps: Sequence[Any], n_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Apply TimeSeriesSplit to unique timestamps, then map folds back to rows."""
    values = pd.to_datetime(pd.Series(timestamps), errors="raise").to_numpy()
    unique = np.unique(values)
    if len(unique) <= n_splits:
        raise ValueError(f"need more than {n_splits} unique timestamps for CV")
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_time_index, valid_time_index in TimeSeriesSplit(
        n_splits=n_splits
    ).split(unique):
        train_times = unique[train_time_index]
        valid_times = unique[valid_time_index]
        train_rows = np.flatnonzero(np.isin(values, train_times))
        valid_rows = np.flatnonzero(np.isin(values, valid_times))
        folds.append((train_rows, valid_rows))
    return folds


def aligned_probabilities(model: Any, x: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=float)
    output = np.zeros((len(x), 4), dtype=float)
    for index, label in enumerate(np.asarray(model.classes_, dtype=int)):
        output[:, label] = raw[:, index]
    return output


def metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    result = evaluate_predictions(y_true, y_pred, y_prob, DEFAULT_COST_MATRIX)
    return {key: result[key] for key in METRIC_KEYS}


def persistence_predictions(frame: pd.DataFrame) -> np.ndarray:
    return np.digitize(
        frame["pm2_5"].to_numpy(dtype=float), np.asarray(C.LEVEL_THRESHOLDS)
    ).astype(int)


def rf_candidates(count: int) -> list[dict[str, Any]]:
    total = math.prod(len(values) for values in RF_SPACE.values())
    if not 1 <= count <= total:
        raise ValueError(f"rf-candidates must be in [1, {total}]")
    return list(ParameterSampler(RF_SPACE, n_iter=count, random_state=C.RANDOM_STATE))


def boost_candidates(count: int) -> list[dict[str, Any]]:
    names = list(BOOST_SPACE)
    all_candidates = [
        dict(zip(names, values))
        for values in product(*(BOOST_SPACE[name] for name in names))
    ]
    if not 1 <= count <= len(all_candidates):
        raise ValueError(
            f"boost-candidates must be in [1, {len(all_candidates)}]"
        )
    if count == len(all_candidates):
        return all_candidates
    sampled = ParameterSampler(
        BOOST_SPACE, n_iter=count, random_state=C.RANDOM_STATE
    )
    return list(sampled)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(LABELS),
            average="macro",
            zero_division=0,
        )
    )


def depth_tiebreak(params: Mapping[str, Any], key: str) -> float:
    value = params.get(key)
    return float("inf") if value is None else float(value)


def search_rf(
    x_train: np.ndarray,
    y_train: np.ndarray,
    timestamps: Sequence[Any],
    x_val: np.ndarray,
    y_val: np.ndarray,
    candidate_count: int,
    n_jobs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folds = chronological_splits(timestamps)
    trace: list[dict[str, Any]] = []
    for number, params in enumerate(rf_candidates(candidate_count), start=1):
        started = time.perf_counter()
        fold_scores: list[float] = []
        for train_index, valid_index in folds:
            model = RandomForestClassifier(
                **params,
                class_weight=None,
                random_state=C.RANDOM_STATE,
                n_jobs=n_jobs,
            )
            model.fit(x_train[train_index], y_train[train_index])
            fold_scores.append(
                macro_f1(y_train[valid_index], model.predict(x_train[valid_index]))
            )

        validation_model = RandomForestClassifier(
            **params,
            class_weight=None,
            random_state=C.RANDOM_STATE,
            n_jobs=n_jobs,
        ).fit(x_train, y_train)
        validation_score = macro_f1(y_val, validation_model.predict(x_val))
        row = {
            "candidate": number,
            "params": json_value(params),
            "cv_fold_macro_f1": [round(score, 6) for score in fold_scores],
            "cv_mean_macro_f1": round(float(np.mean(fold_scores)), 6),
            "cv_std_macro_f1": round(float(np.std(fold_scores)), 6),
            "val_macro_f1": round(validation_score, 6),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        trace.append(row)
        print(
            f"RF {number}/{candidate_count}: "
            f"cv={row['cv_mean_macro_f1']:.4f}, val={row['val_macro_f1']:.4f}"
        )

    best = max(
        trace,
        key=lambda row: (
            row["val_macro_f1"],
            row["cv_mean_macro_f1"],
            -depth_tiebreak(row["params"], "max_depth"),
        ),
    )
    return dict(best["params"]), trace


def search_boosting(
    x_train: np.ndarray,
    y_train: np.ndarray,
    timestamps: Sequence[Any],
    x_val: np.ndarray,
    y_val: np.ndarray,
    candidate_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folds = chronological_splits(timestamps)
    candidates = boost_candidates(candidate_count)
    trace: list[dict[str, Any]] = []
    for number, params in enumerate(candidates, start=1):
        started = time.perf_counter()
        fold_scores: list[float] = []
        for train_index, valid_index in folds:
            model = HistGradientBoostingClassifier(
                **params,
                random_state=C.RANDOM_STATE,
                early_stopping=False,
            )
            model.fit(x_train[train_index], y_train[train_index])
            fold_scores.append(
                macro_f1(y_train[valid_index], model.predict(x_train[valid_index]))
            )

        validation_model = HistGradientBoostingClassifier(
            **params,
            random_state=C.RANDOM_STATE,
            early_stopping=False,
        ).fit(x_train, y_train)
        validation_score = macro_f1(y_val, validation_model.predict(x_val))
        row = {
            "candidate": number,
            "params": json_value(params),
            "cv_fold_macro_f1": [round(score, 6) for score in fold_scores],
            "cv_mean_macro_f1": round(float(np.mean(fold_scores)), 6),
            "cv_std_macro_f1": round(float(np.std(fold_scores)), 6),
            "val_macro_f1": round(validation_score, 6),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        trace.append(row)
        print(
            f"Boosting {number}/{candidate_count}: "
            f"cv={row['cv_mean_macro_f1']:.4f}, val={row['val_macro_f1']:.4f}"
        )

    best = max(
        trace,
        key=lambda row: (
            row["val_macro_f1"],
            row["cv_mean_macro_f1"],
            -depth_tiebreak(row["params"], "max_leaf_nodes"),
        ),
    )
    return dict(best["params"]), trace


def fit_model_set(
    x_train: np.ndarray,
    y_train: np.ndarray,
    rf_params: Mapping[str, Any],
    boost_params: Mapping[str, Any],
    seed: int,
    n_jobs: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    definitions = {
        "M2_Nominal_RF": RandomForestClassifier(
            **rf_params,
            class_weight=None,
            random_state=seed,
            n_jobs=n_jobs,
        ),
        "M3_Balanced_RF": RandomForestClassifier(
            **rf_params,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=n_jobs,
        ),
        "M4_Boosting": HistGradientBoostingClassifier(
            **boost_params,
            random_state=seed,
            early_stopping=False,
        ),
    }
    fitted: dict[str, Any] = {}
    seconds: dict[str, float] = {}
    for name, model in definitions.items():
        started = time.perf_counter()
        model.fit(x_train, y_train)
        seconds[name] = round(time.perf_counter() - started, 3)
        fitted[name] = model
    return fitted, seconds


def evaluate_models(
    models: Mapping[str, Any], x: np.ndarray, y: np.ndarray
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        prediction = np.asarray(model.predict(x), dtype=int)
        probability = aligned_probabilities(model, x)
        output[name] = metrics(y, prediction, probability)
    return output


def mean_std(
    runs: Sequence[Mapping[str, Mapping[str, Any]]]
) -> dict[str, dict[str, dict[str, float | None]]]:
    summary: dict[str, dict[str, dict[str, float | None]]] = {}
    for model_name in runs[0]:
        summary[model_name] = {}
        for metric_name in METRIC_KEYS:
            values = [run[model_name][metric_name] for run in runs]
            numeric = np.asarray(
                [float(value) for value in values if value is not None], dtype=float
            )
            summary[model_name][metric_name] = {
                "mean": round(float(numeric.mean()), 6) if len(numeric) else None,
                "std": (
                    round(float(numeric.std(ddof=1)), 6) if len(numeric) > 1 else 0.0
                ),
            }
    return summary


def plot_comparison(
    test_metrics: Mapping[str, Mapping[str, Any]],
    m0_metrics: Mapping[str, Any],
    output: Path,
) -> None:
    names = ["M0 Persistence", "M2 Nominal RF", "M3 Balanced RF", "M4 Boosting"]
    keys = ["M0", "M2_Nominal_RF", "M3_Balanced_RF", "M4_Boosting"]
    values = {"M0": m0_metrics, **test_metrics}
    panels = (
        ("Macro_F1", "Macro-F1", "higher is better"),
        ("High_Risk_Recall", "High-risk recall", "higher is better"),
        ("Avg_Asymmetric_Cost", "Average asymmetric cost", "lower is better"),
    )
    colors = ["#7F7F7F", "#1F4E79", "#5B9BD5", "#C55A11"]
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
    for axis, (metric_name, title, direction) in zip(axes, panels):
        numbers = [values[key][metric_name] for key in keys]
        bars = axis.bar(range(len(names)), numbers, color=colors)
        axis.set_title(f"{title}\n{direction}")
        axis.set_xticks(range(len(names)), names, rotation=24, ha="right")
        axis.grid(axis="y", alpha=0.25)
        for bar, number in zip(bars, numbers):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{number:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.suptitle("Stage 04 model comparison on the fixed test split")
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_model(model: Any, path: Path) -> None:
    joblib.dump(model, path, compress=("xz", 6))
    limit = 20 * 1024 * 1024
    if path.stat().st_size >= limit:
        raise ValueError(
            f"{path} is {path.stat().st_size / (1024 * 1024):.1f} MB; "
            "the repository limit for a model file is below 20 MB"
        )


def main() -> None:
    args = parse_args()
    frame, features = load_and_validate(
        args.data,
        args.features_json,
        check_contract_counts=not args.skip_contract_counts,
    )
    train = split_frame(frame, "train")
    validation = split_frame(frame, "val")
    test = split_frame(frame, "test")

    x_train = train.loc[:, features].to_numpy(dtype=float)
    y_train = train["target_level_3h"].to_numpy(dtype=int)
    x_val = validation.loc[:, features].to_numpy(dtype=float)
    y_val = validation["target_level_3h"].to_numpy(dtype=int)
    x_test = test.loc[:, features].to_numpy(dtype=float)
    y_test = test["target_level_3h"].to_numpy(dtype=int)

    input_hashes = {
        "features_final": sha256(args.data),
        "selected_features": sha256(args.features_json),
    }
    code_hash = sha256(Path(__file__))
    checkpoint_path = args.metrics_json.parent / "metrics_stage4_search.tmp"
    checkpoint_fingerprint = {
        "code_sha256": code_hash,
        "input_sha256": input_hashes,
        "rf_candidates": args.rf_candidates,
        "boost_candidates": args.boost_candidates,
    }
    checkpoint: dict[str, Any] | None = None
    if checkpoint_path.exists():
        candidate = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if candidate.get("fingerprint") == checkpoint_fingerprint:
            checkpoint = candidate

    if checkpoint is None:
        rf_params, rf_trace = search_rf(
            x_train,
            y_train,
            train["timestamp"],
            x_val,
            y_val,
            args.rf_candidates,
            args.n_jobs,
        )
        boost_params, boost_trace = search_boosting(
            x_train,
            y_train,
            train["timestamp"],
            x_val,
            y_val,
            args.boost_candidates,
        )
        checkpoint = {
            "fingerprint": checkpoint_fingerprint,
            "selected_rf_params": json_value(rf_params),
            "selected_boosting_params": json_value(boost_params),
            "rf_search_trace": rf_trace,
            "boosting_search_trace": boost_trace,
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        print(f"Reusing completed search from {checkpoint_path}")
        rf_params = dict(checkpoint["selected_rf_params"])
        boost_params = dict(checkpoint["selected_boosting_params"])
        rf_trace = list(checkpoint["rf_search_trace"])
        boost_trace = list(checkpoint["boosting_search_trace"])

    seed_runs: list[dict[str, Any]] = []
    seed_models: dict[int, dict[str, Any]] = {}
    seed_seconds: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        fitted, seconds = fit_model_set(
            x_train, y_train, rf_params, boost_params, seed, args.n_jobs
        )
        seed_models[seed] = fitted
        seed_seconds[seed] = seconds
        seed_runs.append(
            {
                "seed": seed,
                "validation_metrics": evaluate_models(fitted, x_val, y_val),
                "training_seconds": seconds,
            }
        )

    final_models = seed_models[C.RANDOM_STATE]
    final_validation = evaluate_models(final_models, x_val, y_val)
    # The test split is first touched here, after all model and parameter choices are fixed.
    final_test = evaluate_models(final_models, x_test, y_test)

    m0_val = metrics(y_val, persistence_predictions(validation))
    m0_test = metrics(y_test, persistence_predictions(test))
    stability = mean_std([run["validation_metrics"] for run in seed_runs])

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    save_model(final_models["M2_Nominal_RF"], args.models_dir / "rf_nominal.joblib")
    save_model(final_models["M3_Balanced_RF"], args.models_dir / "rf_balanced.joblib")
    save_model(final_models["M4_Boosting"], args.models_dir / "boosting.joblib")
    plot_comparison(
        final_test,
        m0_test,
        args.figures_dir / "fig_boosting_vs_rf.png",
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "code_sha256": code_hash,
        "input_sha256": input_hashes,
        "training_domain": "role == receptor",
        "feature_names": features,
        "rows_by_split": {
            "train": len(train),
            "val": len(validation),
            "test": len(test),
        },
        "random_seeds": list(SEEDS),
        "selection_rule": (
            "TimeSeriesSplit(5) on unique train timestamps; validation Macro-F1 "
            "selects the final configuration; test is evaluated after selection"
        ),
        "rf_parameter_space": RF_SPACE,
        "boosting_parameter_space": BOOST_SPACE,
        "selected_rf_params": json_value(rf_params),
        "selected_boosting_params": json_value(boost_params),
        "rf_search_trace": rf_trace,
        "boosting_search_trace": boost_trace,
        "metrics_persistence_M0_val": m0_val,
        "metrics_persistence_M0_test": m0_test,
        "models": {
            "M2_Nominal_RF": {
                "model_file": "models/rf_nominal.joblib",
                "class_weight": None,
                "hyperparameters": json_value(rf_params),
                "validation_metrics_seed42": final_validation["M2_Nominal_RF"],
                "test_metrics_seed42": final_test["M2_Nominal_RF"],
            },
            "M3_Balanced_RF": {
                "model_file": "models/rf_balanced.joblib",
                "class_weight": "balanced_subsample",
                "hyperparameters": json_value(rf_params),
                "validation_metrics_seed42": final_validation["M3_Balanced_RF"],
                "test_metrics_seed42": final_test["M3_Balanced_RF"],
            },
            "M4_Boosting": {
                "model_file": "models/boosting.joblib",
                "hyperparameters": json_value(boost_params),
                "fixed_parameters": {
                    "early_stopping": False,
                    "random_state": C.RANDOM_STATE,
                },
                "validation_metrics_seed42": final_validation["M4_Boosting"],
                "test_metrics_seed42": final_test["M4_Boosting"],
            },
        },
        "validation_seed_runs": seed_runs,
        "validation_seed_mean_std": stability,
        "test_evaluation_count": 1,
    }
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(
        json.dumps(json_value(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    checkpoint_path.unlink(missing_ok=True)
    print(f"Wrote {args.metrics_json}")
    for name, result in final_test.items():
        print(name, result)


if __name__ == "__main__":
    main()
