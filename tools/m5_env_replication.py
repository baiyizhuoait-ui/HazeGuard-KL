#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M8: the v2 control for known defect KD-01 (module 07). Run from the repository root.

    python src/07_m8_v2_control.py                     # step 1: train + validation only
    python src/07_m8_v2_control.py --final-test        # step 2: the single test pass
    python src/07_m8_v2_control.py --update-benchmark  # step 3: add M8 to the benchmark

What M8 is
----------
The M5 recipe, unchanged, trained on feature table v2 instead of v1. v1 keeps only
hour_sin (KD-01, docs/known_defects.json); v2 pairs it with hour_cos. Everything
else is held fixed so the difference can be attributed to the feature table:

* hyperparameters: read from models/stage5_selection.json (selected_regressor_params);
  no new search;
* data loading and checks: src/05_reg_to_cls.py::load_data, pointed at the v2 files;
* level cut points: src/05_reg_to_cls.py::search_cut_points on the validation split,
  seed 42, then fixed for seeds 43 and 44, exactly as for M5_val_tuned;
* point prediction: mean over the forest's trees, as in M5;
* metrics: src/utils_metrics.py only.

Rules
-----
* The v2 files must match the fingerprints registered in data/feature_manifest.json.
* Step 1 never reads test labels or predicts on test rows.
* Step 2 runs once. It refits seed 42, checks that the validation predictions are
  identical to step 1, predicts the test split once and records test_evaluation_count = 1.
  A second run refuses.
* Step 3 adds a feature_set column to models/all_models_benchmark.csv (v1 for the
  existing rows) and writes the M8 row from models/metrics_m8.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils_metrics import evaluate_predictions  # noqa: E402

R = importlib.import_module("05_reg_to_cls")

# Same recipe as M8, but on the v1 table: this re-runs M5 in *this* environment
# so that the M5 -> M8 delta can be read without a library-version confound.
V2_TABLE = "data/features_final.parquet"
V2_FEATURES = "models/selected_features.json"
MANIFEST = "data/feature_manifest.json"
SELECTION = "models/stage5_selection.json"
STAGE5 = "models/metrics_stage5.json"
OUT_METRICS = "models/m5_env_replication.json"
OUT_PRED = "models/m8_predictions.npz"
BENCHMARK = "models/all_models_benchmark.csv"
SEEDS = (42, 43, 44)
METRICS = ("Macro_F1", "Balanced_Accuracy", "High_Risk_Recall", "MAOE",
           "Severe_Jumping_Error_Rate", "Avg_Asymmetric_Cost")
CONTRACT_COLUMNS = ["model", "split", "seed", *METRICS, "n_samples"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                    encoding="utf-8")


def git_commit(root: Path) -> str:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def code_hashes(root: Path) -> dict:
    return {name: sha256(root / "src" / name) for name in
            ("07_m8_v2_control.py", "05_reg_to_cls.py", "utils_metrics.py", "config.py")}


def check_manifest(root: Path) -> dict:
    """The v2 table must be the one registered in data/feature_manifest.json."""
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    entry = manifest["tables"].get(V2_TABLE)
    if entry is None:
        raise SystemExit(f"{V2_TABLE} is not registered in {MANIFEST}")
    actual = sha256(root / V2_TABLE)
    if actual != entry["sha256"]:
        raise SystemExit(f"{V2_TABLE} fingerprint {actual[:12]} does not match the registered "
                         f"{entry['sha256'][:12]}; ask member 1 before training M8")
    return {"features_final_v2": actual, "selected_features_v2": sha256(root / V2_FEATURES)}


def load_v2(root: Path):
    """Run member 5's loader unchanged on copies of the v2 files."""
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        (stage / "data").mkdir()
        (stage / "models").mkdir()
        shutil.copyfile(root / V2_TABLE, stage / "data" / "features_final.parquet")
        shutil.copyfile(root / V2_FEATURES, stage / "models" / "selected_features.json")
        parts, features, _ = R.load_data(stage)
    return parts, features


def fit(params: dict, seed: int, X: np.ndarray, y: np.ndarray, n_jobs: int):
    return RandomForestRegressor(**params, random_state=seed, n_jobs=n_jobs).fit(X, y)


def mean_std(runs: list[dict]) -> dict:
    out = {}
    for key in METRICS:
        values = np.array([r[key] for r in runs], dtype=float)
        out[key] = {"mean": round(float(values.mean()), 6),
                    "std": round(float(values.std(ddof=1)), 6) if len(values) > 1 else 0.0}
    return out


def m5_reference(root: Path) -> dict:
    stage5 = json.loads((root / STAGE5).read_text(encoding="utf-8"))
    return {
        "source": STAGE5,
        "feature_set": "v1",
        "validation_mean_std": stage5["validation_mean_std"]["M5_val_tuned"],
        "test_seed42": stage5["models"]["M5_val_tuned"]["test_metrics_seed42"],
        "val_tuned_thresholds": stage5["m5_threshold_search"]["selected_thresholds"],
    }


def step_validation(root: Path, n_jobs: int) -> int:
    out = root / OUT_METRICS
    if out.exists():
        done = json.loads(out.read_text(encoding="utf-8"))
        if done.get("test_evaluation_count", 0) >= 1:
            raise SystemExit("M8 already has its test result. Nothing to redo.")
    hashes = check_manifest(root)
    params = json.loads((root / SELECTION).read_text(encoding="utf-8"))["selected_regressor_params"]
    parts, features = load_v2(root)
    train, val = parts["train"], parts["val"]
    print(f"v2 rows train {len(train):,}  val {len(val):,}  features {len(features)}; "
          f"M5 params {params}", flush=True)
    X, XV = (p[features].to_numpy(dtype=np.float32) for p in (train, val))
    y_train, yv = train.target_pm25_3h.to_numpy(), val.target_level_3h.to_numpy()

    runs, cuts, fixed_runs, val_pred_hash = [], None, [], None
    for seed in SEEDS:
        print(f"  fitting seed {seed}", flush=True)
        reg = fit(params, seed, X, y_train, n_jobs)
        pred = reg.predict(XV)
        if seed == 42:
            cuts, _ = R.search_cut_points(pred, yv)
            val_pred_hash = hashlib.sha256(np.round(pred, 6).tobytes()).hexdigest()
        runs.append(evaluate_predictions(yv, R.levels(pred, cuts)))
        fixed_runs.append(evaluate_predictions(yv, R.levels(pred)))

    reference = m5_reference(root)
    summary = mean_std(runs)
    delta = {k: round(summary[k]["mean"] - reference["validation_mean_std"][k]["mean"], 6)
             for k in METRICS}
    write_json(out, {
        "model": "M5_replication",
        "feature_set": "v1",
        "role": "replication of M5 in the environment recorded under runtime",
        "recipe": "M5_val_tuned (RandomForestRegressor, tree mean, cut points searched on "
                  "validation seed 42 and fixed for seeds 43/44)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(root),
        "code_sha256": code_hashes(root),
        "input_sha256": hashes,
        "hyperparameters": params,
        "hyperparameter_source": SELECTION,
        "features": features,
        "seeds": list(SEEDS),
        "val_tuned_thresholds": [float(c) for c in cuts],
        "validation_seed_runs": [{"seed": s, "M8": r, "M8_fixed_cuts": f}
                                 for s, r, f in zip(SEEDS, runs, fixed_runs)],
        "validation_mean_std": summary,
        "validation_delta_vs_M5_val_tuned": delta,
        "m5_reference": reference,
        "seed42_validation_prediction_sha256": val_pred_hash,
        "test_evaluation_count": 0,
        "test_metrics_seed42": None,
        "note": "Validation only. This is M5 on v1, re-run here, to separate the library-version effect from the feature-table effect. Test labels are not read.",
    })
    print("validation mean (M8, v2):  " + "  ".join(
        f"{k} {summary[k]['mean']:.4f}" for k in METRICS))
    print("change vs M5_val_tuned (v1): " + "  ".join(f"{k} {v:+.4f}" for k, v in delta.items()))
    print(f"cut points {cuts}; wrote {OUT_METRICS}")
    print("Next, once the team agrees: python src/07_m8_v2_control.py --final-test")
    return 0


def step_test(root: Path, n_jobs: int) -> int:
    path = root / OUT_METRICS
    if not path.exists():
        raise SystemExit("Run the validation step first.")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("test_evaluation_count", 0) >= 1:
        raise SystemExit("M8 has already been evaluated on test once. It will not run again.")
    hashes = check_manifest(root)
    if hashes != record["input_sha256"]:
        raise SystemExit("The v2 files changed since the validation step. Re-run step 1 first.")
    if sha256(root / "src" / "07_m8_v2_control.py") != record["code_sha256"]["07_m8_v2_control.py"]:
        raise SystemExit("This script changed since the validation step. Re-run step 1 first.")
    parts, features = load_v2(root)
    train, val, test = (parts[s] for s in ("train", "val", "test"))
    reg = fit(record["hyperparameters"], 42,
              train[features].to_numpy(dtype=np.float32), train.target_pm25_3h.to_numpy(), n_jobs)
    reg_val = reg.predict(val[features].to_numpy(dtype=np.float32))
    if hashlib.sha256(np.round(reg_val, 6).tobytes()).hexdigest() != \
            record["seed42_validation_prediction_sha256"]:
        raise SystemExit("Refitted seed 42 does not reproduce the validation predictions.")
    cuts = record["val_tuned_thresholds"]

    # mark the pass as started before predicting, so an interrupted run cannot be repeated quietly
    record["test_evaluation_count"] = 1
    record["test_started_at"] = datetime.now(timezone.utc).isoformat()
    write_json(path, record)

    reg_test = reg.predict(test[features].to_numpy(dtype=np.float32))
    yt = test.target_level_3h.to_numpy()
    pred_test = R.levels(reg_test, cuts)
    metrics = evaluate_predictions(yt, pred_test)
    np.savez_compressed(
        root / OUT_PRED,
        reg_val=reg_val, reg_test=reg_test,
        pred_m8_val=R.levels(reg_val, cuts), pred_m8_test=pred_test,
        y_val=val.target_level_3h.to_numpy(), y_test=yt,
        ts_val=val.timestamp.to_numpy(), ts_test=test.timestamp.to_numpy(),
        station_id_val=val.station_id.to_numpy(dtype=str),
        station_id_test=test.station_id.to_numpy(dtype=str))
    reference = record["m5_reference"]["test_seed42"]
    record.update({
        "test_metrics_seed42": metrics,
        "test_n_samples": int(len(yt)),
        "test_delta_vs_M5_val_tuned": {k: round(float(metrics[k]) - float(reference[k]), 6)
                                       for k in METRICS},
        "test_finished_at": datetime.now(timezone.utc).isoformat(),
        "predictions_file": OUT_PRED,
        "note": "Single test pass. Cut points and hyperparameters were fixed before it.",
    })
    write_json(path, record)
    print("test (M8, v2, seed 42): " + "  ".join(f"{k} {metrics[k]:.4f}" for k in METRICS))
    print("change vs M5_val_tuned: " + "  ".join(
        f"{k} {v:+.4f}" for k, v in record["test_delta_vs_M5_val_tuned"].items()))
    print(f"wrote {OUT_METRICS} and {OUT_PRED}")
    print("Next: python src/07_m8_v2_control.py --update-benchmark")
    return 0


def step_benchmark(root: Path) -> int:
    record = json.loads((root / OUT_METRICS).read_text(encoding="utf-8"))
    metrics = record.get("test_metrics_seed42")
    if not metrics:
        raise SystemExit("M8 has no test result yet.")
    path = root / BENCHMARK
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        header = list(rows[0].keys()) if rows else CONTRACT_COLUMNS
    if header[:len(CONTRACT_COLUMNS)] != CONTRACT_COLUMNS:
        raise SystemExit(f"unexpected benchmark columns: {header}")
    for row in rows:
        row.setdefault("feature_set", "v1")
        if not row.get("feature_set"):
            row["feature_set"] = "v1"
    rows = [r for r in rows if r["model"] != "M8"]
    rows.append({"model": "M8", "split": "test", "seed": "42",
                 **{k: f"{float(metrics[k]):.4f}" for k in METRICS},
                 "n_samples": str(record["test_n_samples"]), "feature_set": "v2"})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTRACT_COLUMNS + ["feature_set"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{BENCHMARK}: feature_set column present, M8 row written ({len(rows)} rows)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="M8: M5 recipe on feature table v2 (KD-01)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--final-test", action="store_true", help="the single test pass")
    group.add_argument("--update-benchmark", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.update_benchmark:
        return step_benchmark(args.root)
    if args.final_test:
        return step_test(args.root, args.n_jobs)
    return step_validation(args.root, args.n_jobs)


if __name__ == "__main__":
    sys.exit(main())
