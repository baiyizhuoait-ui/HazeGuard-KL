"""Q5-3：按本地发出时刻汇总 M5 浓度 MAE，只读已保存的 v1 预测。"""

import importlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

R = importlib.import_module("05_reg_to_cls")
ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("cross_08", "cross_20", "other_hours", "not_cross_20")


def summarize(timestamp, y, prediction):
    ts = pd.DatetimeIndex(timestamp)
    y, pred = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if ts.hasnans or ts.tz is not None:
        raise ValueError("naive Kuala Lumpur feature timestamps required")
    if y.shape != (len(ts),) or pred.shape != y.shape:
        raise ValueError("matching one-dimensional timestamps, targets and predictions required")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("non-finite targets or predictions")
    morning = np.isin(ts.hour, [5, 6, 7])
    evening = np.isin(ts.hour, [17, 18, 19])
    masks = (morning, evening, ~(morning | evening), ~evening)
    error = np.abs(y - pred)
    rows = []
    for group, mask in zip(GROUPS, masks):
        if not mask.any():
            raise ValueError(f"no samples in {group}")
        rows.append({"group": group, "n_samples": int(mask.sum()),
                     "MAE": float(error[mask].mean())})
    if sum(row["n_samples"] for row in rows[:3]) != len(ts):
        raise AssertionError("the three disjoint groups must partition the split")
    return rows


def main():
    md = ROOT / "models"
    csv_path = md / "m5_seam_mae.csv"
    json_path = md / "metrics_stage5_seam.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("seam outputs already exist; inspect them instead of overwriting")
    official_path = md / "metrics_stage5.json"
    official = json.loads(official_path.read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data/feature_manifest.json").read_text(encoding="utf-8"))
    parts, names, hashes = R.load_data(ROOT)
    expected = official["input_sha256"]["features_final"]
    if hashes["features_final"] != expected or manifest["tables"]["data/features_final.parquet"]["sha256"] != expected:
        raise ValueError("feature table is not the official stage-05 v1 table")
    if names != official["feature_names"]:
        raise ValueError("feature order differs from the official model")
    for name in ("stage5_predictions.npz", "ordinal_probs_val_test.npz"):
        if R.sha256(md / name) != official["artifacts"][name]:
            raise ValueError(f"original saved predictions changed: {name}")
    input_hashes = {"data/features_final.parquet": hashes["features_final"],
                    "models/selected_features.json": hashes["selected_features"],
                    "models/metrics_stage5.json": R.sha256(official_path),
                    "data/feature_manifest.json": R.sha256(ROOT / "data/feature_manifest.json")}
    for name in ("stage5_predictions.npz", "ordinal_probs_val_test.npz"):
        input_hashes[f"models/{name}"] = R.sha256(md / name)
    records, comparison = [], {}
    with np.load(md / "stage5_predictions.npz", allow_pickle=False) as saved, \
            np.load(md / "ordinal_probs_val_test.npz", allow_pickle=False) as ids:
        for split in ("val", "test"):
            frame = parts[split]
            np.testing.assert_array_equal(ids[f"ts_{split}"], frame.timestamp.to_numpy())
            np.testing.assert_array_equal(ids[f"station_id_{split}"], frame.station_id.to_numpy(dtype=str))
            np.testing.assert_array_equal(ids[f"y_{split}"], frame.target_level_3h.to_numpy())
            rows = summarize(frame.timestamp, frame.target_pm25_3h, saved[f"reg_{split}"])
            records.extend({"split": split, **row} for row in rows)
            by_group = {row["group"]: row for row in rows}
            cross = by_group["cross_20"]["MAE"]
            rest = by_group["not_cross_20"]["MAE"]
            comparison[split] = {"cross_20_minus_all_other_MAE": cross - rest,
                                  "cross_20_over_all_other_MAE": cross / rest if rest else None}
    output = pd.DataFrame(records)
    output.to_csv(csv_path, index=False, float_format="%.12g")
    result = {"generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "code_sha256": R.sha256(Path(__file__)), "input_sha256": input_hashes,
        "model": "M5 official v1 RandomForestRegressor, frozen seed 42",
        "prediction": "reg_val/reg_test: saved forest mean concentration, before level thresholds",
        "feature_set": "v1", "unit": "ug/m3", "target": "CAMS PM2.5 at feature time +3h",
        "timezone": "Asia/Kuala_Lumpur (naive local timestamps)",
        "groups": {"cross_08": "issue hours 05-07; target hours 08-10",
                   "cross_20": "issue hours 17-19; target hours 20-22",
                   "other_hours": "all remaining issue hours; excludes both seam windows",
                   "not_cross_20": "all issue hours except 17-19; includes cross_08 and other_hours"},
        "group_overlap": "First three groups partition each split; not_cross_20 is their comparison aggregate.",
        "rows": records, "comparison": comparison, "new_model_fits": 0,
        "new_model_test_predictions": 0, "csv_sha256": R.sha256(csv_path),
        "limitations": ["Post-hoc descriptive slicing of previously evaluated validation/test predictions.",
            "Raw MAE differences do not establish causality or statistical significance.",
            "No adjustment for concentration, season or serial/spatial dependence; target is a CAMS model product."]}
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    for name, digest in input_hashes.items():
        if R.sha256(ROOT / name) != digest:
            raise AssertionError(f"input changed during analysis: {name}")
    print(output.to_string(index=False))
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
