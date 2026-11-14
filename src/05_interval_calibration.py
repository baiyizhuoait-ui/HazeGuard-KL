"""用验证集绝对残差校准浓度区间；复用保存预测，不重训模型。

运行：python src/05_interval_calibration.py
"""

import importlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

R = importlib.import_module("05_reg_to_cls")
ROOT = Path(__file__).resolve().parents[1]


def fit_calibration(y, median, coverage=0.90):
    y, median = np.asarray(y, dtype=float), np.asarray(median, dtype=float)
    if y.ndim != 1 or y.shape != median.shape or not len(y):
        raise ValueError("matching nonempty one-dimensional targets/medians required")
    if not np.isfinite(y).all() or not np.isfinite(median).all():
        raise ValueError("non-finite calibration values")
    if (y < 0).any() or (median < 0).any() or not 0 < coverage < 1:
        raise ValueError("nonnegative PM2.5 and coverage between 0 and 1 required")
    # 使用有限样本次序统计量，不对分位点插值。
    rank = math.ceil((len(y) + 1) * coverage)
    if rank > len(y):
        raise ValueError("too few calibration samples for a finite interval")
    residual = np.abs(y - median)
    width = float(np.partition(residual, rank - 1)[rank - 1])
    return {"nominal_coverage": coverage, "half_width": width,
            "n_calibration": len(y), "order_statistic_rank": rank,
            "center": "median of M5 individual tree predictions",
            "lower_bound": "max(0, median - half_width)",
            "upper_bound": "median + half_width", "unit": "ug/m3"}


def apply_calibration(median, calibration):
    """给 7 号使用：只需要预测中位数和冻结的宽度，不读取真实值。"""
    median = np.asarray(median, dtype=float)
    width = float(calibration["half_width"])
    if median.ndim != 1 or not np.isfinite(median).all() or (median < 0).any():
        raise ValueError("finite nonnegative one-dimensional medians required")
    if not np.isfinite(width) or width < 0:
        raise ValueError("finite nonnegative half_width required")
    return np.column_stack((np.maximum(0, median - width), median, median + width))


def interval_scores(y, band):
    scores = R.interval_metrics(y, band)
    # 原函数的树百分位元数据不适用于校准区间，只取同口径的三个统计量。
    return {key: scores[key] for key in ("PICP", "MPIW", "n_samples")}


def main():
    md = ROOT / "models"
    outputs = [md / name for name in ("interval_calibration.json",
               "metrics_stage5_interval_calibrated.json", "stage5_intervals_calibrated.npz")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("calibration outputs already exist; review saved results instead")

    parts, _, hashes = R.load_data()
    original = json.loads((md / "metrics_stage5.json").read_text(encoding="utf-8"))
    if hashes != original["input_sha256"]:
        raise ValueError("upstream data differ from the original stage-05 experiment")
    for name in ("stage5_predictions.npz", "ordinal_probs_val_test.npz"):
        if R.sha256(md / name) != original["artifacts"][name]:
            raise ValueError(f"saved predictions changed: {name}")

    with np.load(md / "stage5_predictions.npz", allow_pickle=False) as saved, \
            np.load(md / "ordinal_probs_val_test.npz", allow_pickle=False) as rows:
        val = parts["val"]
        if not np.array_equal(rows["ts_val"], val.timestamp.to_numpy()) or not np.array_equal(
            rows["station_id_val"], val.station_id.to_numpy(dtype=str)
        ):
            raise ValueError("validation row identity mismatch")
        q_val = saved["quantiles_val"]
        # 仅此处计算宽度。冻结后才对缓存测试预测评分，不按测试覆盖率调整。
        calibration = fit_calibration(val.target_pm25_3h.to_numpy(), q_val[:, 1])
        calibration.update({"calibration_split": "val", "pooling": "all three receptor nodes",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_model": "M5 RandomForestRegressor", "input_sha256": hashes,
            "source_code_sha256": R.sha256(Path(__file__)),
            "saved_predictions_sha256": R.sha256(md / "stage5_predictions.npz"),
            "limitations": ["Post-selection calibration: validation was also used for model selection.",
                "Temporally dependent data: exchangeability and 90% future coverage are not guaranteed.",
                "Original test coverage was already known; this is a post-hoc extension, not a new blind test."]})
        with outputs[0].open("x", encoding="utf-8") as stream:
            json.dump(calibration, stream, ensure_ascii=False, indent=2, allow_nan=False)

        result = {"method": "validation absolute residual about frozen M5 tree median",
                  "nominal_coverage": 0.90, "half_width": calibration["half_width"],
                  "calibration_sha256": R.sha256(outputs[0]), "unit": "ug/m3",
                  "new_model_fits": 0, "new_model_test_predictions": 0,
                  "limitations": calibration["limitations"]}
        arrays = {}
        for split in ("val", "test"):
            frame = parts[split]
            if not np.array_equal(rows[f"ts_{split}"], frame.timestamp.to_numpy()) or not np.array_equal(
                rows[f"station_id_{split}"], frame.station_id.to_numpy(dtype=str)
            ):
                raise ValueError(f"{split} row identity mismatch")
            raw = saved[f"quantiles_{split}"]
            band = apply_calibration(raw[:, 1], calibration)
            y = frame.target_pm25_3h.to_numpy()
            result[split] = {"raw_tree_spread": interval_scores(y, raw),
                             "calibrated": interval_scores(y, band), "by_node": {}}
            for node in sorted(frame.station_id.unique()):
                mask = frame.station_id.eq(node).to_numpy()
                result[split]["by_node"][node] = {
                    "raw_tree_spread": interval_scores(y[mask], raw[mask]),
                    "calibrated": interval_scores(y[mask], band[mask])}
            arrays[f"interval_{split}"] = band
            arrays[f"ts_{split}"] = rows[f"ts_{split}"]
            arrays[f"station_id_{split}"] = rows[f"station_id_{split}"]
        np.savez_compressed(outputs[2], **arrays)
        result["intervals_sha256"] = R.sha256(outputs[2])
        with outputs[1].open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
