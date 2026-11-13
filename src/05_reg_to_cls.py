"""读取第 3 位同学的数据，完成回归分级和时间划分。"""

import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import config as C
from utils_metrics import evaluate_predictions

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
THRESHOLDS = np.array([15., 35., 55.])
# 第一组沿用 4 号的结构，后两组分别调整深度、叶节点大小。
CANDIDATES = [
    dict(n_estimators=100, max_depth=14, min_samples_leaf=20, max_features=0.5),
    dict(n_estimators=100, max_depth=10, min_samples_leaf=20, max_features=0.5),
    dict(n_estimators=100, max_depth=14, min_samples_leaf=10, max_features=0.5),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def levels(values, thresholds=THRESHOLDS):
    t = np.asarray(thresholds, dtype=float)
    if t.shape != (3,) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("three finite, increasing cut points are required")
    return np.digitize(np.asarray(values), t).astype(np.int8)


def load_data(root=ROOT):
    selection_path = root / "models/selected_features.json"
    data_path = root / "data/features_final.parquet"
    names = json.loads(selection_path.read_text(encoding="utf-8"))["selected_features"]
    if len(names) != 25 or len(set(names)) != 25:
        raise ValueError("expected exactly 25 distinct selected features")
    if set(names) & {"timestamp", "station_id", "split", "role",
                     "target_pm25_3h", "target_level_3h"}:
        raise ValueError("identifier or target in selected features")
    df = pd.read_parquet(data_path)
    required = set(names) | {"station_id", "timestamp", "role", "split",
                             "target_pm25_3h", "target_level_3h"}
    if not required.issubset(df.columns):
        raise ValueError(f"missing columns: {sorted(required - set(df.columns))}")
    df = df.loc[df.role.eq("receptor")].copy()
    if df["split"].isna().any() or set(df["split"]) != {"train", "val", "test"}:
        raise ValueError("expected train/val/test split labels")
    if set(df.station_id) != set(C.RECEPTORS):
        raise ValueError("unexpected receptor nodes")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    if df.timestamp.isna().any() or df.duplicated(["station_id", "timestamp"]).any():
        raise ValueError("missing timestamp or duplicate row key")
    if df.timestamp.dt.tz is not None:
        raise ValueError("contract timestamps must be naive KL local time")
    numeric = df[names + ["target_pm25_3h", "target_level_3h"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("non-finite feature or target")
    if (df.target_pm25_3h < 0).any() or not np.array_equal(
        levels(df.target_pm25_3h), df.target_level_3h.to_numpy()
    ):
        raise ValueError("concentration/level targets are inconsistent")
    # pandas 3 下直接替换整列，避免原来的 int8 类型限制赋值。
    df["target_level_3h"] = df.target_level_3h.astype(np.int8)
    df = df.sort_values(["timestamp", "station_id"], kind="stable").reset_index(drop=True)
    parts = {s: df.loc[df.split.eq(s)].reset_index(drop=True)
             for s in ("train", "val", "test")}
    for s, p in parts.items():
        expected = C.PANEL_CONTRACT[f"receptor_{s}_eligible_rows"]
        if len(p) != expected:
            raise ValueError(f"{s}: {len(p)} rows; contract requires {expected}")
    for a, b in (("train", "val"), ("val", "test")):
        if parts[a].timestamp.max() + pd.Timedelta(hours=3) >= parts[b].timestamp.min():
            raise ValueError("target horizon crosses split boundary")
    return parts, names, {"features_final": sha256(data_path),
                          "selected_features": sha256(selection_path)}


def chronological_folds(timestamps, n_splits=5):
    """同一时刻的节点一起划分；训练标签的 t+3h 不能进入验证段。"""
    ts = pd.to_datetime(timestamps).to_numpy(dtype="datetime64[us]")
    if np.any(ts[1:] < ts[:-1]):
        raise ValueError("sort by timestamp before TimeSeriesSplit")
    unique = np.unique(ts)
    folds = []
    for fit_idx, hold_idx in TimeSeriesSplit(n_splits=n_splits).split(unique):
        start = unique[hold_idx[0]]
        fit_times = unique[fit_idx]
        fit_times = fit_times[fit_times + np.timedelta64(3, "h") < start]
        tr = np.flatnonzero(np.isin(ts, fit_times))
        va = np.flatnonzero(np.isin(ts, unique[hold_idx]))
        if not len(tr) or not len(va):
            raise ValueError("empty chronological fold after purging")
        if ts[tr].max() + np.timedelta64(3, "h") >= ts[va].min():
            raise AssertionError("forecast target leaks across CV boundary")
        folds.append((tr, va))
    return folds


def tree_predictions(model, X):
    X = np.asarray(X, dtype=np.float32)
    return np.stack([tree.predict(X) for tree in model.estimators_], axis=1)


def interval_metrics(y, quantiles):
    y = np.asarray(y)
    q = np.asarray(quantiles)
    if q.shape != (len(y), 3) or not np.isfinite(q).all() or np.any(np.diff(q, axis=1) < 0):
        raise ValueError("ordered finite (n,3) quantiles required")
    return {"PICP": float(np.mean((y >= q[:, 0]) & (y <= q[:, 2]))),
            "MPIW": float(np.mean(q[:, 2] - q[:, 0])),
            "n_samples": len(y), "nominal_tree_percentiles": [0.05, 0.5, 0.95],
            "unit": "ug/m3", "calibrated_prediction_interval": False}


def search_cut_points(pred, y):
    """只在验证集选切点：先看 F1，再看等级误差和切点偏移。"""
    trace = []
    for offsets in product((-4, -2, 0, 2, 4), repeat=3):
        cuts = THRESHOLDS + offsets
        score = evaluate_predictions(y, levels(pred, cuts))
        trace.append({"thresholds": cuts.tolist(), "metrics": score,
                      "distance_from_fixed": sum(abs(v) for v in offsets)})
    best = max(trace, key=lambda r: (r["metrics"]["Macro_F1"], -r["metrics"]["MAOE"],
                                     -r["distance_from_fixed"]))
    return best["thresholds"], trace


if __name__ == "__main__":
    # 两个入口使用同一训练流程，不需要各跑一次。
    import importlib
    importlib.import_module("05_ordinal_rf").main()
