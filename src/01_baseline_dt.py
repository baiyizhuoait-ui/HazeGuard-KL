#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01b 基线模型：Decision Tree（组长负责）

用途
----
给出「地板性能」：只用当前时刻的基础观测，不做任何滞后、滚动、空间传导特征。
后续 M2–M6 的所有增益都必须相对这条地板线解释，否则无法证明复杂度是必要的。

训练域
------
只在**受体节点**（KL_City / Port_Klang / Sepang 三个真实 CAMS 格点）上训练与评估。
源区节点不参与本基线——它们是 探索负责人 的空间特征来源，不是预测目标。

产出
----
models/baseline_dt.joblib
models/metrics_stage1.json
figures/fig_baseline_dt_confusion.png
figures/fig_baseline_dt_importance.png

用法
----
python src/01_baseline_dt.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import List

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import utils_metrics as UM  # noqa: E402

PANEL = os.path.join(C.DATA_DIR, "haze_panel_clean.parquet")

# 基础特征：当前时刻的污染物 + 气象 + 去基线残差 + 时间位置
BASE_FEATURES = [
    "pm2_5", "pm10", "no2", "so2", "o3", "co",
    "temperature_2m", "relative_humidity_2m", "wind_speed_ms",
    "wind_direction_10m", "precipitation",
    "pm2_5_anomaly", "temp_anomaly",
    "hour", "month", "dayofweek",
]


def load_panel() -> pd.DataFrame:
    df = pd.read_parquet(PANEL)
    df["split"] = df["split"].astype(str)
    return df


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    取受体节点、构造特征矩阵。
    风向是循环量：359° 与 1° 只差 2°，直接当数值会让树在 0/360 处劈出假边界，
    因此额外派生 sin/cos 两列（仍属于「基础气象特征」，不算滞后/传导特征）。
    """
    d = df[df["role"] == "receptor"].copy()
    rad = np.deg2rad(d["wind_direction_10m"])
    d["wd_sin"] = np.sin(rad)
    d["wd_cos"] = np.cos(rad)

    feats = BASE_FEATURES + ["wd_sin", "wd_cos"]
    d = d.dropna(subset=feats + ["target_level_3h"])
    X = d[feats].astype(float)
    y = d["target_level_3h"].astype(int)
    return X, y, d


def main() -> None:
    t_start = time.time()
    os.makedirs(C.FIGURE_DIR, exist_ok=True)

    print("[1/5] 读取面板")
    df = load_panel()
    X, y, d = build_xy(df)
    split = d["split"].to_numpy()
    print(f"      受体样本 {len(d):,}  train/val/test = "
          f"{(split == 'train').sum():,}/{(split == 'val').sum():,}/{(split == 'test').sum():,}")

    X_tr, y_tr = X[split == "train"], y[split == "train"]
    X_va, y_va = X[split == "val"], y[split == "val"]
    X_te, y_te = X[split == "test"], y[split == "test"]

    # 训练集按时间排序，TimeSeriesSplit 才有意义
    order = np.argsort(d.loc[split == "train", "timestamp"].to_numpy())
    X_tr, y_tr = X_tr.iloc[order], y_tr.iloc[order]

    print("[2/5] TimeSeriesSplit(5) 网格搜索 max_depth × min_samples_leaf × criterion")
    grid = {
        "max_depth": [3, 5, 7, 9, 12, None],
        "min_samples_leaf": [5, 20, 50, 100],
        "criterion": ["gini", "entropy"],
    }
    base = DecisionTreeClassifier(random_state=C.RANDOM_STATE)
    gs = GridSearchCV(
        base, grid,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X_tr, y_tr)
    best = gs.best_estimator_
    print(f"      best={gs.best_params_}  cv_macro_f1={gs.best_score_:.4f}")

    print("[3/5] val 段过拟合检查")
    m_tr = UM.evaluate_predictions(y_tr, best.predict(X_tr))
    m_va = UM.evaluate_predictions(y_va, best.predict(X_va))
    gap = m_tr["Macro_F1"] - m_va["Macro_F1"]
    print(f"      train Macro-F1={m_tr['Macro_F1']:.4f}  val={m_va['Macro_F1']:.4f}  gap={gap:.4f}")

    print("[4/5] test 段评估（全程只评估一次）")
    y_te_pred = best.predict(X_te)
    m_te = UM.evaluate_predictions(y_te, y_te_pred)
    UM.print_metrics(m_te, "Baseline DecisionTree · test")

    per_node = {}
    for sid in C.RECEPTORS:
        mask = (d.loc[split == "test", "station_id"] == sid).to_numpy()
        if mask.sum() == 0:
            continue
        per_node[sid] = UM.evaluate_predictions(y_te[mask], y_te_pred[mask])
    print("      分节点 Macro-F1：" +
          "  ".join(f"{k}={v['Macro_F1']:.3f}" for k, v in per_node.items()))

    print("[5/5] 保存模型与指标")
    model_path = os.path.join(C.MODEL_DIR, "baseline_dt.joblib")
    joblib.dump({"model": best, "features": list(X.columns)}, model_path, compress=3)

    payload = {
        "module": "01 数据工程与基线",
        "owner": "1号（组长）",
        "model": "DecisionTreeClassifier (baseline)",
        "stage": "test",
        "metrics": {k: m_te[k] for k in (
            "Macro_F1", "Balanced_Accuracy", "High_Risk_Recall",
            "MAOE", "Severe_Jumping_Error_Rate", "Avg_Asymmetric_Cost",
            "n_samples", "n_high_risk")},
        "metrics_train": m_tr,
        "metrics_val": m_va,
        "per_node_test": per_node,
        "best_params": {k: (None if v is None else v) for k, v in gs.best_params_.items()},
        "cv_macro_f1": round(float(gs.best_score_), 4),
        "overfit_gap_train_minus_val": round(float(gap), 4),
        "features": list(X.columns),
        "n_features": len(X.columns),
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_test": int(len(X_te)),
        "class_distribution_train": {str(k): int(v) for k, v in y_tr.value_counts().sort_index().items()},
        "class_distribution_test": {str(k): int(v) for k, v in y_te.value_counts().sort_index().items()},
        "tree_structure": {
            "n_leaves": int(best.get_n_leaves()),
            "depth": int(best.get_depth()),
        },
        "train_universe": "receptor nodes only (KL_City, Port_Klang, Sepang)",
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    UM.save_metrics(payload, os.path.join(C.MODEL_DIR, "metrics_stage1.json"))

    _plot_confusion(y_te, y_te_pred)
    _plot_importance(best, X.columns)

    print(f"\n✓ {model_path}")
    print(f"✓ {C.MODEL_DIR}/metrics_stage1.json")
    print(f"✓ {C.FIGURE_DIR}/fig_baseline_dt_confusion.png")
    print(f"✓ {C.FIGURE_DIR}/fig_baseline_dt_importance.png")
    print(f"  叶子 {best.get_n_leaves()} 个，深度 {best.get_depth()}，耗时 {time.time() - t_start:.0f}s")


def _plot_confusion(y_true, y_pred) -> None:
    cm = UM.confusion(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=200)
    im = ax.imshow(cm, cmap="Blues")
    names = [f"L{i}" for i in range(4)]
    ax.set_xticks(range(4)); ax.set_xticklabels(names)
    ax.set_yticks(range(4)); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted level"); ax.set_ylabel("True level")
    ax.set_title("Baseline Decision Tree — Test Confusion Matrix", fontsize=11)
    thr = cm.max() / 2.0
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thr else "#222222", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURE_DIR, "fig_baseline_dt_confusion.png"))
    plt.close(fig)


def _plot_importance(model, cols: List[str]) -> None:
    imp = pd.Series(model.feature_importances_, index=list(cols)).sort_values()
    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=200)
    ax.barh(imp.index, imp.values, color="#2E5496")
    ax.set_xlabel("Gini importance")
    ax.set_title("Baseline Decision Tree — Feature Importance", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURE_DIR, "fig_baseline_dt_importance.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
