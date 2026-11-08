#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全组统一评估接口。

全组必须调用本模块产出指标，禁止各自实现——否则口径不同，横向对比失效。

六项指标（顺序即报告中的推荐排序依据）
--------------------------------------
1. Macro_F1                 四等级 F1 的算术平均
2. Balanced_Accuracy        各类召回率的平均
3. High_Risk_Recall         真实 Level 2/3 中，预测落在 {2,3} 的比例（核心防漏报）
                            口径是「高危有没有被预警」，不是「等级有没有报对」。
                            真实 L2 报成 L3 属于过度预警，不算漏报。
4. MAOE                     平均绝对序差 (1/N)Σ|y-ŷ|
5. Severe_Jumping_Error_Rate 跨两级误判比例 (1/N)Σ1(|y-ŷ|≥2)
6. Avg_Asymmetric_Cost      平均非对称代价 (1/N)ΣC[y][ŷ]
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score)

# 真实为 i、预测为 j 的代价。左下三角（漏报高危）= 普通误报的 8–15 倍
DEFAULT_COST_MATRIX = [
    [0, 1, 3, 5],
    [2, 0, 1, 3],
    [8, 4, 0, 1],
    [15, 10, 5, 0],
]

LABELS = [0, 1, 2, 3]


def _as_int_array(x: Iterable) -> np.ndarray:
    a = np.asarray(list(x) if not isinstance(x, np.ndarray) else x)
    return a.astype(int)


def evaluate_predictions(
    y_true: Sequence,
    y_pred: Sequence,
    y_prob: Optional[np.ndarray] = None,
    cost_matrix: Optional[Sequence[Sequence[float]]] = None,
    high_risk_levels: Sequence[int] = (2, 3),
) -> Dict[str, float]:
    """
    统一评估。

    Parameters
    ----------
    y_true, y_pred : array-like of int
        真实与预测的有序等级（0–3）。
    y_prob : (N, 4) array, optional
        概率矩阵。当前六项指标不需要它，保留以便下游扩展
        （如 Log-Loss、期望代价最小化），传入时会被校验形状。
    cost_matrix : 4×4, optional
        非对称代价矩阵，默认 DEFAULT_COST_MATRIX。
    high_risk_levels : tuple
        哪些等级算「高危」，默认 (2, 3)。

    Returns
    -------
    dict  恰好包含六项指标 + 两个辅助计数（n_samples / n_high_risk）
    """
    yt = _as_int_array(y_true)
    yp = _as_int_array(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"长度不一致：y_true={len(yt)} y_pred={len(yp)}")
    if len(yt) == 0:
        raise ValueError("空输入无法评估")
    if y_prob is not None:
        p = np.asarray(y_prob)
        if p.ndim != 2 or p.shape[0] != len(yt):
            raise ValueError(f"y_prob 形状应为 (N,4)，实际 {p.shape}")

    cm = np.asarray(cost_matrix if cost_matrix is not None else DEFAULT_COST_MATRIX, dtype=float)
    if cm.shape != (4, 4):
        raise ValueError(f"cost_matrix 应为 4×4，实际 {cm.shape}")

    # 1) Macro-F1：显式给出 labels，避免某类缺席时静默改变平均基数
    macro_f1 = float(f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0))

    # 2) Balanced Accuracy
    bal_acc = float(balanced_accuracy_score(yt, yp))

    # 3) High-Risk Recall —— 口径是"高危有没有被预警"，不是"等级有没有报对"
    #    真实属于高危（默认 L2/L3）的样本中，预测落在高危集合内的比例。
    #    注意：把真实 L2 报成 L3 属于"过度预警"，不是漏报，两种口径差别很大
    #    （基线 DT 上 union=0.7693、exact=0.5707），报告里必须写清用的是哪一种。
    hr_mask = np.isin(yt, high_risk_levels)
    n_hr = int(hr_mask.sum())
    hr_recall = float(np.isin(yp[hr_mask], high_risk_levels).mean()) if n_hr > 0 else float("nan")

    # 辅助诊断（不进六项主指标）：严格逐类召回，用于混淆矩阵讨论
    hr_exact = float((yp[hr_mask] == yt[hr_mask]).mean()) if n_hr > 0 else float("nan")
    l3_mask = yt == max(high_risk_levels)
    l3_recall = float((yp[l3_mask] == yt[l3_mask]).mean()) if l3_mask.sum() > 0 else float("nan")

    # 4) MAOE
    abs_err = np.abs(yt - yp)
    maoe = float(abs_err.mean())

    # 5) 跨两级误判率
    severe = float((abs_err >= 2).mean())

    # 6) 平均非对称代价
    avg_cost = float(cm[yt, yp].mean())

    return {
        "Macro_F1": round(macro_f1, 4),
        "Balanced_Accuracy": round(bal_acc, 4),
        "High_Risk_Recall": round(hr_recall, 4) if n_hr > 0 else None,
        "MAOE": round(maoe, 4),
        "Severe_Jumping_Error_Rate": round(severe, 4),
        "Avg_Asymmetric_Cost": round(avg_cost, 4),
        "n_samples": int(len(yt)),
        "n_high_risk": n_hr,
        # 诊断项：不参与六项主指标的横向排序，只在分析里用
        "diag_High_Risk_Exact_Recall": round(hr_exact, 4) if n_hr > 0 else None,
        "diag_Level3_Recall": round(l3_recall, 4) if l3_mask.sum() > 0 else None,
    }


def confusion(y_true: Sequence, y_pred: Sequence) -> np.ndarray:
    """4×4 混淆矩阵，行=真实，列=预测。"""
    return confusion_matrix(_as_int_array(y_true), _as_int_array(y_pred), labels=LABELS)


def print_metrics(metrics: Dict[str, float], title: str = "") -> None:
    if title:
        print(f"\n{title}")
        print("-" * 46)
    order = ["Macro_F1", "Balanced_Accuracy", "High_Risk_Recall",
             "MAOE", "Severe_Jumping_Error_Rate", "Avg_Asymmetric_Cost"]
    for k in order:
        v = metrics.get(k)
        print(f"  {k:26s} {v if v is not None else 'n/a'}")
    for k in ("diag_High_Risk_Exact_Recall", "diag_Level3_Recall"):
        if k in metrics:
            print(f"  {k:26s} {metrics[k]}   (诊断项)")
    print(f"  {'n_samples':26s} {metrics.get('n_samples')}"
          f"   (高危 {metrics.get('n_high_risk')})")


def save_metrics(metrics: Dict, path: str, extra: Optional[Dict] = None) -> None:
    payload = dict(metrics)
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def expected_cost(y_prob: np.ndarray, cost_matrix: Optional[Sequence[Sequence[float]]] = None
                  ) -> np.ndarray:
    """
    最小期望风险决策：ŷ = argmin_j Σ_i P(Y=i|x)·C[i][j]
    供 优化与解释负责人 与代价敏感模型复用。
    """
    cm = np.asarray(cost_matrix if cost_matrix is not None else DEFAULT_COST_MATRIX, dtype=float)
    return np.argmin(np.asarray(y_prob) @ cm, axis=1)


if __name__ == "__main__":
    # 自检：构造一个可解释的小例子
    yt = [0, 1, 2, 3, 2, 3, 1, 0]
    yp = [0, 1, 1, 3, 2, 0, 1, 1]
    m = evaluate_predictions(yt, yp)
    print_metrics(m, "自检样例")
    print("\n混淆矩阵：\n", confusion(yt, yp))
