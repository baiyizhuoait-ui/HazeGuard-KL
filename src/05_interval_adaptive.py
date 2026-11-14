"""按树间分散程度调整区间宽度；参数只在验证集拟合。"""

import math

import numpy as np


def interval_scale(quantiles, floor=1.0):
    q = np.asarray(quantiles, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
        raise ValueError("finite (n,3) tree percentiles required")
    if (q < 0).any() or (np.diff(q, axis=1) < 0).any():
        raise ValueError("nonnegative ordered tree percentiles required")
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("positive finite scale floor required")
    return np.maximum((q[:, 2] - q[:, 0]) / 2, floor)


def fit_calibration(y, quantiles, coverage=0.90, floor=1.0):
    q = np.asarray(quantiles, dtype=float)
    scale = interval_scale(q, floor)
    y = np.asarray(y, dtype=float)
    if y.shape != (len(q),) or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("matching finite nonnegative target vector required")
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1")
    rank = math.ceil((len(y) + 1) * coverage)
    if not 1 <= rank <= len(y):
        raise ValueError("too few calibration samples for a finite interval")
    residual = np.abs(y - q[:, 1]) / scale
    multiplier = float(np.partition(residual, rank - 1)[rank - 1])
    return {"method": "tree-spread-normalized absolute residual",
            "nominal_coverage": coverage, "scale_floor": floor,
            "scale_multiplier": multiplier, "n_calibration": len(y),
            "order_statistic_rank": rank, "unit": "ug/m3",
            "formula": "scale=max((q95-q05)/2,scale_floor); width=scale_multiplier*scale; "
                       "interval=[max(0,q50-width),q50+width]"}


def apply_calibration(quantiles, calibration):
    """新输入只需要 5/50/95 树百分位和冻结参数，不读取目标值。"""
    q = np.asarray(quantiles, dtype=float)
    scale = interval_scale(q, calibration["scale_floor"])
    multiplier = float(calibration["scale_multiplier"])
    if not np.isfinite(multiplier) or multiplier < 0:
        raise ValueError("finite nonnegative scale multiplier required")
    width = multiplier * scale
    return np.column_stack((np.maximum(0, q[:, 1] - width), q[:, 1], q[:, 1] + width))
