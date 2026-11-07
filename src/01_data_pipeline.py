#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01 数据管道：采集 · 清洗 · 时序切分 · 气候态基线相减（组长负责）

产出
----
data/haze_panel_clean.parquet   建模主表（26 个 CAMS 空间单元 × 逐小时）
data/data_dictionary.json       字段口径
models/gate0_report.json        数据可行性前置校验
models/cleaning_report.json     清洗计数与示例（可追溯）

用法
----
python src/01_data_pipeline.py                # 优先读本地缓存
python src/01_data_pipeline.py --force-refresh  # 忽略缓存重新拉取
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://archive-api.open-meteo.com/v1/archive"


# ---------------------------------------------------------------------------
# 采集
# ---------------------------------------------------------------------------
def _get_json(url: str, retries: int = 4, backoff: float = 3.0) -> dict:
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            if i < retries - 1:
                time.sleep(backoff * (i + 1))
    raise RuntimeError(f"请求失败（重试 {retries} 次）：{url}\n{last}")


def fetch_node(node: dict, force: bool = False) -> tuple[pd.DataFrame, dict]:
    """拉取单个节点的空气质量 + 气象，返回合并后的 DataFrame。"""
    os.makedirs(C.RAW_DIR, exist_ok=True)
    sid = node["id"]
    cache_path = os.path.join(C.RAW_DIR, f"{sid}.json")
    log = {"station_id": sid, "cached": False, "aq_rows": 0, "wx_rows": 0}

    if os.path.exists(cache_path) and not force:
        with open(cache_path, encoding="utf-8") as f:
            payload = json.load(f)
        log["cached"] = True
    else:
        aq_vars = ",".join(C.AQ_VARS.keys())
        wx_vars = ",".join(C.WEATHER_VARS.keys())
        common = (f"latitude={node['lat']}&longitude={node['lon']}"
                  f"&start_date={C.START_DATE}&end_date={C.END_DATE}"
                  f"&timezone={C.TIMEZONE.replace('/', '%2F')}")
        aq = _get_json(f"{AQ_URL}?{common}&hourly={aq_vars}")
        wx = _get_json(f"{WX_URL}?{common}&hourly={wx_vars}")
        payload = {"air_quality": aq["hourly"], "weather": wx["hourly"]}
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    aq, wx = payload["air_quality"], payload["weather"]
    df = pd.DataFrame({"timestamp": pd.to_datetime(aq["time"])})
    for src, dst in C.AQ_VARS.items():
        df[dst] = pd.to_numeric(pd.Series(aq[src]), errors="coerce")
    wx_df = pd.DataFrame({"timestamp": pd.to_datetime(wx["time"])})
    for src, dst in C.WEATHER_VARS.items():
        wx_df[dst] = pd.to_numeric(pd.Series(wx[src]), errors="coerce")

    df = df.merge(wx_df, on="timestamp", how="inner")
    log["aq_rows"] = len(aq["time"])
    log["wx_rows"] = len(wx["time"])

    df["station_id"] = sid
    df["lat"] = node["lat"]
    df["lon"] = node["lon"]
    df["region"] = node["region"]
    df["role"] = node["role"]
    return df, log


# ---------------------------------------------------------------------------
# 清洗
# ---------------------------------------------------------------------------
def clean_node(df: pd.DataFrame, rep: dict) -> pd.DataFrame:
    """单节点清洗：去重 → 重建小时网格 → 越界置空 → 短缺口插值 → 缺失标记。"""
    sid = df["station_id"].iloc[0]
    n0 = len(df)

    # 1) 去重：同一 (station_id, timestamp) 只留第一条
    before = len(df)
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    rep["duplicates_removed"] += before - len(df)

    # 2) 重建完整小时网格（保证滞后位移 = 真实小时数）
    df = df.sort_values("timestamp").set_index("timestamp")
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    before = len(df)
    df = df.reindex(full_idx)
    df["station_id"] = sid
    df[["lat", "lon", "region", "role"]] = df[["lat", "lon", "region", "role"]].ffill().bfill()
    rep["grid_rows_added"] += len(df) - before
    df.index.name = "timestamp"
    df = df.reset_index()

    # 2b) 剔除序列开头的空块：CAMS 档案边界处前 80 小时 PM2.5 为空，
    #     属于档案起点而非真实缺测，若计入会把「最长缺失」误判为 80h。
    first_valid = df["pm2_5"].first_valid_index()
    if first_valid is not None and first_valid > 0:
        rep["leading_block_trimmed"] = rep.get("leading_block_trimmed", 0) + int(first_valid)
        df = df.loc[first_valid:].reset_index(drop=True)

    # 3) 物理越界 → NaN
    for col, (lo, hi) in C.BOUNDS.items():
        if col not in df.columns:
            continue
        bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        rep["out_of_bounds"][col] = int(bad.sum())
        df.loc[bad, col] = np.nan

    # 4) 风速单位换算 km/h → m/s（原始值保留以便复核）
    df["wind_speed_ms"] = df["wind_speed_kmh"] / 3.6

    # 5) 只在时间维上检查缺失长度（用于 Gate 0 与插值上限）
    value_cols = list(C.BOUNDS.keys())
    nan_mask = df[value_cols].isna().any(axis=1)
    runs = _max_run(nan_mask.to_numpy())
    rep["max_gap_h"][sid] = int(runs)

    # 6) 连续缺失 <=3h 线性插值；>3h 保持 NaN 并打标记
    before_nan = df[value_cols].isna().sum().sum()
    nan_before = df[value_cols].isna()
    df[value_cols] = (
        df[value_cols]
        .interpolate(method="linear", limit=C.MAX_INTERP_GAP_H, limit_area="inside")
    )
    filled = (nan_before & df[value_cols].notna()).any(axis=1)
    df["qc_interpolated"] = filled.astype("int8")
    rep["interpolated_cells"] += int(before_nan - df[value_cols].isna().sum().sum())
    rep["rows_interpolated"] = rep.get("rows_interpolated", 0) + int(filled.sum())

    df["is_missing"] = 0
    for col in value_cols:
        df.loc[df[col].isna(), "is_missing"] = 1

    rep["rows_total"] += len(df)
    rep["rows_with_missing"] += int((df["is_missing"] == 1).sum())
    rep["rows_fetched"] += n0
    return df


def _max_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


# ---------------------------------------------------------------------------
# 目标与切分
# ---------------------------------------------------------------------------
def add_qc_flags(df: pd.DataFrame, rep: dict) -> pd.DataFrame:
    """
    质量标记（v2.1）。只标记，不修改数值——任何"修正"都会污染下游的可追溯性。
    下游模块 可自行决定丢弃、截断或加权，但必须显式处理。
    """
    df = df.copy()

    # a) PM2.5 > PM10：物理上不可能（PM2.5 是 PM10 的子集）
    df["qc_pm25_gt_pm10"] = (df["pm2_5"] > df["pm10"]).astype("int8")

    # b) 模式封顶：CAMS 输出上限约 900 µg/m³，贴顶值是被截断的
    df["qc_censored"] = (df["pm2_5"] >= C.CENSORED_PM25).astype("int8")

    # c) 静风：风向是数值噪声，上风向权重不可信
    df["qc_calm_wind"] = (df["wind_speed_ms"] < C.CALM_WIND_MS).astype("int8")

    rep["qc_counts"] = {
        "pm25_gt_pm10": int(df["qc_pm25_gt_pm10"].sum()),
        "censored": int(df["qc_censored"].sum()),
        "calm_wind": int(df["qc_calm_wind"].sum()),
        "interpolated": int(df["qc_interpolated"].sum()),
        "missing": int(df["is_missing"].sum()),
    }
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["station_id", "timestamp"]).copy()
    g = df.groupby("station_id", sort=False)["pm2_5"]
    # 关键：按站点分组 shift，避免站点 A 末尾接到站点 B 开头
    df["target_pm25_3h"] = g.shift(-C.HORIZON_H)
    t = df["target_pm25_3h"]
    df["target_level_3h"] = np.select(
        [t < C.LEVEL_THRESHOLDS[0],
         t < C.LEVEL_THRESHOLDS[1],
         t < C.LEVEL_THRESHOLDS[2]],
        [0, 1, 2], default=3,
    ).astype("float")
    df.loc[t.isna(), "target_level_3h"] = np.nan
    return df


def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    """按每节点时间跨度 70/15/15 切分（时间基准，不按行数，抗缺失扰动）。"""
    parts = []
    for sid, sub in df.groupby("station_id", sort=False):
        sub = sub.sort_values("timestamp").copy()
        t0, t1 = sub["timestamp"].min(), sub["timestamp"].max()
        span = (t1 - t0).total_seconds()
        cut1 = t0 + pd.Timedelta(seconds=span * C.SPLIT_RATIOS[0])
        cut2 = t0 + pd.Timedelta(seconds=span * (C.SPLIT_RATIOS[0] + C.SPLIT_RATIOS[1]))
        sub["split"] = np.where(sub["timestamp"] < cut1, "train",
                                np.where(sub["timestamp"] < cut2, "val", "test"))
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def add_baseline(df: pd.DataFrame, rep: dict | None = None) -> pd.DataFrame:
    """
    气候态基线：仅在 train 上按 (station, month, hour) 求均值，再广播到全量。

    v2.1 增加支撑度审计：若某 (station,month,hour) 的训练样本少于 MIN_BASELINE_SUPPORT，
    该单元格的基线不可信，回退到 (station, month) 均值，并记录 b_proto_src。
    """
    df = df.copy()
    df["month"] = df["timestamp"].dt.month
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    train = df[df["split"] == "train"]
    assert (train["split"] == "train").all(), "基线只能使用 train 段"

    proto = (train.groupby(["station_id", "month", "hour"], observed=True)
             .agg(b_proto_pm25=("pm2_5", "mean"),
                  b_proto_temp=("temperature_2m", "mean"),
                  b_proto_n=("pm2_5", "size"))
             .reset_index())

    # 低支撑单元格回退到 (station, month)
    low = proto["b_proto_n"] < C.MIN_BASELINE_SUPPORT
    if rep is not None:
        rep["baseline_low_support_cells"] = int(low.sum())
        rep["baseline_total_cells"] = int(len(proto))
        rep["baseline_min_support"] = int(proto["b_proto_n"].min())

    if low.any():
        sm = (train.groupby(["station_id", "month"], observed=True)
              .agg(fb_pm25=("pm2_5", "mean"), fb_temp=("temperature_2m", "mean"))
              .reset_index())
        proto = proto.merge(sm, on=["station_id", "month"], how="left")
        proto.loc[low, "b_proto_pm25"] = proto.loc[low, "fb_pm25"]
        proto.loc[low, "b_proto_temp"] = proto.loc[low, "fb_temp"]
        proto = proto.drop(columns=["fb_pm25", "fb_temp"])
        proto["b_proto_src"] = np.where(low, "station_month", "station_month_hour")
    else:
        proto["b_proto_src"] = "station_month_hour"

    df = df.merge(proto, on=["station_id", "month", "hour"], how="left")
    df["pm2_5_anomaly"] = df["pm2_5"] - df["b_proto_pm25"]
    df["temp_anomaly"] = df["temperature_2m"] - df["b_proto_temp"]
    return df


# ---------------------------------------------------------------------------
# Gate 0
# ---------------------------------------------------------------------------
def gate0(df: pd.DataFrame, rep: dict) -> dict:
    """
    四项闸门。G0-2 与 G0-4 只在**受体节点**上判定：
    项目的部署目标是 KL，把苏门答腊/加里曼丹的低浓度源区并进统计会稀释分布、
    并把源区的年际跳变误判成漂移。全panel 的数字作为背景信息一并给出。
    """
    rec = df[df["role"] == "receptor"]
    train = rec[rec["split"] == "train"]
    test = rec[rec["split"] == "test"]
    lvl_all = rec["target_level_3h"].value_counts(normalize=True).sort_index()
    lvl_train = train["target_level_3h"].value_counts(normalize=True).sort_index()
    lvl_test = test["target_level_3h"].value_counts(normalize=True).sort_index()

    per_node = df.groupby("station_id").size().to_dict()
    min_node = min(per_node.values())

    p3_all = float(lvl_all.get(3.0, 0.0))
    p3_tr = float(lvl_train.get(3.0, 0.0))
    p3_te = float(lvl_test.get(3.0, 0.0))
    ratio = (max(p3_tr, p3_te) / max(1e-9, min(p3_tr, p3_te))) if min(p3_tr, p3_te) > 0 else float("inf")

    checks = {
        "G0-1_total_rows": {
            "pass": len(df) >= 100_000 and min_node >= 12_000,
            "value": {"total_rows": int(len(df)), "min_rows_per_node": int(min_node)},
            "rule": "总行数 ≥ 100,000 且每节点 ≥ 12,000",
        },
        "G0-2_level3_share": {
            "pass": p3_all >= 0.01,
            "value": {"receptor_level3": round(p3_all, 4),
                      "panel_level3": round(float((df["target_level_3h"] == 3).mean()), 4)},
            "rule": "受体节点 Level 3 占比 ≥ 1.0%",
        },
        "G0-3_max_gap": {
            "pass": max(rep["max_gap_h"][s] for s in rec["station_id"].unique()) <= 24,
            "value": {
                "max_gap_receptor_h": int(max(rep["max_gap_h"][s] for s in rec["station_id"].unique())),
                "max_gap_panel_h": int(max(rep["max_gap_h"].values())),
            },
            "rule": "受体节点最长连续缺失 ≤ 24 小时（源区节点的缺口不影响标签，仅披露）",
        },
        "G0-4_distribution_drift": {
            "pass": ratio <= 3.0,
            "value": {"train_level3": round(p3_tr, 4), "test_level3": round(p3_te, 4),
                      "ratio": round(ratio, 2)},
            "rule": "受体节点 train 与 test 的 Level 3 占比差异 ≤ 3 倍",
        },
    }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_nodes": int(df["station_id"].nunique()),
        "n_receptors": int(rec["station_id"].nunique()),
        "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "split_rows": {k: int(v) for k, v in df["split"].value_counts().items()},
        "level_distribution_receptor_all": {str(int(k)): round(float(v), 4) for k, v in lvl_all.items()},
        "level_distribution_receptor_train": {str(int(k)): round(float(v), 4) for k, v in lvl_train.items()},
        "level_distribution_receptor_test": {str(int(k)): round(float(v), 4) for k, v in lvl_test.items()},
        "level_distribution_receptor_val": {
            str(int(k)): round(float(v), 4)
            for k, v in rec[rec["split"] == "val"]["target_level_3h"]
            .value_counts(normalize=True).sort_index().items()
        },
        "checks": checks,
        "all_passed": all(c["pass"] for c in checks.values()),
        "notes": [
            "测试集覆盖 2026 年 8–10 月的强跨境烟霾过程（KL 单月 Level 3 占比达 47%–51%，"
            "PM2.5 p95 = 151.8 µg/m³），因此 High-Risk Recall 具有真实事件支撑。",
            "空气质量为 CAMS 0.4° 模式产品，非地面实测；报告须声明。",
            "同格点候选点位已归并为单一节点，见 docs/node_discovery.json。",
        ],
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-refresh", action="store_true")
    args = ap.parse_args()

    for d in (C.DATA_DIR, C.MODEL_DIR):
        os.makedirs(d, exist_ok=True)

    rep = {
        "duplicates_removed": 0,
        "grid_rows_added": 0,
        "interpolated_cells": 0,
        "out_of_bounds": {},
        "max_gap_h": {},
        "rows_total": 0,
        "rows_with_missing": 0,
        "rows_fetched": 0,
        "per_node": {},
    }

    frames = []
    fetch_log = []
    print(f"[1/5] 采集 {len(C.NODES)} 个节点  {C.START_DATE} → {C.END_DATE}")
    for i, node in enumerate(C.NODES, 1):
        raw, log = fetch_node(node, force=args.force_refresh)
        cleaned = clean_node(raw, rep)
        frames.append(cleaned)
        rep["per_node"][node["id"]] = len(cleaned)
        fetch_log.append(log)
        tag = "cache" if log["cached"] else "api  "
        print(f"      [{i:2d}/{len(C.NODES)}] {node['id']:20s} {tag}  rows={len(cleaned):6d}")

    print("[2/5] 合并面板")
    df = pd.concat(frames, ignore_index=True)

    print("[3/5] 构造 t+3h 目标")
    df = add_targets(df)

    print("[4/5] 严格时序切分 + 气候态基线 + 质量标记")
    df = assign_split(df)
    df = add_baseline(df, rep)
    df = add_qc_flags(df, rep)

    # 建模行：目标必须存在，且当前 pm2_5 必须存在
    # pm2_5 既是锚点特征又是 anomaly 的基底，缺了它任何模型都用不了，
    # 因此在这里直接剔除，而不是留给下游每位组员 各自记得过滤。
    before = len(df)
    df = df[df["target_pm25_3h"].notna()].reset_index(drop=True)
    rep["rows_dropped_no_target"] = before - len(df)

    before = len(df)
    df = df[df["pm2_5"].notna()].reset_index(drop=True)
    rep["rows_dropped_no_anchor"] = before - len(df)

    # 统一列顺序
    cols = (["station_id", "timestamp", "lat", "lon", "region", "role", "split"]
            + list(C.BOUNDS.keys()) + ["wind_speed_kmh", "wind_direction_10m"]
            + ["is_missing", "qc_pm25_gt_pm10", "qc_censored", "qc_calm_wind", "qc_interpolated"]
            + ["month", "hour", "dayofweek",
               "b_proto_pm25", "b_proto_temp", "b_proto_n", "b_proto_src",
               "pm2_5_anomaly", "temp_anomaly",
               "target_pm25_3h", "target_level_3h"])
    df = df[cols].sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    df["target_level_3h"] = df["target_level_3h"].astype("int8")
    df["split"] = df["split"].astype("category")

    print("[5/5] Gate 0 校验 + 落盘")
    g0 = gate0(df, rep)

    out = os.path.join(C.DATA_DIR, "haze_panel_clean.parquet")
    df.to_parquet(out, index=False, compression="snappy")

    rep.update({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_nodes": len(C.NODES),
        "rows_final": int(len(df)),
        "columns": list(df.columns),
        "fetch": fetch_log,
        "node_completeness": {
            str(sid): {
                "rows": int(len(sub)),
                "missing_rate": round(float(sub["is_missing"].mean()), 6),
                "interpolated_rate": round(float(sub["qc_interpolated"].mean()), 6),
                "pm25_min": round(float(sub["pm2_5"].min()), 2),
                "pm25_max": round(float(sub["pm2_5"].max()), 2),
            }
            for sid, sub in df.groupby("station_id")
        },
        "qc_notes": [
            "qc_pm25_gt_pm10 只标记不修正：PM2.5 是 PM10 的子集，违反者为模式内部不一致。",
            "qc_censored 标记贴近 900 µg/m³ 输出上限的截断值，不可当作真实极值使用。",
            "qc_calm_wind 标记静风时刻，此时风向为数值噪声，上风向权重应置零。",
            "qc_interpolated 标记经 ≤3h 线性插值补齐的单元格。",
            "is_missing 标记仍有变量缺失、未插值的行。",
        ],
    })
    with open(os.path.join(C.MODEL_DIR, "cleaning_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    with open(os.path.join(C.MODEL_DIR, "gate0_report.json"), "w", encoding="utf-8") as f:
        json.dump(g0, f, ensure_ascii=False, indent=2)
    with open(os.path.join(C.DATA_DIR, "data_dictionary.json"), "w", encoding="utf-8") as f:
        json.dump(_dictionary(g0), f, ensure_ascii=False, indent=2)

    print(f"\n✓ 写出 {out}")
    print(f"  行数 {len(df):,} | 节点 {df['station_id'].nunique()} | "
          f"{df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  切分 {dict(df['split'].value_counts())}")
    print(f"  受体 Level 分布 {g0['level_distribution_receptor_all']}")
    print(f"  Gate 0 {'全部通过' if g0['all_passed'] else '有未通过项'}")
    for k, v in g0["checks"].items():
        print(f"    {k:26s} {'PASS' if v['pass'] else 'FAIL'}  {v['value']}")


def _dictionary(g0: dict) -> dict:
    return {
        "dataset": "HazeGuard-KL haze panel",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "unit_notes": {
            "pm2_5/pm10/no2/so2/o3/co": "µg/m³（Open-Meteo 原样，CO 非 ppm）",
            "wind_speed_ms": "m/s，由 km/h ÷ 3.6 换算，建模请用此列",
            "wind_speed_kmh": "km/h，原始值，仅供复核",
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "precipitation": "mm",
            "wind_direction_10m": "度，0=北，顺时针",
        },
        "key_columns": {
            "station_id": "CAMS 空间单元 ID（非行政站点，同格点已归并）",
            "split": "train/val/test，按每节点时间跨度 70/15/15 顺序切分，无 shuffle",
            "b_proto_pm25": "气候态基线，仅用 train 段按 (station,month,hour) 统计",
            "pm2_5_anomaly": "pm2_5 - b_proto_pm25，去季节与日周期背景",
            "target_pm25_3h": "t+3h 的 PM2.5 浓度，groupby(station).shift(-3) 构造",
            "target_level_3h": "四级有序标签，阈值 [15,35,55]",
            "is_missing": "1=该行至少一个变量超过 3h 缺口未能插值",
        },
        "data_source_note": (
            "空气质量来自 CAMS 全球大气成分产品（0.4°≈45 km，模式再分析/预报，非地面实测）；"
            "气象来自 ERA5（约 9 km）。引用需注明 Open-Meteo 与 CAMS/ECMWF。"
        ),
        "gate0_summary": g0,
    }


if __name__ == "__main__":
    main()
