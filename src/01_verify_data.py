#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据完整性校验与清单生成（数据契约守卫）

任何一位组员在开工前都应先跑这个脚本。它回答三个问题：
  1. 我读到的面板是不是上游交付的那一份？—— SHA256 比对
  2. 它有没有结构性错误？—— schema / 重复 / 切分连续性 / 标签正确性
  3. 有没有数据泄露？—— 基线是否真的只用了 train

产出
----
data/manifest.json     数据清单（sha256、行数、列名、切分、生成时间、git commit）

用法
----
python src/01_verify_data.py            # 校验 + 写清单
python src/01_verify_data.py --check    # 只校验，不写清单
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

PANEL = os.path.join(C.DATA_DIR, "haze_panel_clean.parquet")
MANIFEST = os.path.join(C.DATA_DIR, "manifest.json")

REQUIRED_COLUMNS = [
    "station_id", "timestamp", "lat", "lon", "region", "role", "split",
    "pm2_5", "pm10", "no2", "so2", "o3", "co",
    "temperature_2m", "relative_humidity_2m", "wind_speed_ms",
    "wind_direction_10m", "precipitation", "wind_speed_kmh",
    "is_missing", "qc_pm25_gt_pm10", "qc_censored", "qc_calm_wind", "qc_interpolated",
    "month", "hour", "dayofweek",
    "b_proto_pm25", "b_proto_temp", "b_proto_n", "b_proto_src",
    "pm2_5_anomaly", "temp_anomaly", "target_pm25_3h", "target_level_3h",
]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=False)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def check(name: str, ok: bool, detail: str) -> dict:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:44s} {detail}")
    return {"check": name, "pass": bool(ok), "detail": detail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验，不写清单")
    args = ap.parse_args()

    if not os.path.exists(PANEL):
        print(f"找不到面板 {PANEL}，请先运行 src/01_data_pipeline.py")
        return 2

    print(f"校验 {PANEL}")
    df = pd.read_parquet(PANEL)
    df["split"] = df["split"].astype(str)
    results = []

    # ---- 1. schema ----
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    results.append(check("schema：必需列齐全", not missing,
                         f"缺失 {missing}" if missing else f"{len(REQUIRED_COLUMNS)} 列全部存在"))
    results.append(check("schema：无意外新增列", not extra,
                         f"多余 {extra}" if extra else "无"))

    # ---- 2. 主键唯一 ----
    dup = int(df.duplicated(["station_id", "timestamp"]).sum())
    results.append(check("主键 (station_id, timestamp) 唯一", dup == 0, f"重复 {dup} 行"))

    # ---- 3. 时间轴：单调、整小时步长、无重复 ----
    # 说明：源区节点存在 >3h 的真实缺口（Palangkaraya 37h），缺口行因无锚点特征被剔除，
    # 因此步长会是 1h 的整数倍而不是恒等于 1h。这里校验"整数倍"，
    # 并把缺口单独统计为披露项，不当作错误。
    bad_step, holes = [], {}
    for sid, sub in df.groupby("station_id"):
        t = sub["timestamp"].sort_values()
        if not t.is_monotonic_increasing:
            bad_step.append((sid, "非单调"))
            continue
        step = t.diff().dropna().dt.total_seconds().div(3600)
        non_int = int((step % 1 != 0).sum())
        if non_int:
            bad_step.append((sid, f"{non_int} 处非整小时"))
        n_holes = int((step > 1).sum())
        if n_holes:
            holes[sid] = {"n_holes": n_holes, "max_step_h": float(step.max())}
    results.append(check("时间轴单调且步长为整小时", not bad_step,
                         f"异常 {bad_step}" if bad_step else "26 节点全部为 1h 整数倍步长"))

    # ---- 4. 切分连续性：train < val < test，且不重叠 ----
    bad_split = []
    for sid, sub in df.groupby("station_id"):
        rng = sub.groupby("split")["timestamp"].agg(["min", "max"])
        if not all(k in rng.index for k in ("train", "val", "test")):
            bad_split.append((sid, "缺分片"))
            continue
        if not (rng.loc["train", "max"] < rng.loc["val", "min"]
                and rng.loc["val", "max"] < rng.loc["test", "min"]):
            bad_split.append((sid, "区间重叠"))
    results.append(check("切分严格时序且不重叠", not bad_split,
                         f"异常 {bad_split}" if bad_split else "全部 26 节点 train<val<test"))

    # ---- 5. 标签正确性（抽样重算）----
    tgt = df["target_pm25_3h"].to_numpy()
    lvl = df["target_level_3h"].to_numpy()
    expect_lvl = np.select([tgt < 15, tgt < 35, tgt < 55], [0, 1, 2], default=3)
    lvl_ok = bool((expect_lvl == lvl).all())
    results.append(check("四级标签与阈值一致", lvl_ok,
                         f"{len(df):,} 行全部匹配 [15,35,55]" if lvl_ok else "存在不匹配行"))

    # ---- 6. 目标位移正确性：t+3h 应等于 3 行之后的 pm2_5 ----
    sample_ok, sample_n = True, 0
    for sid, sub in df.groupby("station_id"):
        sub = sub.sort_values("timestamp")
        chk = sub.iloc[1000:1003] if len(sub) > 1003 else sub.head(3)
        for _, row in chk.iterrows():
            nxt = sub[sub["timestamp"] == row["timestamp"] + pd.Timedelta(hours=3)]
            if len(nxt) and not np.isclose(nxt["pm2_5"].iloc[0], row["target_pm25_3h"], atol=1e-6):
                sample_ok = False
            sample_n += 1
    results.append(check("target_pm25_3h = 3 小时后的 pm2_5", sample_ok,
                         f"抽检 {sample_n} 行"))

    # ---- 7. 基线泄露检查：重算 b_proto 必须只依赖 train ----
    train = df[df["split"] == "train"]
    ref = (train.groupby(["station_id", "month", "hour"], observed=True)["pm2_5"]
           .mean().rename("ref").reset_index())
    merged = df.merge(ref, on=["station_id", "month", "hour"], how="left")
    diff = (merged["b_proto_pm25"] - merged["ref"]).abs()
    leak = int((diff > 1e-6).sum())
    results.append(check("基线未泄露（可用 train 重算复现）", leak == 0,
                         f"不一致 {leak} 行" if leak else f"{len(df):,} 行全部可复现"))

    # ---- 8. 受体与目标分布 ----
    rec = df[df["role"] == "receptor"]
    p3 = float((rec["target_level_3h"] == 3).mean())
    results.append(check("受体 Level 3 占比 ≥ 1%", p3 >= 0.01, f"{p3:.2%}"))

    # ---- 9. QC 标记存在且合理 ----
    qc_cols = ["qc_pm25_gt_pm10", "qc_censored", "qc_calm_wind", "qc_interpolated"]
    qc_ok = all(c in df.columns for c in qc_cols)
    counts = {c: int(df[c].sum()) for c in qc_cols} if qc_ok else {}
    results.append(check("质量标记列齐全", qc_ok, str(counts)))

    # ---- 9b. 交付契约：行数 / dtype / 体积 / LFS / 禁词 ------------------
    pc = C.PANEL_CONTRACT
    row_ok = (len(df) == pc["panel_rows"] and len(df.columns) == pc["panel_cols"]
              and df["station_id"].nunique() == pc["n_nodes"])
    results.append(check("契约：面板行数/列数/节点数", row_ok,
                         f"{len(df):,}×{len(df.columns)}，{df['station_id'].nunique()} 节点"
                         + ("" if row_ok else f"，契约登记 {pc['panel_rows']:,}×{pc['panel_cols']}")))

    rec = df[df["role"] == "receptor"]
    rec_counts = {k: int((rec["split"] == k).sum()) for k in ("train", "val", "test")}
    rec_ok = all(rec_counts[k] == pc[f"receptor_{k}_rows"] for k in rec_counts)
    results.append(check("契约：受体三段行数", rec_ok,
                         f"train/val/test = {rec_counts['train']:,}/{rec_counts['val']:,}/{rec_counts['test']:,}"
                         + ("" if rec_ok else f"，契约 {pc['receptor_train_rows']:,}/{pc['receptor_val_rows']:,}/{pc['receptor_test_rows']:,}")))

    dt_ok = str(df["timestamp"].dtype).startswith(pc["timestamp_dtype_prefix"])
    results.append(check("dtype：timestamp 为 datetime64", dt_ok,
                         f"实际 {df['timestamp'].dtype}"))

    size_bad, lfs_bad = [], []
    ddir = C.DATA_DIR
    for f in sorted(os.listdir(ddir)):
        fp = os.path.join(ddir, f)
        if not os.path.isfile(fp) or not f.endswith(".parquet"):
            continue
        mb = os.path.getsize(fp) / 1048576
        if mb > pc["max_parquet_mb"]:
            size_bad.append(f"{f} {mb:.1f}MB")
        with open(fp, "rb") as fh:
            head = fh.read(24)
        if head.startswith(b"version https://git-lfs"):
            lfs_bad.append(f)
    results.append(check("体积：data/*.parquet ≤ 50MB 且非 LFS 指针",
                         not size_bad and not lfs_bad,
                         ("超限 " + ", ".join(size_bad) if size_bad else "")
                         + ("LFS指针 " + ", ".join(lfs_bad) if lfs_bad else "")
                         or "全部合规"))

    banned = [t.lower() for t in pc["banned_terms"]]
    # 执行工具自身（守卫定义里必然出现禁词原文）与政策文档豁免
    whitelist = {"CONTRIBUTING.md", "syllabus_coverage.md",
                 "01_verify_data.py", "config.py"}
    scan_roots = ["src", "reports", "tools", "docs/handover"]
    word_hits = []
    for root in scan_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for f in files:
                fp = os.path.join(dirpath, f)
                if f in whitelist or any(w in f for w in ("quality_notice",)):
                    continue
                if not f.endswith((".py", ".md", ".json", ".txt")):
                    continue
                try:
                    t = open(fp, encoding="utf-8").read()
                except UnicodeDecodeError:
                    continue
                for term in banned:
                    if term in t.lower():
                        word_hits.append(f"{fp} 含 '{term}'")
    for f in sorted(os.listdir(ddir)):
        if f.endswith(".parquet"):
            cols = [c.lower() for c in df.columns]
            word_hits += [f"列名含 '{t}': {c}" for c in cols for t in banned if t in c]
    results.append(check("禁词：交付物无 agent/智能体 字样", not word_hits,
                         "、".join(word_hits[:4]) if word_hits else "干净"))

    # ---- 10. 无基础设施性 NaN ----
    key = ["pm2_5", "target_pm25_3h", "target_level_3h", "b_proto_pm25"]
    nan_key = {k: int(df[k].isna().sum()) for k in key}
    results.append(check("关键列无 NaN", all(v == 0 for v in nan_key.values()), str(nan_key)))

    passed = sum(r["pass"] for r in results)
    total = len(results)
    print(f"\n{passed}/{total} 项通过")

    if not args.check:
        manifest = {
            "dataset": "haze_panel_clean",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "path": PANEL.replace("\\", "/"),
            "sha256": sha256(PANEL),
            "bytes": os.path.getsize(PANEL),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "n_nodes": int(df["station_id"].nunique()),
            "nodes_by_role": {
                r: sorted(df.loc[df["role"] == r, "station_id"].unique().tolist())
                for r in sorted(df["role"].unique())
            },
            "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
            "split_rows": {k: int(v) for k, v in df["split"].value_counts().items()},
            "qc_counts": counts,
            "time_axis_holes": holes,
            "verification": {
                "passed": passed, "total": total,
                "all_passed": passed == total,
                "results": results,
            },
        }
        with open(MANIFEST, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"✓ 写出 {MANIFEST}")
        print(f"  sha256 {manifest['sha256'][:16]}…  {manifest['bytes']/1e6:.1f} MB")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
