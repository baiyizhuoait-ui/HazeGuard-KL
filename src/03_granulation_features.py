"""3号 · 多尺度区间粒化与趋势粒化。

依据：仓库 CONTRIBUTING.md（3 号：多尺度趋势粒化与特征选择）与 docs/syllabus_coverage.md §3：
不做模糊 C 均值、隶属度和解粒化重构，只保留区间粒化与趋势粒化。

对每个格点、每个小时 t，取 PM2.5 尾随窗口 x = [x(t-W+1), ..., x(t)]，W ∈ {6, 12}：
  gran_low_{W}h / gran_core_{W}h / gran_up_{W}h   窗口下界、中枢（均值）、上界，µg/m³
  gran_width_{W}h                                  上界 - 下界，µg/m³
  gran_slope_{W}h                                  窗口内最小二乘斜率，µg/m³/h
都是窗口统计量，不拟合任何参数，所以不存在折内/折外的泄露问题。

窗口规则与 02 相同：按 station_id × split 在完整小时网格上取；qc_interpolated 或 is_missing
的行置空；窗口内任一小时为空，该行粒化特征为 NaN。

输出：data/feat_granulation.parquet（全面板，主键 + 10 列；溯源信息写在 parquet 元数据 hazeguard 键里）
      figures/fig_granulation_bands.png（受体 train 期 12h 中枢最高的时刻前后各 5 天）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parents[1]
IN_REPO = (HERE / "src" / "utils_metrics.py").exists()   # 在仓库 src/ 里运行时，输入和输出都是仓库根目录
DEFAULT_REPO = HERE if IN_REPO else HERE.parent / "hazeguard-kl"
DEFAULT_OUT = HERE if IN_REPO else HERE / "02_outputs"
WINDOWS = (6, 12)
ELIGIBLE = ("audit_model_eligible",)
KEYS = ["station_id", "timestamp"]
LEVELS = (15.0, 35.0, 55.0)
HALF_SPAN = pd.Timedelta(days=5)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def table_digest(repo: Path, path: Path) -> str:
    """单个文件取文件 SHA256；2 号的有序分片目录取清单摘要（与审计 JSON 的 output_sha256 同口径）。"""
    if path.is_file():
        return sha256(path)
    sys.path.insert(0, str(repo / "src"))
    from stage2_storage import digest
    return digest(path)


def eligibility(repo: Path, df: pd.DataFrame):
    """合格标记：先在特征表里找，找不到再看 data/audit_stage2.parquet（QN-001 允许的拆分方式）。"""
    for c in ELIGIBLE:
        if c in df.columns:
            return c, df[c].astype(bool).to_numpy()
    side = repo / "data" / "audit_stage2.parquet"
    col = next((c for c in ELIGIBLE if side.exists() and c in pq.read_schema(side).names), None)
    if col is None:
        raise ValueError(f"no eligibility column {ELIGIBLE} in the feature table or {side.name}")
    flag = df[KEYS].merge(pd.read_parquet(side, columns=KEYS + [col]), on=KEYS, how="left",
                          validate="one_to_one")[col]
    if flag.isna().any():
        raise ValueError(f"{side.name} does not cover every row of the feature table")
    return f"{side.name}:{col}", flag.astype(bool).to_numpy()


def trailing_windows(df: pd.DataFrame, w: int) -> np.ndarray:
    """(n, w) matrix, columns ordered oldest -> t; NaN where any hour is missing."""
    clean = df["qc_interpolated"].eq(0) & df["is_missing"].eq(0)
    x_all = df["pm2_5"].where(clean).to_numpy(float)
    ts_all = df["timestamp"].to_numpy()
    out = np.full((len(df), w), np.nan)
    for idx in df.groupby(["station_id", "split"], observed=True).indices.values():
        idx = idx[np.argsort(ts_all[idx])]
        ts = pd.DatetimeIndex(ts_all[idx])
        grid = pd.date_range(ts[0], ts[-1], freq="h", unit=ts.unit)
        s = pd.Series(x_all[idx], index=ts).reindex(grid)
        lagged = np.column_stack([s.shift(k).to_numpy() for k in range(w - 1, -1, -1)])
        out[idx] = lagged[grid.get_indexer(ts)]
    return out


def granules(X: np.ndarray, w: int) -> dict:
    t = np.arange(w) - (w - 1) / 2
    low, up = X.min(1), X.max(1)
    return {f"gran_low_{w}h": low, f"gran_core_{w}h": X.mean(1), f"gran_up_{w}h": up,
            f"gran_width_{w}h": up - low, f"gran_slope_{w}h": X @ t / (t @ t)}


def plot_bands(seg: pd.DataFrame, station: str, peak: pd.Timestamp, path: Path) -> None:
    t = seg["timestamp"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 3, 2]})
    for ax, w in zip(axes[:2], sorted(WINDOWS, reverse=True)):
        ax.fill_between(t, seg[f"gran_low_{w}h"], seg[f"gran_up_{w}h"], color="#b7d3f6", lw=0,
                        label=f"{w}-h granule [low, up]")
        ax.plot(t, seg[f"gran_core_{w}h"], color="#2a78d6", lw=2, label=f"{w}-h core (window mean)")
        ax.plot(t, seg["pm2_5"], color="#0b0b0b", lw=0.9, label="Hourly PM2.5")
        for level, th in enumerate(LEVELS, start=1):
            ax.axhline(th, color="#898781", ls="--", lw=0.7)
            ax.annotate(f"L{level} ≥ {th:g}", (1.0, th), xycoords=("axes fraction", "data"), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=7.5, color="#52514e")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.set_title(f"{w}-hour trailing window", fontsize=9, loc="left")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    axes[2].plot(t, seg["gran_slope_6h"], color="#2a78d6", lw=1.4, label="6-h slope")
    axes[2].plot(t, seg["gran_slope_12h"], color="#eb6834", lw=1.4, label="12-h slope")
    axes[2].axhline(0, color="#898781", lw=0.7)
    axes[2].set_ylabel("Slope (µg/m³/h)")
    axes[2].set_title("Trend granule: least-squares slope inside the window", fontsize=9, loc="left")
    axes[2].legend(loc="upper left", fontsize=8, frameon=False)
    for ax in axes:
        ax.axvline(peak, color="#c3c2b7", lw=0.8)
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Interval and trend granules of PM2.5 at {station} (CAMS grid cell, train period); "
                 f"grey line = hour with the highest 12-h core in receptor train data", fontsize=10)
    fig.autofmt_xdate()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    feat = args.repo / "data" / "feat_spatiotemporal.parquet"

    cols = KEYS + ["role", "split", "pm2_5", "qc_interpolated", "is_missing", *ELIGIBLE]
    present = set(pq.ParquetDataset(feat).schema.names)   # 单文件或分片目录都可以
    df = pd.read_parquet(feat, columns=[c for c in cols if c in present])
    elig, ok = eligibility(args.repo, df)
    split = df["split"].astype(str)
    rec = df["role"].eq("receptor").to_numpy() & ok

    out = df[KEYS].copy()
    for w in WINDOWS:
        for k, v in granules(trailing_windows(df, w), w).items():
            out[k] = v
    gran_cols = [c for c in out.columns if c.startswith("gran_")]
    out[gran_cols] = out[gran_cols].astype("float32")

    train_rec = rec & split.eq("train").to_numpy()
    peak_row = out.loc[train_rec, "gran_core_12h"].idxmax()
    station, peak = out.at[peak_row, "station_id"], out.at[peak_row, "timestamp"]
    near = (out["station_id"].eq(station) & split.eq("train") & out["timestamp"].between(peak - HALF_SPAN, peak + HALF_SPAN))
    seg = out.loc[near].assign(pm2_5=df.loc[near, "pm2_5"].where(
        df.loc[near, "qc_interpolated"].eq(0) & df.loc[near, "is_missing"].eq(0))).sort_values("timestamp")
    plot_bands(seg, station, peak, args.out / "figures" / "fig_granulation_bands.png")

    meta = {"generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.run(["git", "-C", str(args.repo), "rev-parse", "HEAD"],
                                         capture_output=True, text=True).stdout.strip(),
            "code_sha256": sha256(Path(__file__)), "input_sha256": table_digest(args.repo, feat),
            "eligible_column": elig, "windows": list(WINDOWS), "columns": gran_cols, "rows": len(out),
            "nan_fraction_all_rows": out[gran_cols].isna().mean().round(5).to_dict(),
            "nan_rows_eligible_receptor": {s: int(out.loc[rec & split.eq(s).to_numpy(), gran_cols].isna().any(axis=1).sum())
                                           for s in ("train", "val", "test")},
            "figure_window": {"station_id": station, "peak": str(peak), "half_span_days": HALF_SPAN.days}}
    table = pa.Table.from_pandas(out, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"hazeguard": json.dumps(meta).encode()})
    (args.out / "data").mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out / "data" / "feat_granulation.parquet", compression="zstd")
    print(f"wrote {len(gran_cols)} granulation columns for {len(out):,} rows; "
          f"eligible receptor rows with NaN: {meta['nan_rows_eligible_receptor']}; figure at {station} {peak}")


if __name__ == "__main__":
    main()
