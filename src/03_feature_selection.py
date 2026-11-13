"""3号 · 特征选择与消融（CONTRIBUTING.md 3 号职责；方法按 docs/syllabus_coverage.md W3：
互信息 + RFE + permutation importance）。

候选 = 2 号审计白名单 model_feature_columns（唯一预测白名单）+ 3 号粒化特征（去掉与 02 完全相同的列）。
粒化和填补都不拟合参数，所以下面的选择器是唯一用到标签的步骤，并且只在训练数据上拟合。
选择器只用 receptor ∧ 合格 ∧ train（按 timestamp 排序），不碰 val/test：
  a) mutual_info_classif（离散等级标签），取前 N_TOP
  b) 准常量过滤 + Spearman |rho| > 0.92 贪心剪枝（按与 target_pm25_3h 的 |rho| 优先保留）
  c) RF permutation importance：train 末 20% 时间作内部留出，留出前 3h 隔离，取前 N_TOP
  三者并集 → RF-RFE（step=1）→ Top-25。
消融：同一默认 RF、3 个种子，在 val 上评估六项指标（src/utils_metrics.py），附 M0 持续性；
  ① 原始 → ② + 滞后/滚动 → ③ + 空间（近场与远场）→ ④ + 粒化 → ⑤ 选择后的 Top-25。

features_final.parquet 只含 receptor ∧ 合格行，Top-25 全部为有限值（4 号读取时会校验）。
空间列缺失按下面的规则填，都不拟合任何参数；选择与消融都在填补后的数据上做：
  近场：没有有效上风向邻居 → 邻域等于本格点：平流列填本格点同滞后 PM2.5，梯度填 0
  远场：长窗口均值历史不够（分段开头）→ 用最接近的更短窗口均值（48h → 24h → 12h）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer

HERE = Path(__file__).resolve().parents[1]
IN_REPO = (HERE / "src" / "utils_metrics.py").exists()   # 在仓库 src/ 里运行时，输入和输出都是仓库根目录
DEFAULT_REPO = HERE if IN_REPO else HERE.parent / "hazeguard-kl"
DEFAULT_OUT = HERE if IN_REPO else HERE / "02_outputs"
TOP_K, N_TOP, RHO_MAX, MODE_MAX = 25, 30, 0.92, 0.99
SEED, SEEDS, INNER_FRAC, GAP_H = 42, (42, 43, 44), 0.2, 3
SPATIAL = ("upwind_", "spatial_", "far_source_", "u_wind", "v_wind")
ELIGIBLE = ("audit_model_eligible",)
KEYS, ID_COLS = ["station_id", "timestamp"], ["station_id", "timestamp", "region", "role", "split"]
TARGETS = ["target_pm25_3h", "target_level_3h"]
FAR_MEAN = re.compile(r"far_source_pm25_mean_(\d+)h")


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


def fill_spatial(frame: pd.DataFrame, cols) -> dict:
    """Parameter-free fills for spatial NaN (rules in the module docstring). In place; returns counts."""
    split, counts = frame["split"].astype(str), {}
    far = sorted((int(m[1]), c) for c in frame.columns if (m := FAR_MEAN.fullmatch(c)))
    for c in cols:
        miss = frame[c].isna()
        lag = re.fullmatch(r"upwind_pm25_advection_(\d+)h", c)
        win = FAR_MEAN.fullmatch(c)
        if not miss.any():
            continue
        if lag and f"pm2_5_lag_{lag[1]}" in frame:
            frame[c] = frame[c].where(~miss, frame[f"pm2_5_lag_{lag[1]}"])
        elif c == "spatial_pm25_gradient":
            frame[c] = frame[c].fillna(0.0)
        elif win:
            for _, shorter in reversed([f for f in far if f[0] < int(win[1])]):
                frame[c] = frame[c].fillna(frame[shorter])
        else:
            continue
        counts[c] = {s: int((miss & split.eq(s)).sum()) for s in ("train", "val", "test")}
    return counts


def rf(seed=SEED):
    return RandomForestClassifier(random_state=seed, n_jobs=-1)


def strategy_mi(X, y):
    mi = mutual_info_classif(X.fillna(X.median()), y, random_state=SEED)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


def strategy_filter(X, y_reg):
    mode = X.apply(lambda c: c.value_counts(normalize=True, dropna=False).iloc[0])
    quasi = mode.index[(mode >= MODE_MAX) | (X.std() == 0)].tolist()
    R = X.drop(columns=quasi).rank()
    rel = R.corrwith(y_reg.rank()).abs().sort_values(ascending=False)
    rho = R.corr().abs()
    kept, dropped = [], {}
    for c in rel.index:
        hit = [k for k in kept if rho.loc[c, k] > RHO_MAX]
        if hit:
            dropped[c] = {"kept": hit[0], "abs_rho": round(float(rho.loc[c, hit[0]]), 4)}
        else:
            kept.append(c)
    return rel[kept], quasi, dropped


def strategy_perm(tr, cols, scorer):
    ts = tr["timestamp"]
    cut = np.sort(ts.unique())[int(ts.nunique() * (1 - INNER_FRAC))]
    fit, hold = ts + pd.Timedelta(hours=GAP_H) < cut, ts >= cut
    model = rf().fit(tr.loc[fit, cols], tr.loc[fit, "target_level_3h"])
    r = permutation_importance(model, tr.loc[hold, cols], tr.loc[hold, "target_level_3h"],
                               scoring=scorer, n_repeats=5, random_state=SEED)
    table = pd.DataFrame({"mean": r.importances_mean, "std": r.importances_std}, index=cols)
    info = {"cut": str(cut), "fit_rows": int(fit.sum()), "holdout_rows": int(hold.sum()), "n_repeats": 5}
    return table.sort_values("mean", ascending=False), info


def ablate(tr, va, cols, evaluate):
    runs = []
    for s in SEEDS:
        pred = rf(s).fit(tr[cols], tr["target_level_3h"]).predict(va[cols])
        runs.append(evaluate(va["target_level_3h"], pred))
    keys = [k for k in runs[0] if k not in ("n_samples", "n_high_risk")]
    return {"n_features": len(cols), "features": list(cols), "seeds": list(SEEDS), "runs": runs,
            "mean": {k: float(np.mean([r[k] for r in runs])) for k in keys},
            "std": {k: float(np.std([r[k] for r in runs], ddof=1)) for k in keys}}


def plot_ablation(groups, m0, path: Path):
    panels = [("Macro_F1", "Macro-F1 (higher is better)"),
              ("High_Risk_Recall", "High-risk recall, union L2+L3 (higher is better)"),
              ("Avg_Asymmetric_Cost", "Average asymmetric cost (lower is better)")]
    labels = list(groups)[::-1]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (key, title) in zip(axes, panels):
        mean = np.array([groups[g]["mean"][key] for g in labels])
        std = np.array([groups[g]["std"][key] for g in labels])
        ax.errorbar(mean, y, xerr=std, fmt="o", ms=7, color="#2a78d6", ecolor="#2a78d6", capsize=3, lw=1.5)
        for xi, yi in zip(mean, y):
            ax.annotate(f"{xi:.4f}", (xi, yi), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=7.5, color="#52514e")
        ax.axvline(m0[key], color="#898781", ls="--", lw=1.2)
        ax.annotate(f"M0 persistence {m0[key]:.4f}", (m0[key], -0.5), xytext=(3, 0),
                    textcoords="offset points", rotation=90, ha="left", va="bottom",
                    fontsize=7.5, color="#52514e")
        ax.margins(x=0.08)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylim(-0.6, len(labels) - 0.3)
        ax.grid(axis="x", color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, labels, fontsize=8.5)
    fig.suptitle(f"Feature ablation on val (receptor cells, n={m0['n_samples']:,}; default RF, "
                 f"mean ± s.d. over seeds {', '.join(map(str, SEEDS))})", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    sys.path.insert(0, str(args.repo / "src"))
    from config import LEVEL_THRESHOLDS
    from utils_metrics import evaluate_predictions
    scorer = make_scorer(lambda yt, yp: evaluate_predictions(yt, yp)["Macro_F1"])

    feat = args.repo / "data" / "feat_spatiotemporal.parquet"
    gran_path = args.out / "data" / "feat_granulation.parquet"
    audit = json.loads((args.repo / "models" / "metrics_stage2_features.json").read_text(encoding="utf-8"))
    base = audit["model_feature_columns"]
    present = [c for c in ELIGIBLE if c in pq.ParquetDataset(feat).schema.names]
    df = pd.read_parquet(feat, columns=list(dict.fromkeys(ID_COLS + TARGETS + present + base)))
    elig, ok = eligibility(args.repo, df)
    gran = pd.read_parquet(gran_path)
    gran_meta = json.loads(pq.read_schema(gran_path).metadata[b"hazeguard"])
    if not (df[KEYS].to_numpy() == gran[KEYS].to_numpy()).all():
        raise ValueError("feat_granulation.parquet row order differs; rerun 03_granulation_features.py")
    if gran_meta["input_sha256"] != table_digest(args.repo, feat):
        raise ValueError("feat_granulation.parquet was built from another feature table; rerun 03_granulation_features.py")
    gran_cols = [c for c in gran.columns if c.startswith("gran_")]
    df = pd.concat([df, gran[gran_cols]], axis=1)

    rec = df[df["role"].eq("receptor").to_numpy() & ok].copy()
    new = [c for c in base if c in audit["new_columns"]]
    spatial = [c for c in new if c.startswith(SPATIAL)]
    filled = fill_spatial(rec, spatial)
    split = rec["split"].astype(str)
    tr = rec[split.eq("train")].sort_values(KEYS[::-1], kind="stable")
    va = rec[split.eq("val")].sort_values(KEYS[::-1], kind="stable")

    dups = {f: u for f in gran_cols for u in base
            if np.allclose(tr[f], tr[u], atol=1e-4, equal_nan=True)}
    gran_unique = [c for c in gran_cols if c not in dups]
    cands = base + gran_unique

    X, y = tr[cands], tr["target_level_3h"]
    mi = strategy_mi(X, y)
    rel, quasi, corr_drop = strategy_filter(X, tr["target_pm25_3h"])
    perm, perm_info = strategy_perm(tr, cands, scorer)
    top_a, set_b = list(mi.index[:N_TOP]), list(rel.index)
    top_c = list(perm.index[perm["mean"] > 0][:N_TOP])
    union = [c for c in cands if c in set(top_a) | set(set_b) | set(top_c)]
    rfe = RFE(rf(), n_features_to_select=TOP_K, step=1).fit(tr[union], y)
    final_imp = pd.Series(rfe.estimator_.feature_importances_, index=np.array(union)[rfe.support_])
    selected = list(final_imp.sort_values(ascending=False).index)

    raw = [c for c in base if c not in new]
    temporal = [c for c in new if c not in spatial]
    group_cols = {"1 Raw only": raw, "2 + Lag/Rolling": raw + temporal,
                  "3 + Spatial (near & far field)": raw + temporal + spatial,
                  "4 + Granulation": raw + temporal + spatial + gran_unique,
                  f"5 Selected Top-{TOP_K}": selected}
    groups = {}
    for name, cols in group_cols.items():
        groups[name] = ablate(tr, va, cols, evaluate_predictions)
        print(f"{name:28s} n={len(cols):3d}  " + "  ".join(
            f"{k}={groups[name]['mean'][k]:.4f}±{groups[name]['std'][k]:.4f}"
            for k in ("Macro_F1", "High_Risk_Recall", "Avg_Asymmetric_Cost")))
    m0 = evaluate_predictions(va["target_level_3h"], np.searchsorted(LEVEL_THRESHOLDS, va["pm2_5"], side="right"))
    plot_ablation(groups, m0, args.out / "figures" / "fig_feature_ablation.png")

    final = rec[ID_COLS + TARGETS + selected].reset_index(drop=True)
    bad = [c for c in selected if not np.isfinite(final[c].to_numpy(float)).all()]
    if bad:
        raise ValueError(f"non-finite Top-{TOP_K} values in {bad}; extend fill_spatial for them")
    final.to_parquet(args.out / "data" / "features_final.parquet", index=False, compression="zstd")

    stamp = {"generated_at": datetime.now(timezone.utc).isoformat(),
             "git_commit": subprocess.run(["git", "-C", str(args.repo), "rev-parse", "HEAD"],
                                          capture_output=True, text=True).stdout.strip(),
             "code_sha256": sha256(Path(__file__)),
             "input_sha256": {"feat_spatiotemporal": table_digest(args.repo, feat),
                              "feat_granulation": sha256(gran_path)}}
    selection = {**stamp, "selected_features": selected, "n_selected": len(selected),
                 "granulation": gran_meta,
                 "id_columns": ID_COLS, "target_columns": TARGETS,
                 "row_filter": f"role == receptor & {elig}",
                 "rows_by_split": final["split"].astype(str).value_counts().to_dict(),
                 "spatial_fill": {"rule": "upwind_pm25_advection_Kh <- pm2_5_lag_K; spatial_pm25_gradient <- 0; "
                                          "far_source_pm25_mean_Wh <- nearest shorter far_source mean window; "
                                          "only where the upstream value is NaN",
                                  "filled_rows": filled},
                 "fit_rows": "role==receptor & eligible & split==train", "train_rows": len(tr),
                 "params": {"top_k": TOP_K, "n_top_per_strategy": N_TOP, "spearman_abs_max": RHO_MAX,
                            "quasi_constant_mode_share": MODE_MAX, "seed": SEED, "rfe_step": 1},
                 "candidates": cands, "n_candidates": len(cands), "exact_duplicates_removed": dups,
                 "strategy_a_mutual_info": {"top": top_a, "scores": mi.round(5).to_dict()},
                 "strategy_b_variance_spearman": {"quasi_constant_removed": quasi,
                                                  "correlated_removed": corr_drop,
                                                  "survivors_abs_rho_target": rel.round(4).to_dict()},
                 "strategy_c_permutation": {"top": top_c, "holdout": perm_info,
                                            "importance": perm.round(5).to_dict(orient="index")},
                 "union": union, "n_union": len(union),
                 "rfe_ranking": dict(zip(union, map(int, rfe.ranking_))),
                 "final_rf_importance": final_imp.round(5).to_dict(),
                 "selected_in_strategies": {c: [k for k, s in (("a", top_a), ("b", set_b), ("c", top_c))
                                                if c in s] for c in selected}}
    metrics = {**stamp, "evaluated_on": "val, role==receptor & eligible, spatial NaN filled "
                                        "(see selected_features.json spatial_fill)", "val_rows": len(va),
               "model": "RandomForestClassifier() defaults, n_jobs=-1",
               "high_risk_recall_definition": "union: P(pred in {2,3} | true in {2,3})",
               "ablation": groups, "metrics_persistence_M0_val": m0}
    (args.out / "models").mkdir(parents=True, exist_ok=True)
    for name, obj in (("selected_features.json", selection), ("metrics_stage3.json", metrics)):
        (args.out / "models" / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"selected {len(selected)} of {len(cands)} candidates (union {len(union)}); "
          f"features_final {final.shape}")


if __name__ == "__main__":
    main()
