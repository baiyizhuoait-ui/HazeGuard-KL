"""合并闸门检查：一个分支能不能合进 main，跑这个。

用法：
    python tools/check_merge_gate.py                # 检查当前分支
    python tools/check_merge_gate.py feat/05-ordinal-rf
    python tools/check_merge_gate.py --all          # 体检所有 feat/* 分支

检查九项，任何一项 FAIL 就不该点合并：
    1. 落后 main 的提交数（不是 0 就先 merge main，别带着分叉提 PR）
    2. 交付物齐全（对照 CONTRIBUTING 锁定表）
    3. 越权改动（动了别人的文件）
    4. 禁词（文件名与正文不得出现 agent / 智能体）
    5. 体积（parquet ≤ 50MB，且不能是 LFS 指针）
    6. 交接文档
    7. 报告章节
    8. 指标文件（存在，且带 generated_at / git_commit）
    9. test 段使用次数（有登记的话不能大于 1）

只读 git 对象，不切换分支、不碰工作区。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PROTECTED = {
    "CONTRIBUTING.md", "README.md", ".gitignore", "requirements.txt",
    "src/config.py", "src/utils_metrics.py",
    "src/01_data_pipeline.py", "src/01_baseline_dt.py", "src/01_verify_data.py",
}

# 与 CONTRIBUTING.md 第 1 节的锁定表同步；改表必须同时改这里
EXPECTED = {
    "02": {
        "files": [
            "data/feat_spatiotemporal.parquet",
            "models/metrics_stage2_eda.json",
            "models/metrics_stage2_features.json",
            "models/metrics_stage2_verification.json",
            "figures/fig_eda_diurnal_seasonal.png",
            "figures/fig_eda_correlation_matrix.png",
            "figures/fig_eda_class_imbalance.png",
            "figures/fig_eda_wind_rose_pollution.png",
            "figures/fig_eda_spatial_crosscorr.png",
        ],
        "handover": "02_eda_and_spatiotemporal.md",
        "report_prefix": "sec2_3",
        "metrics": ["models/metrics_stage2_eda.json", "models/metrics_stage2_features.json"],
    },
    "03": {
        "files": [
            "data/features_final.parquet",
            "models/selected_features.json",
            "models/metrics_stage3.json",
            "figures/fig_feature_ablation.png",
            "figures/fig_granulation_bands.png",
        ],
        "handover": "03_granulation_and_selection.md",
        "report_prefix": "sec2_4_2",
        "metrics": ["models/metrics_stage3.json"],
    },
    "04": {
        "files": [
            "models/rf_nominal.joblib",
            "models/rf_balanced.joblib",
            "models/boosting.joblib",
            "models/metrics_stage4.json",
            "figures/fig_rf_complexity_curve.png",
            "figures/fig_window_sensitivity.png",
            "figures/fig_boosting_vs_rf.png",
        ],
        "handover": "04_nominal_rf_and_boosting.md",
        "report_prefix": "sec2_6",
        "metrics": ["models/metrics_stage4.json"],
    },
    "05": {
        "files": [
            "models/rf_regressor.joblib",
            "models/ordinal_rf.joblib",
            "models/ordinal_probs_val_test.npz",
            "models/metrics_stage5.json",
            "figures/fig_ordinal_error_distribution.png",
            "figures/fig_interval_band.png",
        ],
        "handover": "05_ordinal_and_interval.md",
        "report_prefix": ("sec2_6", "sec2_7"),
        "metrics": ["models/metrics_stage5.json"],
    },
    "06": {
        "files": [
            "models/cost_thresholds.json",
            "models/all_models_benchmark.csv",
            "models/loso_results.csv",
            "models/seasonal_metrics.csv",
            "figures/fig_perm_importance.png",
            "figures/fig_pdp_wind_humidity.png",
            "figures/fig_roc_high_risk.png",
            "figures/fig_confusion_matrices_comparison.png",
        ],
        "handover": "06_cost_and_explain.md",
        "report_prefix": "sec3",
        "metrics": ["models/cost_thresholds.json"],
    },
    "07": {
        "files": ["app.py", "reports/HazeGuard_KL_Demonstration_Report.md"],
        "handover": None,
        "report_prefix": None,
        "metrics": [],
    },
}

BANNED = re.compile(r"\bagent\b|智能体", re.IGNORECASE)
POLICY_EXEMPT = {"docs/quality_notice_001.md", "CONTRIBUTING.md", "docs/spec_v3.md",
                 "tools/check_merge_gate.py"}
MAX_PARQUET_MB = 50.0
LFS_MAGIC = b"version https://git-lfs"


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], capture_output=True)
    return r.stdout.decode("utf-8", errors="replace")


def tree(branch: str) -> dict[str, int]:
    out = {}
    for line in git("ls-tree", "-r", "-l", f"origin/{branch}").splitlines():
        parts = line.split(None, 4)
        if len(parts) == 5:
            out[parts[4]] = int(parts[3])
    return out


def blob(branch: str, path: str, limit: int = 2_000_000) -> bytes:
    r = subprocess.run(["git", "show", f"origin/{branch}:{path}"], capture_output=True)
    return r.stdout[:limit]


def prose_check(branch: str, paths: dict[str, int]) -> tuple[bool, str]:
    """对本分支的 reports/*.md 跑 check_prose，返回 (是否通过, 说明)。"""
    reps = [p for p in paths if p.startswith("reports/") and p.endswith(".md")]
    if not reps:
        return False, "分支里没有报告文件"
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_prose import check_file  # 同目录，复用同一套规则

    bad = []
    with tempfile.TemporaryDirectory() as td:
        for p in reps:
            fp = Path(td) / Path(p).name
            fp.write_bytes(blob(branch, p))
            fails = check_file(str(fp))["fails"]
            if fails:
                bad.append(f"{Path(p).name} — " + "；".join(fails))
    return (not bad), ("干净" if not bad else " | ".join(bad))


PERIODIC_BASES = ("hour", "month", "dayofweek", "wind_direction")


def feature_pair_check(branch: str) -> tuple[bool, str] | None:
    """周期特征的 sin/cos 必须成对。

    只留 sin 的话，24 个钟点会塌成 13 个不同取值：11 组两两同值，
    而且两个起报接缝时刻都在里面（04↔08、16↔20）。
    模型没法区分 08:00 和 04:00，而 08:00 恰恰是跳变最大的时刻之一。
    """
    paths = tree(branch)
    if "models/selected_features.json" not in paths:
        return None
    try:
        payload = json.loads(blob(branch, "models/selected_features.json").decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False, "selected_features.json 读不出来"
    feats = set(payload.get("selected_features") or [])
    bad = []
    for base in PERIODIC_BASES:
        a, b = f"{base}_sin", f"{base}_cos"
        if (a in feats) != (b in feats):
            bad.append(f"{a}/{b} 只进了一个")
    if bad:
        return False, "；".join(bad) + f"（共 {len(feats)} 个特征）"
    pairs = [f"{b}_sin/cos" for b in PERIODIC_BASES if f"{b}_sin" in feats]
    return True, ("成对：" + "、".join(pairs)) if pairs else "无周期特征"


def module_of(branch: str) -> str | None:
    m = re.match(r"(?:feat|fix)/(\d{2})-", branch)
    return m.group(1) if m else None


def audit(branch: str) -> tuple[list[tuple[str, bool, str]], str]:
    """返回 [(检查项, 是否通过, 说明)] 与整体结论。"""
    rows: list[tuple[str, bool, str]] = []
    mod = module_of(branch)
    if mod is None:
        return [("分支命名", False, f"{branch} 不符合 feat/NN-xxx")], "FAIL"
    if mod not in EXPECTED:
        return [("分支命名", False, f"模块号 {mod} 未在检查表中登记")], "FAIL"

    paths = tree(branch)
    if not paths:
        return [("分支存在", False, "远程分支不存在或为空")], "FAIL"

    # 0 已合入？（合过之后分支自然"落后"，那不是失败）
    merged = subprocess.run(
        ["git", "merge-base", "--is-ancestor", f"origin/{branch}", "origin/main"]
    ).returncode == 0
    if merged:
        # 区分"已合并"与"空分支恰好指向旧 main"：看 main 上有没有这个模块的产物
        main_paths = tree("main")
        spec = EXPECTED[mod]
        landed = all(p in main_paths or any(x.startswith(p + "/") for x in main_paths)
                     for p in spec["files"])
        if landed:
            return [("已合入 main", True, "全部产物都在 main 上，此分支可删")], "MERGED"
        return [("未开工", True, "分支指向旧 main，尚无自己的提交")], "UNSTARTED"

    # 1 落后 main
    behind = int(git("rev-list", "--count", f"origin/{branch}..origin/main").strip() or 0)
    rows.append(("落后 main", behind == 0,
                 "与 main 同步" if behind == 0 else f"落后 {behind} 个提交，先 merge main"))

    # 2 交付物
    spec = EXPECTED[mod]
    missing = [p for p in spec["files"] if p not in paths
               and not any(x.startswith(p + "/") for x in paths)]
    rows.append(("交付物", not missing,
                 "齐全" if not missing else "缺 " + "、".join(missing)))

    # 3 越权
    base = git("merge-base", "origin/main", f"origin/{branch}").strip()
    changed = [p for p in git("diff", "--name-only", base, f"origin/{branch}").splitlines() if p]
    bad = sorted(p for p in changed if p in PROTECTED)
    rows.append(("越权改动", not bad,
                 "无" if not bad else "动了别人的文件：" + "、".join(bad)))

    # 4 禁词
    name_hits = [p for p in paths if re.search(r"agent", p, re.I)]
    body_hits = []
    for p in changed:
        if p in POLICY_EXEMPT or not p.endswith((".py", ".md", ".json", ".txt")):
            continue
        if paths.get(p, 0) > 2_000_000:
            continue
        if BANNED.search(blob(branch, p).decode("utf-8", errors="replace")):
            body_hits.append(p)
    rows.append(("禁词", not name_hits and not body_hits,
                 "干净" if not (name_hits or body_hits)
                 else f"文件名 {name_hits}、正文 {body_hits}"))

    # 5 体积
    big = [(p, s / 1048576) for p, s in paths.items()
           if p.endswith(".parquet") and s > MAX_PARQUET_MB * 1048576]
    lfs = [p for p, s in paths.items()
           if p.endswith(".parquet") and s < 200 and blob(branch, p, 40).startswith(LFS_MAGIC)]
    rows.append(("体积", not big and not lfs,
                 "合规" if not (big or lfs)
                 else "超限 " + "、".join(f"{p} {s:.0f}MB" for p, s in big)
                      + ("；LFS 指针 " + "、".join(lfs) if lfs else "")))

    # 6 交接
    if spec["handover"]:
        hp = "docs/handover/" + spec["handover"]
        rows.append(("交接文档", hp in paths, hp if hp in paths else f"缺 {hp}"))

    # 7 报告
    if spec["report_prefix"]:
        rp = [p for p in paths if p.startswith("reports/") and p.endswith(".md")
              and Path(p).name.startswith(spec["report_prefix"])]
        rows.append(("报告章节", bool(rp), rp[0] if rp else f"缺 reports/{spec['report_prefix']}*.md"))

    # 7b 报告文字（课程 C.2：AI 生成文本零分）
    ok_prose, note_prose = prose_check(branch, paths)
    rows.append(("报告文字", ok_prose, note_prose))

    # 7c 周期特征成对（sin/cos）
    pair = feature_pair_check(branch)
    if pair is not None:
        rows.append(("周期特征配对", pair[0], pair[1]))

    # 8 指标文件
    for mp in spec["metrics"]:
        if mp not in paths:
            rows.append((f"指标 {Path(mp).name}", False, "文件不存在"))
            continue
        txt = blob(branch, mp).decode("utf-8", errors="replace")
        has = ("generated_at" in txt) and ("git_commit" in txt)
        rows.append((f"指标 {Path(mp).name}", has,
                     "含 generated_at / git_commit" if has else "缺 generated_at 或 git_commit"))

    # 9 test 使用次数
    txt = blob(branch, spec["metrics"][0]).decode("utf-8", errors="replace") if spec["metrics"] else ""
    m = re.search(r'"test_evaluation_count"\s*:\s*(\d+)', txt)
    if m:
        n = int(m.group(1))
        rows.append(("test 使用次数", n <= 1, f"{n} 次" + ("" if n <= 1 else "，超过 1 次要说明")))

    verdict = "PASS" if all(ok for _, ok, _ in rows) else "FAIL"
    return rows, verdict


def main() -> int:
    ap = argparse.ArgumentParser(description="合并闸门检查")
    ap.add_argument("branch", nargs="?", default=None)
    ap.add_argument("--all", action="store_true", help="体检所有 feat/* 分支")
    ap.add_argument("--main", action="store_true", help="只查 main 上的周期特征配对")
    args = ap.parse_args()

    git("fetch", "--all", "--prune", "-q")

    if args.main:
        pair = feature_pair_check("main")
        print(f"main 周期特征配对：{'PASS' if pair and pair[0] else 'FAIL'}")
        if pair:
            print(f"  {pair[1]}")
        return 0 if pair and pair[0] else 1

    if args.all:
        branches = [l.split("refs/heads/")[-1]
                    for l in git("ls-remote", "--heads", "origin").splitlines()
                    if "refs/heads/feat/" in l]
        branches.sort()
    else:
        b = args.branch or git("rev-parse", "--abbrev-ref", "HEAD").strip()
        branches = [b]

    overall = "PASS"
    for br in branches:
        rows, verdict = audit(br)
        print(f"\n{'=' * 62}\n{br}   [{verdict}]\n{'=' * 62}")
        for name, ok, note in rows:
            print(f"  {'PASS' if ok else 'FAIL'}  {name:16s} {note}")
        if verdict == "FAIL":
            overall = "FAIL"

    print(f"\n总判定：{overall}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
