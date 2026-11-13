"""报告数字溯源：报告里写的每个指标值，能不能在 models/*.json 里查到。

背景：这个项目出过一次事故——清洗流程改了之后报告没同步，差 89 行，
被外部评审抓出来。人工核对不可靠，所以做成脚本。

做法：把所有 json 里的数值摊平成一个大集合（按 3 位和 4 位小数两种写法都登记），
再扫 reports/*.md 里的小数，凡是查不到的列出来，交给人复核。

注意：**这不是门禁**。报告里有大量派生数字（差值、百分比、行数换算），
它们本来就不在 json 里，所以脚本只输出"待复核清单"，不判定通过与否。
真正要看的是：清单里有没有本该能查到、却查不到的数。

用法：
    python tools/check_report_numbers.py
    python tools/check_report_numbers.py reports/sec2_6_models.md
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSON_DIRS = [ROOT / "models", ROOT / "data"]
REPORT_DIR = ROOT / "reports"

NUM = re.compile(r"(?<![\w.])-?\d+\.\d{2,4}(?![\w])")


def collect_json_numbers() -> dict[str, str]:
    """把 json 与 csv 里的数值按 3/4 位小数登记，值 → 来源文件。"""
    facts: dict[str, str] = {}

    # CSV 先登记（LOSO、分季度这类结果只存在 csv 里）
    for d in JSON_DIRS:
        for p in sorted(d.glob("*.csv")):
            try:
                rows = list(csv.DictReader(p.read_text(encoding="utf-8").splitlines()))
            except (UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                for v in row.values():
                    try:
                        f = float(v)
                    except (TypeError, ValueError):
                        continue
                    for nd in (3, 4):
                        facts[f"{f:.{nd}f}"] = p.name

    for d in JSON_DIRS:
        for p in sorted(d.glob("*.json")):
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            def walk(o):
                if isinstance(o, dict):
                    for v in o.values():
                        walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)
                elif isinstance(o, float) or (isinstance(o, int) and not isinstance(o, bool)):
                    # 比值类指标都在 0~1，只按 3/4 位登记，避免 0.66 误配 0.6620；
                    # 其余（百分比、时延、浓度变化）报告常写两位小数，额外登记 2 位
                    nds = (3, 4) if abs(o) < 1 else (2, 3, 4)
                    for nd in nds:
                        facts[f"{o:.{nd}f}"] = p.name

            walk(payload)
    return facts


def derived(a: str, b: str) -> bool:
    """两数之差是否也常出现在报告里（差值本身查不到属正常，这里只做提示）。"""
    return abs(float(a) - float(b)) < 1e-9


def scan(report: Path, facts: dict[str, str]) -> list[tuple[str, str]]:
    text = report.read_text(encoding="utf-8")
    body = "\n".join(ln for ln in text.split("\n")
                     if not ln.strip().startswith("```"))
    misses: list[tuple[str, str]] = []
    for m in NUM.finditer(body):
        raw = m.group(0)
        val = float(raw)
        # 百分数：报告写 93.60%，json 里存的是 0.9360
        tail = body[m.end():m.end() + 2].lstrip()
        is_pct = tail.startswith("%")
        candidates = [val, val / 100] if is_pct else [val]
        # 与登记侧一致：0~1 的比值只比 3/4 位，其余允许两位
        nds = (3, 4) if abs(val) < 1 else (2, 3, 4)
        if any(f"{c:.{nd}f}" in facts for c in candidates for nd in nds):
            continue
        else:
            line = body[:m.start()].count("\n") + 1
            ctx = body.split("\n")[line - 1].strip()[:80]
            misses.append((raw, f"第 {line} 行：{ctx}"))
    return misses


def main() -> int:
    args = sys.argv[1:]
    reports = [Path(a) for a in args] if args else sorted(REPORT_DIR.glob("*.md"))
    facts = collect_json_numbers()
    print(f"json 事实表：{len(facts)} 个数值（models/ 与 data/ 下的 *.json）\n")

    total = 0
    for rep in reports:
        misses = scan(rep, facts)
        total += len(misses)
        print(f"{rep.name}：{len(misses)} 个待复核")
        for val, ctx in misses:
            print(f"    {val:>8s}   {ctx}")
        print()

    print(f"合计 {total} 个数字未在 json 中直接命中。")
    print("其中差值、百分比、行数换算属正常；**若某个指标值查不到，那只可能是报告写错了**。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
