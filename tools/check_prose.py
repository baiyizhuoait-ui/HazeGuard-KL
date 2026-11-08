#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
报告文字自查（交稿前必跑）

为什么有这个脚本：课程 PDF 第 2 页 C.2 明确规定——作业必须是学生自己的文字，
使用 AI 工具生成的文本（含直接粘贴与改写）会导致该项作业 0 分。而 AI 写的英文
有几个很稳定的特征，自己看不出来，机器一数就很明显。

本脚本查四类问题：
  1. 破折号（em-dash）密度过高：正式学术文体里几乎不用，是一眼可辨的机器痕迹
  2. 套话/框架词：plays a crucial role、it is worth noting、delve into 之类
  3. 对偶句式：not only ... but also、It is not X but Y（模型最爱的修辞骨架）
  4. 句子节奏过于均匀：长度变异系数过低，读起来像模板填充

用法
----
python tools/check_prose.py reports/sec1_data_and_baseline.md
python tools/check_prose.py reports/            # 整个目录
python tools/check_prose.py reports/ --strict   # 有任一项不过即返回 1

退出码：0 通过（或只有提示）/ 1 有硬性问题（--strict 下含警告）/ 2 用法错误
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import sys

# --- 阈值（按英文理工科实验报告校准，不是绝对的，够用即可）------------------
EM_DASH_PER_1K_MAX = 5.0      # 每千词破折号数
MEAN_SENT_LEN_MAX = 32.0      # 平均句长（词）
SENT_CV_MIN = 0.35            # 句长变异系数下限

BANNED = [
    # 框架修饰族
    "crucial", "pivotal", "paramount", "notably", "importantly",
    "plays a crucial role", "plays a vital role", "plays a key role",
    # 八股开场
    "it is worth noting", "it should be noted", "it is important to note",
    "in the realm of", "in today's world", "in recent years, with the rapid",
    # 模型高频动词
    "delve into", "leverage", "utilize", "harness", "navigate the",
    # 空泛形容
    "comprehensive", "holistic", "seamless", "robust solution", "cutting-edge",
    "game-changer", "state-of-the-art solution",
    # 收尾套路
    "in conclusion,", "to sum up,", "all in all,",
]

# 对偶句式：模型最爱的骨架，学生写法里很少连着出现
ANTITHESIS = [
    r"\bnot only\b[^.]{0,80}\bbut also\b",
    r"\bit is not\b[^.]{0,60}\bbut\b",
    r"\bnot\s+\w+,\s+but\s+\w+",
    r"\bis not (?:a|an|the)\b[^.]{0,50}\.\s+It is\b",
]


def words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9µ][A-Za-z0-9µ'\-\.]*", text))


def sentences(text: str) -> list[str]:
    # 去掉 markdown 表格行与代码块，只看正文
    body = "\n".join(
        ln for ln in text.split("\n")
        if not ln.strip().startswith("|") and not ln.strip().startswith("```")
        and not ln.strip().startswith("#")
    )
    parts = re.split(r"(?<=[.!?])\s+", body)
    return [p.strip() for p in parts if len(p.split()) >= 4]


def check_file(path: str) -> dict:
    text = open(path, encoding="utf-8").read()
    body = "\n".join(
        ln for ln in text.split("\n")
        if not ln.strip().startswith("|") and not ln.strip().startswith("```")
    )
    nw = max(1, words(body))
    sents = sentences(text)
    lens = [len(s.split()) for s in sents] or [0]

    em = body.count("—")
    em_density = em / nw * 1000
    mean_len = statistics.mean(lens)
    cv = (statistics.pstdev(lens) / mean_len) if mean_len else 0.0

    low = text.lower()
    hits = [b for b in BANNED if b in low]

    anti = []
    for pat in ANTITHESIS:
        for m in re.finditer(pat, body, flags=re.IGNORECASE):
            snippet = body[max(0, m.start() - 20):m.end() + 20].replace("\n", " ")
            anti.append(snippet.strip())

    fails, warns = [], []
    if em_density > EM_DASH_PER_1K_MAX:
        fails.append(f"破折号密度 {em_density:.1f}/千词（上限 {EM_DASH_PER_1K_MAX}），共 {em} 处")
    if mean_len > MEAN_SENT_LEN_MAX:
        warns.append(f"平均句长 {mean_len:.1f} 词（建议 ≤ {MEAN_SENT_LEN_MAX}）")
    if cv < SENT_CV_MIN:
        warns.append(f"句长变异系数 {cv:.2f}（建议 ≥ {SENT_CV_MIN}，太均匀像模板）")
    if hits:
        fails.append(f"套话/框架词命中 {len(hits)} 个：" + "、".join(hits[:8]))
    if len(anti) >= 3:
        warns.append(f"对偶句式出现 {len(anti)} 次，读一下是否刻意")

    return {
        "file": path, "words": nw, "sentences": len(sents),
        "em_dash": em, "em_density": round(em_density, 2),
        "mean_len": round(mean_len, 1), "cv": round(cv, 2),
        "banned": hits, "antithesis": anti[:5],
        "fails": fails, "warns": warns,
    }


def report(r: dict) -> tuple[bool, bool]:
    ok = not r["fails"]
    clean = ok and not r["warns"]
    mark = "PASS" if clean else ("WARN" if ok else "FAIL")
    print(f"[{mark}] {os.path.relpath(r['file'])}")
    print(f"       {r['words']} 词 / {r['sentences']} 句 | "
          f"破折号 {r['em_dash']} 处 ({r['em_density']}/千词) | "
          f"平均句长 {r['mean_len']} | 变异系数 {r['cv']}")
    for f in r["fails"]:
        print(f"       ✗ {f}")
    for w in r["warns"]:
        print(f"       ! {w}")
    for a in r["antithesis"]:
        print(f"       · 对偶示例: {a[:90]}")
    return ok, clean


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="单个 .md 文件或目录")
    ap.add_argument("--strict", action="store_true", help="警告也算不通过")
    args = ap.parse_args()

    if os.path.isdir(args.target):
        files = sorted(
            os.path.join(args.target, f) for f in os.listdir(args.target)
            if f.endswith(".md")
        )
    elif os.path.isfile(args.target):
        files = [args.target]
    else:
        print(f"找不到 {args.target}")
        return 2

    if not files:
        print("目录里没有 .md 文件")
        return 2

    results = [check_file(f) for f in files]
    all_ok, all_clean = True, True
    for r in results:
        ok, clean = report(r)
        all_ok &= ok
        all_clean &= clean

    print()
    bad = [r for r in results if r["fails"]]
    print(f"共 {len(results)} 个文件，{len(bad)} 个有硬性问题。")
    if bad:
        print("改法：破折号换逗号/括号/句号；套话直接删；对偶句拆成两句陈述。")
    if args.strict:
        return 0 if all_clean else 1
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
