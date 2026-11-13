#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Assemble the members' report sections into one Markdown file (module 07).

    python src/07_compile_report.py --group G07 --members "Name A, Name B"
    python src/07_compile_report.py --check          # preflight only, writes nothing

The script only stitches files together. It does not write or reword any report
text: each section stays exactly as its owner committed it. Section order follows
the ownership table in CONTRIBUTING.md.

Preflight problems are printed before anything is written. With --strict the
script exits with status 1 when a problem is found.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "HazeGuard_KL_Demonstration_Report.md"

# (file, sections it is expected to carry, owner)
SECTIONS = [
    ("sec1_data_and_baseline.md", "1, 2.1, 2.2", "member 1"),
    ("sec2_3_and_2_4_1.md", "2.3, 2.4.1", "member 2"),
    ("sec2_4_2_and_2_5.md", "2.4.2, 2.5", "member 3"),
    ("sec2_6_models.md", "2.6", "member 4"),
    ("sec2_6_and_2_7.md", "2.7, 2.8", "member 5"),
    ("sec3_cost_and_explain.md", "3", "member 6"),
    ("sec4_system_and_demo.md", "4", "member 7"),
]

HEADING = re.compile(r"^(#{1,3})\s+(.*\S)\s*$")
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
CJK = re.compile(r"[一-鿿]")
DRAFT = re.compile(r"handover draft|not the final|不是直接提交|提纲|TODO|TBD", re.IGNORECASE)


def slug(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"\s+", "-", text.strip())


def preflight(name: str, text: str) -> list[str]:
    """Things the owner of the section has to fix; this script never fixes them."""
    problems = []
    letters = [c for c in text if not c.isspace()]
    if letters and len(CJK.findall(text)) / len(letters) > 0.05:
        problems.append("is mostly not in English")
    hit = DRAFT.search(text)
    if hit:
        problems.append(f"still carries a draft marker ({hit.group(0)!r})")
    words = len(re.findall(r"[A-Za-z]{2,}", text))
    if words < 400:
        problems.append(f"has only {words} English words")
    for target in IMAGE.findall(text):
        if target.startswith(("http://", "https://")):
            continue
        if not (REPORTS / target).resolve().exists():
            problems.append(f"links a figure that does not exist: {target}")
    return [f"{name} {p}" for p in problems]


def collect() -> tuple[list[tuple[str, str, str, str | None]], list[str]]:
    parts, problems = [], []
    for name, covers, owner in SECTIONS:
        path = REPORTS / name
        if not path.exists():
            parts.append((name, covers, owner, None))
            problems.append(f"{name} is missing (sections {covers}, {owner})")
            continue
        text = path.read_text(encoding="utf-8").strip()
        parts.append((name, covers, owner, text))
        problems.extend(preflight(name, text))
    return parts, problems


def table_of_contents(body: str) -> str:
    lines, in_code = [], False
    for line in body.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        match = None if in_code else HEADING.match(line)
        if match and len(match.group(1)) <= 2:
            indent = "  " * (len(match.group(1)) - 1)
            title = match.group(2)
            lines.append(f"{indent}- [{title}](#{slug(title)})")
    return "\n".join(lines)


def build(group: str, members: str, parts) -> str:
    chunks = []
    for name, covers, owner, text in parts:
        if text is None:
            chunks.append(f"# [Sections {covers} not delivered: reports/{name}, {owner}]")
        else:
            chunks.append(f"<!-- source: reports/{name} ({owner}) -->\n\n{text}")
    body = "\n\n".join(chunks)
    cover = "\n".join([
        "# HazeGuard-KL",
        "",
        "AIT201 Applied Machine Learning, 2026/09",
        "",
        f"Group: {group}",
        "",
        f"Members: {members}",
        "",
        "Repository: https://github.com/baiyizhuoait/HazeGuard-KL",
        "",
        "<!-- Add the data and model download link here before export. -->",
    ])
    return "\n\n".join([cover, "## Table of Contents", table_of_contents(body), body]) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--group", default="<group number>")
    parser.add_argument("--members", default="<member names>")
    parser.add_argument("--check", action="store_true", help="preflight only")
    parser.add_argument("--strict", action="store_true", help="exit 1 on any problem")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()

    parts, problems = collect()
    for problem in problems:
        print(f"PROBLEM  {problem}")
    print(f"{len(problems)} problem(s) across {len(parts)} section files.")
    if args.check:
        return 1 if (problems and args.strict) else 0
    if problems and args.strict:
        print("Nothing written (--strict).")
        return 1

    report = build(args.group, args.members, parts)
    args.out.write_text(report, encoding="utf-8")
    words = len(re.findall(r"[A-Za-z]{2,}", report))
    print(f"Wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out} "
          f"({words:,} English words).")
    print("Next: python tools/check_prose.py reports/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
