#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build the Moodle submission package (module 07). Run from the repository root:

    python src/07_package_submission.py --group G07 --report path/to/G07_Report.pdf
    python src/07_package_submission.py --group G07 --dry-run     # list and size only

Course rules this script enforces (docs/spec_v3.md section 6.3):
  * the report is named <group>_Report.pdf;
  * a plain-text README.txt is included (generated from README.md);
  * the whole package stays under the 20 MB upload limit.

Large binaries (data/*.parquet, models/*.joblib, models/*.npz) are left out on
purpose; put their download link on the report cover and in README.md.
The package is written next to the repository, not inside it, so it cannot be
committed by accident.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT_MB = 20.0
PATTERNS = [
    "app.py", "requirements.txt", ".streamlit/config.toml",
    "src/*.py", "tools/*.py", "tests/*.py",
    "models/*.json", "models/*.csv",
    "figures/*.png",
    "data/data_dictionary.json", "data/manifest.json",
]


def plain_text(markdown: str) -> str:
    """README.md to README.txt: drop the Markdown marks, keep the words and tables."""
    lines, in_code = [], False
    for line in markdown.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            lines.append("    " + line)
            continue
        if re.fullmatch(r"\s*\|?[\s:|-]+\|[\s:|-]*", line):      # table separator row
            continue
        if re.fullmatch(r"\s*(-{3,}|\*{3,})\s*", line):          # horizontal rule
            lines.append("")
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            title = heading.group(2).strip()
            mark = "=" if len(heading.group(1)) <= 2 else "-"
            width = sum(2 if ord(ch) > 0x2E80 else 1 for ch in title)   # CJK is double width
            lines.extend(["", title, mark * max(width, 4)])
            continue
        line = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"[figure: \1]", line)
        line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", line)
        line = re.sub(r"(\*\*|__)(.+?)\1", r"\2", line)
        line = re.sub(r"`([^`]*)`", r"\1", line)
        line = re.sub(r"^\s*>\s?", "", line)
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            line = "  ".join(cells)
        lines.append(line.rstrip())
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def collect() -> list[Path]:
    files: list[Path] = []
    for pattern in PATTERNS:
        files.extend(sorted(p for p in ROOT.glob(pattern) if p.is_file()))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Moodle submission package")
    parser.add_argument("--group", required=True, help="group number, e.g. G07")
    parser.add_argument("--report", type=Path, help="the exported report PDF")
    parser.add_argument("--out", type=Path, default=ROOT.parent / "HazeGuard-KL_submission")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    problems = []
    files = collect()
    readme = plain_text((ROOT / "README.md").read_text(encoding="utf-8"))
    report_name = f"{args.group}_Report.pdf"
    if args.report is None:
        problems.append(f"no --report given; {report_name} is missing from the package")
    elif not args.report.exists():
        problems.append(f"report not found: {args.report}")
    elif args.report.suffix.lower() != ".pdf":
        problems.append("the report must be a PDF")

    raw = sum(p.stat().st_size for p in files) + len(readme.encode("utf-8"))
    if args.report and args.report.exists():
        raw += args.report.stat().st_size
    by_folder: dict[str, int] = {}
    for p in files:
        top = p.relative_to(ROOT).parts[0]
        by_folder[top] = by_folder.get(top, 0) + p.stat().st_size
    print(f"{len(files)} files, {raw / 1048576:.1f} MB before compression")
    for name, size in sorted(by_folder.items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s} {size / 1048576:6.2f} MB")

    if args.dry_run:
        for problem in problems:
            print(f"PROBLEM  {problem}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{args.group}_HazeGuard-KL.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        bundle.writestr("README.txt", readme)
        if args.report and args.report.exists() and args.report.suffix.lower() == ".pdf":
            bundle.write(args.report, report_name)
        for p in files:
            bundle.write(p, p.relative_to(ROOT).as_posix())
    size_mb = target.stat().st_size / 1048576
    if size_mb > LIMIT_MB:
        problems.append(f"package is {size_mb:.1f} MB, over the {LIMIT_MB:g} MB limit")
    for problem in problems:
        print(f"PROBLEM  {problem}")
    print(f"Wrote {target} ({size_mb:.1f} MB of {LIMIT_MB:g} MB allowed)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
