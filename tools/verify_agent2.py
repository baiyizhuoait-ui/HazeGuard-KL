"""Run the Agent 2 handover checks and record the actual environment and outcomes."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    commands = [
        [sys.executable, "src/01_verify_data.py", "--check"],
        [sys.executable, "-m", "pytest", "tests/test_agent2.py", "-q"],
        [sys.executable, "tools/check_prose.py", "reports/"],
        [sys.executable, "-m", "compileall", "-q", "src/02_eda_analysis.py",
         "src/02_spatiotemporal_features.py", "tests/test_agent2.py"],
        ["git", "diff", "--check", "HEAD"],
    ]
    results = []
    for command in commands:
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, encoding="utf-8",
                                errors="replace", capture_output=True)
        label = ["python", *command[1:]] if command[0] == sys.executable else command
        results.append({"command": label, "exit_code": result.returncode,
                        "stdout": result.stdout.strip(), "stderr": result.stderr.strip()})
        print(f"{'PASS' if result.returncode == 0 else 'FAIL'} {' '.join(label)}")
    sources = ["src/02_eda_analysis.py", "src/02_spatiotemporal_features.py", "tests/test_agent2.py"]
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "pyarrow", "matplotlib", "scipy", "pytest", "Pillow"]},
        "source_sha256": {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources},
        "checks": results, "all_exit_codes_zero": all(x["exit_code"] == 0 for x in results),
        "limitations": ["Runtime versions differ from requirements.txt; exact pinned environment not exercised.",
                         "Report style checker can emit a rhythm warning; its output is retained verbatim.",
                         "No downstream modelling, live-data availability or ground-observation validation performed."]}
    (ROOT / "models/metrics_stage2_verification.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if output["all_exit_codes_zero"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
