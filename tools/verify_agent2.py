"""Verify actual Agent 2 data, figures and tests in a clone or an extracted ZIP."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    spec = importlib.util.spec_from_file_location("features", ROOT / "src/02_spatiotemporal_features.py")
    features = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(features)
    # A developer integration-test skip is never sufficient release evidence.
    import pandas as pd
    from PIL import Image
    path = ROOT / "data/feat_spatiotemporal.parquet"
    if not path.is_file() or path.stat().st_size < 1024:
        raise RuntimeError("Full feature parquet required; LFS pointers are not release data. Run both 02 scripts.")
    audit = json.loads((ROOT / "models/metrics_stage2_features.json").read_text(encoding="utf-8"))
    original = pd.read_parquet(ROOT / "data/haze_panel_clean.parquet")
    data = pd.read_parquet(path)
    pd.testing.assert_frame_equal(data[original.columns], original)
    assert features.digest(path) == audit["output_sha256"]
    assert features.digest(ROOT / "data/haze_panel_clean.parquet") == audit["input_sha256"]
    assert not data.duplicated(["station_id", "timestamp"]).any()
    assert len(data) == audit["rows"]
    assert set(data) - set(original) == set(audit["new_columns"])
    for column in audit["model_feature_columns"]:
        assert column in data and not column.startswith(("target_", "agent2_", "b_proto_"))
    figure_info = {}
    for slug in ("diurnal_seasonal", "correlation_matrix", "class_imbalance", "wind_rose_pollution", "spatial_crosscorr"):
        figure = ROOT / f"figures/fig_eda_{slug}.png"
        with Image.open(figure) as im:
            im.load()
            assert im.width >= 3000 and all(abs(d - 300) < .01 for d in im.info["dpi"])
            figure_info[figure.name] = {"pixels": list(im.size), "dpi": list(im.info["dpi"]), "sha256": features.digest(figure)}
    del data, original
    commands = [
        [sys.executable, "src/01_verify_data.py", "--check"],
        [sys.executable, "-m", "pytest", "tests/test_agent2.py", "-q", "-rs"],
        [sys.executable, "tools/check_prose.py", "reports/"],
        [sys.executable, "-m", "compileall", "-q", "src/02_eda_analysis.py", "src/02_spatiotemporal_features.py", "tests/test_agent2.py"],
    ]
    git_available = (ROOT / ".git").exists()
    if git_available:
        commands.append(["git", "diff", "--check", "HEAD", "--", "src/02_eda_analysis.py",
                         "src/02_spatiotemporal_features.py", "tests/test_agent2.py",
                         "tools/verify_agent2.py", "README.md", "requirements-agent2.txt", "reports/", "docs/handover/"])
    results = []
    def portable_log(value):
        for prefix, replacement in ((sys.prefix, "<python-env>"), (str(ROOT.parent), "<workspace>"),
                                    (sys.base_prefix, "<python-base>")):
            value = value.replace(prefix, replacement)
        return value.strip()
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                                text=True, encoding="utf-8", errors="replace", capture_output=True)
        label = ["python", *command[1:]] if command[0] == sys.executable else command
        results.append({"command": label, "exit_code": result.returncode,
                        "stdout": portable_log(result.stdout), "stderr": portable_log(result.stderr)})
        print(f"{'PASS' if result.returncode == 0 else 'FAIL'} {' '.join(label)}")
    sources = ["src/02_eda_analysis.py", "src/02_spatiotemporal_features.py", "tests/test_agent2.py", "tools/verify_agent2.py"]
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "git_commit": features.git_revision(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "pyarrow", "matplotlib", "scipy", "pytest", "Pillow", "scikit-learn", "joblib"]},
        "source_sha256": {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources},
        "full_data_preflight_passed": True, "figures": figure_info,
        "git_diff_applicable": git_available,
        "checks": results, "all_exit_codes_zero": all(x["exit_code"] == 0 for x in results),
        "limitations": ["Agent 2 is tested with requirements-agent2.txt; the upstream full-project pins are a separate environment.",
                         "No downstream model performance, operational availability or ground-observation validation is claimed."]}
    (ROOT / "models/metrics_stage2_verification.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if output["all_exit_codes_zero"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
