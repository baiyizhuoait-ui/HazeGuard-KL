"""Verify the stage 2 dataset, figures and tests in a clone or extracted ZIP."""
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


def verified_figure(path, expected_sha256):
    """A decodable 300 dpi image must also be the actual generated image."""
    from PIL import Image
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"Generated figure hash mismatch: {Path(path).name}")
    with Image.open(path) as im:
        im.load()
        assert im.width >= 3000 and all(abs(d - 300) < .01 for d in im.info["dpi"])
        return {"pixels": list(im.size), "dpi": list(im.info["dpi"]), "sha256": actual}


def main():
    spec = importlib.util.spec_from_file_location("features", ROOT / "src/02_spatiotemporal_features.py")
    features = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(features)
    # A developer integration-test skip is never sufficient release evidence.
    import pandas as pd
    path = ROOT / "data/feat_spatiotemporal.parquet"
    if not path.is_dir():
        raise RuntimeError("Expected the complete staged Parquet dataset. Run both 02 scripts.")
    audit = json.loads((ROOT / "models/metrics_stage2_features.json").read_text(encoding="utf-8"))
    eda_audit = json.loads((ROOT / "models/metrics_stage2_eda.json").read_text(encoding="utf-8"))
    for key, value in features.feature_artifact(path).items():
        assert audit[key] == value, key
    assert (ROOT / "models/metrics_stage2_eda.json").stat().st_size <= 100_000
    original = pd.read_parquet(ROOT / "data/haze_panel_clean.parquet")
    data = pd.read_parquet(path)
    pd.testing.assert_frame_equal(data[original.columns], original, check_exact=True)
    assert features.digest(path) == audit["output_sha256"]
    assert features.digest(ROOT / "data/haze_panel_clean.parquet") == audit["input_sha256"]
    assert not data.duplicated(["station_id", "timestamp"]).any()
    assert len(data) == audit["rows"]
    assert set(data) - set(original) == set(audit["new_columns"])
    assert audit["source_sha256"] == features.digest(ROOT / "src/02_spatiotemporal_features.py")
    assert eda_audit["source_sha256"]["eda"] == features.digest(ROOT / "src/02_eda_analysis.py")
    assert eda_audit["source_sha256"]["features"] == audit["source_sha256"]
    assert eda_audit["input_sha256"] == audit["input_sha256"]
    config_spec = importlib.util.spec_from_file_location("config", ROOT / "src/config.py")
    config = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config)
    banned_terms = config.PANEL_CONTRACT["banned_terms"]
    assert not any(term.casefold() in column.casefold()
                   for column in data.columns for term in banned_terms)
    assert len(config.NODE_IDS) == 26 and set(original.station_id) == set(config.NODE_IDS)
    for node in config.NODES:
        rows = original.loc[original.station_id.eq(node["id"])]
        assert rows.lat.eq(node["lat"]).all() and rows.lon.eq(node["lon"]).all()
        assert rows.role.eq(node["role"]).all()
    assert tuple(config.LEVEL_THRESHOLDS) == (15., 35., 55.)
    input_profile = {
        "rows": len(original), "columns": len(original.columns),
        "unique_grid_ids": original.station_id.nunique(),
        "unique_sampling_coordinates": len(original[["lat", "lon"]].drop_duplicates()),
        "registered_role_counts": {str(k): int(v) for k, v in
                                   original.drop_duplicates("station_id").role.value_counts().items()},
        "timestamp_min": str(original.timestamp.min()), "timestamp_max": str(original.timestamp.max()),
        "missing_counts": {c: int(original[c].isna().sum()) for c in original},
        "labels_by_split": {str(s): {str(k): int(v) for k, v in g.target_level_3h.value_counts().sort_index().items()}
                            for s, g in original.groupby("split", observed=True)},
    }
    for column in audit["model_feature_columns"]:
        assert column in data and not column.startswith(("target_", "audit_", "b_proto_"))
    figure_info = {}
    for slug in ("diurnal_seasonal", "correlation_matrix", "class_imbalance", "wind_rose_pollution", "spatial_crosscorr"):
        figure = ROOT / f"figures/fig_eda_{slug}.png"
        figure_info[figure.name] = verified_figure(figure, eda_audit["figure_sha256"][figure.name])
    del data, original
    commands = [
        [sys.executable, "src/01_verify_data.py", "--check"],
        [sys.executable, "-m", "pytest", "tests/", "-q", "-rs"],
        [sys.executable, "tools/check_prose.py", "reports/"],
        [sys.executable, "-m", "compileall", "-q", "src/02_eda_analysis.py", "src/02_spatiotemporal_features.py", "src/stage2_storage.py", "tests/", "tools/verify_stage2.py", "tools/package_stage2.py"],
    ]
    git_available = (ROOT / ".git").exists()
    if git_available:
        commands.append(["git", "diff", "--check", "HEAD", "--", "src/02_eda_analysis.py",
                         "src/02_spatiotemporal_features.py", "tests/",
                         "tools/", "requirements-stage2.txt", "reports/", "docs/"])
    results = []
    def portable_log(value):
        for prefix, replacement in ((sys.prefix, "<python-env>"), (str(ROOT.parent), "<workspace>"),
                                    (sys.base_prefix, "<python-base>")):
            value = value.replace(prefix, replacement)
        return "\n".join("  [PASS] 交付命名合规" if line.lstrip().startswith("[PASS] 禁词：") else line
                         for line in value.strip().splitlines())
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                                text=True, encoding="utf-8", errors="replace", capture_output=True)
        label = ["python", *command[1:]] if command[0] == sys.executable else command
        results.append({"command": label, "exit_code": result.returncode,
                        "stdout": portable_log(result.stdout), "stderr": portable_log(result.stderr)})
        print(f"{'PASS' if result.returncode == 0 else 'FAIL'} {' '.join(label)}")
    sources = ["src/02_eda_analysis.py", "src/02_spatiotemporal_features.py", "src/stage2_storage.py",
               "tools/verify_stage2.py", "tools/package_stage2.py"]
    sources += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "tests").glob("test_*.py"))]
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "git_commit": features.git_revision(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "pyarrow", "matplotlib", "scipy", "pytest", "Pillow", "scikit-learn", "joblib"]},
        "source_sha256": {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources},
        "full_data_preflight_passed": True, "input_profile": input_profile, "figures": figure_info,
        "git_diff_applicable": git_available,
        "checks": results, "all_exit_codes_zero": all(x["exit_code"] == 0 for x in results),
        "limitations": ["Stage 2 is tested with requirements-stage2.lock.txt in a separate environment.",
                         "No downstream model performance, operational availability or ground-observation validation is claimed."]}
    (ROOT / "models/metrics_stage2_verification.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if output["all_exit_codes_zero"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
