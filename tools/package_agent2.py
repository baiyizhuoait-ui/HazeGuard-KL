"""Export a verified commit with full LFS data; extract and rerun the release.

Run from the repository root with the tested Agent 2 Python environment.
No Git history, branches, or source data are changed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--feature-data", type=Path, default=ROOT / "data/feat_spatiotemporal.parquet")
    ap.add_argument("--verify-python", type=Path, default=Path(sys.executable))
    args = ap.parse_args()
    package, interpreter = args.output.resolve(), args.verify_python.resolve()
    if package.exists():
        ap.error("output already exists; choose a new release path")
    package.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", args.commit], cwd=ROOT, text=True).strip()
    feature_data = args.feature_data.resolve()
    with tempfile.TemporaryDirectory(prefix="agent2-release-", dir=package.parent) as scratch:
        scratch = Path(scratch)
        archive, stage = scratch / "source.zip", scratch / "stage"
        subprocess.run(["git", "-c", "filter.lfs.process=", "-c", "filter.lfs.smudge=", "-c", "filter.lfs.required=false",
                        "archive", "--format=zip", f"--output={archive}", commit], cwd=ROOT, check=True)
        stage.mkdir()
        with ZipFile(archive) as z:
            assert z.testzip() is None
            z.extractall(stage)
        audit = json.loads((stage / "models/metrics_stage2_features.json").read_text(encoding="utf-8"))
        pointer = (stage / "data/feat_spatiotemporal.parquet").read_bytes()
        assert len(pointer) < 1024 and audit["output_sha256"].encode() in pointer, "commit LFS pointer/audit mismatch"
        assert digest(feature_data) == audit["output_sha256"], "full data does not belong to this commit"
        assert feature_data.stat().st_size == audit["output_bytes"]
        shutil.copyfile(feature_data, stage / "data/feat_spatiotemporal.parquet")
        (stage / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="ascii")
        (stage / "DELIVERY.md").write_text(
            "# HazeGuard KL Agent 2 verified 26 grid release\n\n"
            f"Source commit: {commit}\n\n"
            "This is an export of that commit with the full, checksum-verified LFS object. "
            "No Git directory or API download is required for Agent 2. See docs/technical_spec_26grid.md, "
            "docs/repository_audit_agent2_26grid.md and docs/handover/02_eda_and_spatiotemporal.md.\n\n"
            "Create a Python 3.13 environment, activate it, then run:\n\n```sh\n"
            "python -m pip install -r requirements-agent2.lock.txt\n"
            "python tools/verify_agent2.py\npython src/02_spatiotemporal_features.py\n"
            "python src/02_eda_analysis.py\npython tools/verify_agent2.py\n```\n\n"
            "The adjacent verification JSON records actual extraction and regeneration of this exact ZIP. "
            "The repository's generation JSON retains its original generation commit and source hashes. "
            "SOURCE_COMMIT.txt identifies the exported commit. Audit-only future-label flags are never predictors.\n",
            encoding="utf-8")
        files = {p.relative_to(stage).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
                 for p in sorted(stage.rglob("*")) if p.is_file()}
        manifest = {"commit": commit, "repository": "baiyizhuoait/HazeGuard-KL", "files": files}
        (stage / "DELIVERY_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        with ZipFile(package, "w", ZIP_DEFLATED, compresslevel=6) as z:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    z.write(path, "HazeGuard-KL/" + path.relative_to(stage).as_posix())
        extracted = scratch / "extracted"
        with ZipFile(package) as z:
            assert z.testzip() is None
            z.extractall(extracted)
        release = extracted / "HazeGuard-KL"
        for name, item in files.items():
            assert digest(release / name) == item["sha256"], name
        assert not (release / ".git").exists()
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        results = []
        for command in (["-m", "pip", "check"], ["tools/verify_agent2.py"],
                        ["src/02_spatiotemporal_features.py"], ["src/02_eda_analysis.py"], ["tools/verify_agent2.py"]):
            started = time.monotonic()
            result = subprocess.run([str(interpreter), *command], cwd=release, env=env, text=True,
                                    encoding="utf-8", errors="replace", capture_output=True, timeout=600)
            portable = lambda s: s.replace(str(interpreter.parent.parent), "<python-env>").replace(str(scratch), "<release>").strip()
            results.append({"command": ["python", *command], "exit_code": result.returncode,
                            "seconds": round(time.monotonic()-started, 2),
                            "stdout": portable(result.stdout), "stderr": portable(result.stderr)})
            print(f"{'PASS' if result.returncode == 0 else 'FAIL'} {' '.join(command)}", flush=True)
            if result.returncode:
                raise RuntimeError(results[-1])
        verified = json.loads((release / "models/metrics_stage2_verification.json").read_text(encoding="utf-8"))
        assert verified["all_exit_codes_zero"] and verified["git_commit"] == commit
        rebuilt = {name: digest(release / name) == item["sha256"] for name, item in files.items()
                   if name == "data/feat_spatiotemporal.parquet" or name.startswith("figures/fig_eda_")}
        # This release is verified in its locked build environment. A differing
        # platform build needs its own reviewed artifacts and cannot pass silently.
        record = {"generated_at": datetime.now(timezone.utc).isoformat(), "commit": commit,
                  "zip": package.name, "zip_bytes": package.stat().st_size, "zip_sha256": digest(package),
                  "zip_files": len(files)+1, "all_manifest_files_verified": True,
                  "no_git_directory": True, "commands": results, "regenerated_byte_identical": rebuilt,
                  "verification": verified, "all_checks_passed": all(rebuilt.values())}
        package.with_suffix(".verification.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        package.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        package.with_suffix(".zip.sha256").write_text(digest(package) + "  " + package.name + "\n", encoding="ascii")
        if not record["all_checks_passed"]:
            raise RuntimeError(f"Regenerated artifacts differ from this commit; release verification failed: {rebuilt}")
        print(json.dumps({k: record[k] for k in ("commit", "zip_files", "zip_bytes", "zip_sha256", "regenerated_byte_identical")}, indent=2))


if __name__ == "__main__":
    main()
