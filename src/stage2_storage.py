"""Ordered, checksum-verified Parquet parts for the stage 2 feature table."""
import hashlib
import json
from pathlib import Path
import tempfile

import pandas as pd
from pyarrow.parquet import ParquetFile

MAX_PART_BYTES = 50_000_000
ROWS_PER_PART = 250_000


def _file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def feature_artifact(path):
    """Return total bytes and a digest of the ordered part manifest.

    A historic single Parquet file remains readable for comparisons.
    """
    path = Path(path)
    if path.is_file():
        ParquetFile(path)  # A pointer or other non-Parquet file is an error.
        return {"output_sha256": _file_hash(path), "output_bytes": path.stat().st_size,
                "output_parts": []}
    parts = sorted(path.iterdir())
    expected = [f"part-{i:04d}.parquet" for i in range(len(parts))]
    if not parts or [part.name for part in parts] != expected:
        raise ValueError(f"Invalid ordered Parquet parts at {path}")
    manifest = []
    for part in parts:
        size = part.stat().st_size
        if not part.is_file() or size > MAX_PART_BYTES:
            raise ValueError(f"Parquet part must be a file no larger than 50 MB: {part}")
        manifest.append({"name": part.name, "bytes": size, "sha256": _file_hash(part),
                         "rows": ParquetFile(part).metadata.num_rows})
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("ascii")
    return {"output_sha256": hashlib.sha256(encoded).hexdigest(),
            "output_bytes": sum(p["bytes"] for p in manifest), "output_parts": manifest}


def digest(path):
    path = Path(path)
    return feature_artifact(path)["output_sha256"] if path.is_dir() else _file_hash(path)


def write_feature_parts(frame, path):
    """Stage and validate all parts before replacing the existing dataset."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(f".{path.name}.write.lock")
    # Exclusive creation prevents two writers from replacing each other's output.
    with lock.open("x"):
        pass
    try:
        with tempfile.TemporaryDirectory(prefix=".stage2-write-", dir=path.parent) as directory:
            work = Path(directory).resolve()
            if work.parent != path.parent:
                raise ValueError("Staging must remain in the output directory")
            staged = work / "new"
            staged.mkdir()
            for i, first in enumerate(range(0, len(frame), ROWS_PER_PART)):
                frame.iloc[first:first + ROWS_PER_PART].to_parquet(
                    staged / f"part-{i:04d}.parquet", index=False, compression="zstd",
                    compression_level=15, row_group_size=40_000)
            metadata = feature_artifact(staged)
            pd.testing.assert_frame_equal(pd.read_parquet(staged), frame.reset_index(drop=True), check_exact=True)
            previous = work / "previous"
            if path.exists():
                path.replace(previous)
            try:
                staged.replace(path)
            except OSError:
                if previous.exists():
                    previous.replace(path)
                raise
        return metadata
    finally:
        lock.unlink()
