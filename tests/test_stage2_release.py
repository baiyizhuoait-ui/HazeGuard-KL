"""Release validation must reject substituted PNGs even with valid size/DPI."""
import hashlib
import importlib.util
from pathlib import Path

from PIL import Image
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_release", ROOT / "tools/verify_stage2.py")
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


def test_same_size_and_dpi_do_not_validate_a_substituted_figure(tmp_path):
    path = tmp_path / "generated.png"
    Image.new("RGB", (3000, 30), color="blue").save(path, dpi=(300, 300))
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert V.verified_figure(path, expected)["sha256"] == expected
    Image.new("RGB", (3000, 30), color="red").save(path, dpi=(300, 300))
    with pytest.raises(ValueError, match="hash mismatch"):
        V.verified_figure(path, expected)


def test_matching_hash_does_not_validate_wrong_resolution(tmp_path):
    path = tmp_path / "wrong_dpi.png"
    Image.new("RGB", (3000, 30)).save(path, dpi=(72, 72))
    with pytest.raises(AssertionError):
        V.verified_figure(path, hashlib.sha256(path.read_bytes()).hexdigest())



def test_ordered_parts_preserve_consumer_interface_and_integrity(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("release_storage", ROOT / "src/stage2_storage.py")
    storage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(storage)
    monkeypatch.setattr(storage, "ROWS_PER_PART", 2)
    path = tmp_path / "features.parquet"
    data = pd.DataFrame({
        "station_id": pd.Categorical(["B", "A", "B", "C", "A"]),
        "pm25_lag_1h": pd.Series([0.2, None, 0.4, 0.1, 0.3], dtype="float32"),
    })
    data.to_parquet(path, index=False)
    manifest = storage.write_feature_parts(data, path)
    pd.testing.assert_frame_equal(pd.read_parquet(path), data, check_exact=True)
    assert manifest == storage.feature_artifact(path)
    assert [part["rows"] for part in manifest["output_parts"]] == [2, 2, 1]
    assert manifest["output_bytes"] == sum(part.stat().st_size for part in path.iterdir())
    (path / "part-0001.parquet").unlink()
    with pytest.raises(ValueError, match="Invalid ordered"):
        storage.feature_artifact(path)
