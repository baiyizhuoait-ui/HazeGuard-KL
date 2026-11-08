"""Release validation must reject substituted PNGs even with valid size/DPI."""
import hashlib
import importlib.util
from pathlib import Path

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_release", ROOT / "tools/verify_agent2.py")
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
