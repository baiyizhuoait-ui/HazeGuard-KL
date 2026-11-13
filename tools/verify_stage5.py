"""Verify stage-05 artifacts from saved predictions; never predict test again."""

import hashlib
import importlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ordinal_forest import cumulative_to_proba
from utils_metrics import evaluate_predictions
R = importlib.import_module("05_reg_to_cls")


def text_hash_matches(path, expected):
    """Git can convert LF/CRLF on checkout; accept only that byte difference."""
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return expected in {hashlib.sha256(b).hexdigest()
                        for b in (raw, lf, lf.replace(b"\n", b"\r\n"))}


def verify():
    md = ROOT / "models"
    m = json.loads((md / "metrics_stage5.json").read_text(encoding="utf-8"))
    parts, names, hashes = R.load_data(ROOT)
    assert m["input_sha256"]["features_final"] == hashes["features_final"]
    assert text_hash_matches(md / "selected_features.json", m["input_sha256"]["selected_features"])
    assert m["feature_names"] == names
    assert m["test_evaluation_count"] == 1
    # 原训练摘要保留；重构后的代码使用单独登记的摘要。
    sources = m.get("refactor", {}).get("code_sha256", m["code_sha256"])
    for name, expected in sources.items():
        assert text_hash_matches(ROOT / "src" / name, expected), name
    for name, expected in m["artifacts"].items():
        if name.endswith(".json"):
            assert text_hash_matches(md / name, expected), name
        else:
            assert R.sha256(md / name) == expected, name
    frozen = json.loads((md / "stage5_selection.json").read_text(encoding="utf-8"))
    assert frozen["frozen_at"] <= m["generated_at"]
    cuts = m["m5_threshold_search"]["selected_thresholds"]
    assert cuts == frozen["m5_val_tuned_thresholds"]
    with np.load(md / "ordinal_probs_val_test.npz", allow_pickle=False) as p, \
         np.load(md / "stage5_predictions.npz", allow_pickle=False) as d:
        for split, suffix in (("val", "validation_metrics_seed42"), ("test", "test_metrics_seed42")):
            part = parts[split]
            n = len(part)
            prob, y = p[f"proba_{split}"], p[f"y_{split}"]
            assert prob.shape == (n,4) and y.shape == (n,)
            assert np.isfinite(prob).all() and (prob >= 0).all() and (prob <= 1).all()
            np.testing.assert_allclose(prob.sum(axis=1), 1, atol=1e-12)
            np.testing.assert_array_equal(p[f"ts_{split}"], part.timestamp.to_numpy())
            np.testing.assert_array_equal(p[f"station_id_{split}"], part.station_id.to_numpy(dtype=str))
            np.testing.assert_array_equal(y, part.target_level_3h)
            np.testing.assert_allclose(prob, cumulative_to_proba(d[f"raw_cumulative_{split}"]))
            predictions = {"M5_fixed": R.levels(d[f"reg_{split}"]),
                           "M5_val_tuned": R.levels(d[f"reg_{split}"], cuts),
                           "M6_Ordinal_RF": prob.argmax(axis=1)}
            for name, pred in predictions.items():
                actual = evaluate_predictions(y, pred, prob if name.startswith("M6") else None)
                assert actual == m["models"][name][suffix], (split, name)
            interval = R.interval_metrics(part.target_pm25_3h, d[f"quantiles_{split}"])
            assert interval == m["interval"]["validation" if split == "val" else "test"]
    # Load independently of a training __main__ module; only inspect metadata here.
    for filename in ("rf_regressor.joblib", "ordinal_rf.joblib"):
        model = joblib.load(md / filename)
        assert model.selected_features_ == names
        assert model.random_state == 42 and model.n_features_in_ == 25
        assert (md / filename).stat().st_size < 20 * 1024**2
    for filename in ("fig_ordinal_error_distribution.png", "fig_interval_band.png"):
        with Image.open(ROOT / "figures" / filename) as im:
            assert im.width >= 3000 and im.height >= 1000
            assert min(im.info.get("dpi", (0,0))) >= 299
    print("PASS: inputs/code/artifact hashes; probability and row contracts; saved metrics; "
          "intervals; model loading; 300dpi figures. No new test prediction.")


if __name__ == "__main__":
    verify()
