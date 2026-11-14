"""Synthetic checks: local-hour boundaries, group partition and aggregate MAE."""
import importlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
S = importlib.import_module("05_seam_error")


class SeamChecks(unittest.TestCase):
    def test_local_hour_boundaries_and_weighted_comparator(self):
        ts = pd.date_range("2026-01-01", periods=24, freq="h")
        y = np.zeros(24)
        error = np.ones(24)
        error[[5, 6, 7]] = 2
        error[[17, 18, 19]] = 5
        rows = {row["group"]: row for row in S.summarize(ts, y, error)}
        self.assertEqual(rows["cross_08"], {"group": "cross_08", "n_samples": 3, "MAE": 2.})
        self.assertEqual(rows["cross_20"], {"group": "cross_20", "n_samples": 3, "MAE": 5.})
        self.assertEqual(rows["other_hours"]["n_samples"], 18)
        self.assertAlmostEqual(rows["not_cross_20"]["MAE"], (3*2 + 18)/21)
        self.assertEqual(rows["not_cross_20"]["n_samples"], 21)

    def test_bad_alignment_and_timezone_rejected(self):
        ts = pd.date_range("2026-01-01", periods=24, freq="h")
        with self.assertRaises(ValueError):
            S.summarize(ts, np.zeros(23), np.zeros(24))
        with self.assertRaises(ValueError):
            S.summarize(ts.tz_localize("UTC"), np.zeros(24), np.zeros(24))
        y = np.zeros(24)
        y[0] = np.nan
        with self.assertRaises(ValueError):
            S.summarize(ts, y, np.zeros(24))


if __name__ == "__main__":
    unittest.main()
