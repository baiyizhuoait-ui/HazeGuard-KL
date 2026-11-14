"""Calibration checks use synthetic targets, never the real test split."""

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
C = importlib.import_module("05_interval_calibration")


class IntervalChecks(unittest.TestCase):
    def test_finite_sample_rank(self):
        # n=10 时 ceil(11 * .9)=10，必须取最大残差，不能取插值后的 90% 分位。
        fitted = C.fit_calibration(np.arange(1, 11), np.zeros(10))
        self.assertEqual(fitted["order_statistic_rank"], 10)
        self.assertEqual(fitted["half_width"], 10)

    def test_inference_only_uses_median_and_frozen_width(self):
        band = C.apply_calibration([2, 20], {"half_width": 5})
        np.testing.assert_array_equal(band, [[0, 2, 7], [15, 20, 25]])
        self.assertEqual(C.interval_scores([0, 26], band)["PICP"], .5)
        self.assertEqual(C.interval_scores([0, 26], band)["MPIW"], 8.5)

    def test_invalid_calibration(self):
        for y, median in (([], []), ([1], [1, 2]), ([np.nan], [1]), ([-1]*10, [1]*10)):
            with self.assertRaises(ValueError):
                C.fit_calibration(y, median)
        with self.assertRaises(ValueError):
            C.fit_calibration([1], [1])
        with self.assertRaises(ValueError):
            C.apply_calibration([1], {"half_width": -1})


if __name__ == "__main__":
    unittest.main()
