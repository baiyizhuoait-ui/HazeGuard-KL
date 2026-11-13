"""Synthetic checks for adaptive width and frozen-parameter inference."""
import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
A = importlib.import_module("05_interval_adaptive")


class AdaptiveIntervalChecks(unittest.TestCase):
    def test_normalized_finite_sample_quantile(self):
        q = np.tile([1., 2., 3.], (10, 1))
        cal = A.fit_calibration(np.arange(3., 13.), q)
        self.assertEqual(cal["order_statistic_rank"], 10)
        self.assertEqual(cal["scale_multiplier"], 10.)

    def test_width_tracks_spread_and_floor(self):
        q = [[2., 2., 2.], [6., 10., 14.]]
        band = A.apply_calibration(q, {"scale_floor": 1., "scale_multiplier": 2.})
        np.testing.assert_array_equal(band, [[0., 2., 4.], [2., 10., 18.]])

    def test_bad_shapes_order_and_parameters(self):
        for q in ([[2., 1., 3.]], [[1., np.nan, 3.]], [[-1., 1., 2.]], [1., 2., 3.]):
            with self.assertRaises(ValueError):
                A.interval_scale(q)
        with self.assertRaises(ValueError):
            A.fit_calibration([1.], [[0., 1., 2.]])
        with self.assertRaises(ValueError):
            A.apply_calibration([[0., 1., 2.]], {"scale_floor": 0., "scale_multiplier": 1.})


if __name__ == "__main__":
    unittest.main()
