"""Small stage-05 checks; synthetic data only, never real held-out scoring."""

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ordinal_forest import OrdinalRandomForest, cumulative_to_proba
R = importlib.import_module("05_reg_to_cls")


class Stage5Checks(unittest.TestCase):
    def test_probability_reconstruction(self):
        q = np.array([[.8, .2, .4], [0, 0, 0], [1, 1, 1], [.9, .6, .2]])
        p = cumulative_to_proba(q)
        np.testing.assert_allclose(p, [[.2,.6,0,.2], [1,0,0,0], [0,0,0,1], [.1,.3,.4,.2]])
        np.testing.assert_allclose(p.sum(axis=1), 1)
        self.assertTrue(np.all(p >= 0))
        with self.assertRaises(ValueError):
            cumulative_to_proba([[.2, np.nan, .1]])

    def test_single_class_and_serialization(self):
        X = np.arange(40, dtype=np.float32).reshape(20,2)
        for label in (0, 3):
            m = OrdinalRandomForest(n_estimators=3, min_samples_leaf=1, n_jobs=1)
            m = clone(m).fit(X, np.full(len(X), label))
            np.testing.assert_array_equal(m.predict(X), np.full(len(X), label))
            np.testing.assert_allclose(m.predict_proba(X).sum(axis=1), 1)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "model.joblib"
                joblib.dump(m, path)
                np.testing.assert_array_equal(joblib.load(path).predict(X), m.predict(X))
        # Train with missing intermediate levels, a case encountered in early CV folds.
        m.fit(X, np.repeat([0, 2], 10))
        self.assertEqual(m.predict_proba(X).shape, (20,4))
        self.assertTrue(np.isin(m.predict(X), [0,1,2,3]).all())

    def test_boundary_levels_and_interval(self):
        np.testing.assert_array_equal(R.levels([0,14.999,15,35,55,99]), [0,0,1,2,3,3])
        q = np.array([[1,2,3],[1,2,3],[1,2,3]])
        metric = R.interval_metrics([1,2,4], q)
        self.assertAlmostEqual(metric["PICP"], 2/3)
        self.assertEqual(metric["MPIW"], 2)

    def test_time_folds_group_nodes_and_purge_horizon(self):
        ts = pd.date_range("2024-01-01", periods=80, freq="h").delete([6,9,21])
        repeated = pd.Series(np.repeat(ts.to_numpy(), 3))
        for tr, va in R.chronological_folds(repeated):
            self.assertLess(repeated.iloc[tr].max()+pd.Timedelta(hours=3), repeated.iloc[va].min())
            for t in repeated.iloc[va].unique():
                self.assertEqual(np.count_nonzero(repeated.iloc[va].to_numpy() == t), 3)
            self.assertEqual(len(np.intersect1d(tr,va)), 0)
        with self.assertRaises(ValueError):
            R.chronological_folds(repeated.iloc[::-1])


if __name__ == "__main__":
    unittest.main()
