"""Checks for module 07: the analysis logic, the saved results and the dashboard inputs.

    python -m unittest discover -s tests -p test_stage7.py -v

None of these tests needs a parquet engine or streamlit; they use the arrays and
JSON files that are committed under models/.
"""

import csv
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT / "src"))

from utils_metrics import evaluate_predictions  # noqa: E402


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load_module("system_analysis", "src/07_system_analysis.py")
BENCHMARK_COLUMNS = ["model", "split", "seed", "Macro_F1", "Balanced_Accuracy",
                     "High_Risk_Recall", "MAOE", "Severe_Jumping_Error_Rate",
                     "Avg_Asymmetric_Cost", "n_samples"]
# docs/spec_v3.md v3.2 adds feature_set as the second column
BENCHMARK_COLUMNS_V32 = BENCHMARK_COLUMNS[:1] + ["feature_set"] + BENCHMARK_COLUMNS[1:]


class OnsetLogic(unittest.TestCase):
    def test_onset_needs_six_quiet_observed_hours(self):
        series = np.array([10] * 6 + [40, 50, 20] + [10] * 6 + [60] + [10] * 3 + [70], float)
        self.assertEqual(analysis.find_onsets(series), [6, 15])
        gap = series.copy()
        gap[3] = np.nan                       # a missing hour inside the quiet run
        self.assertEqual(analysis.find_onsets(gap), [15])

    def test_lead_counts_only_forecasts_valid_at_or_after_onset(self):
        series = np.array([10] * 6 + [40, 50, 20] + [10] * 6 + [60] + [10] * 4, float)
        grid = pd.DataFrame({"pm2_5": series,
                             **{c: 0.0 for c in analysis.RULE_COLUMNS.values()}},
                            index=pd.date_range("2026-01-01", periods=len(series), freq="h"))
        grid.loc[grid.index[3], "pred_m5_tuned"] = 2     # issued 3 h before the onset at 6
        grid.loc[grid.index[1], "pred_m7"] = 3           # valid at hour 4, before the onset
        grid.loc[grid.index[16], "pred_m6"] = 3          # after the one-hour episode at 15 ended
        events = analysis.station_events(grid, "X")
        self.assertEqual([e["onset"][11:16] for e in events], ["06:00", "15:00"])
        self.assertEqual(events[0]["model_lead_h"]["M5_val_tuned"], analysis.HORIZON_H)
        self.assertIsNone(events[0]["model_lead_h"]["M7"])
        self.assertIsNone(events[1]["model_lead_h"]["M6"])
        self.assertEqual(events[0]["duration_h"], 2)

    def test_breakdown_partitions_the_onsets(self):
        events = [{"onset": f"2026-01-01T{h:02d}:00:00", "peak_pm25": peak,
                   "model_lead_h": {rule: lead for rule in analysis.RULE_COLUMNS}}
                  for h, peak, lead in [(8, 60.0, 3), (20, 40.0, None), (13, 80.0, 0)]]
        split = analysis.onset_breakdown(events, [8, 20])
        self.assertEqual(split["at_run_boundary"]["n_onsets"], 2)
        self.assertEqual(split["other_hours"]["n_onsets"], 1)
        self.assertEqual(split["major"]["n_onsets"], 2)
        self.assertEqual(split["at_run_boundary"]["rules"]["M7"]["missed"], 1)
        self.assertEqual(sum(split["onsets_by_hour"].values()), 3)

    def test_reliability_of_a_known_case(self):
        result = analysis.reliability(np.array([.05, .05, .95, .95]), np.array([0, 0, 1, 1.]))
        self.assertAlmostEqual(result["ece"], 0.05)
        self.assertAlmostEqual(result["brier"], 0.0025)


class SavedInputs(unittest.TestCase):
    def test_threshold_rule_reproduces_the_registered_m7_cost(self):
        saved = json.loads((MODELS / "cost_thresholds.json").read_text(encoding="utf-8"))
        probs = np.load(MODELS / "ordinal_probs_val_test.npz")
        pred = analysis.apply_thresholds(probs["proba_val"], saved["thresholds"])
        cost = evaluate_predictions(probs["y_val"], pred)["Avg_Asymmetric_Cost"]
        self.assertAlmostEqual(cost, saved["validation_cost"], places=3)

    def test_the_two_prediction_files_are_row_aligned(self):
        probs = np.load(MODELS / "ordinal_probs_val_test.npz")
        stage5 = np.load(MODELS / "stage5_predictions.npz")
        for split in ("val", "test"):
            self.assertEqual(len(probs[f"proba_{split}"]), len(stage5[f"reg_{split}"]))
            self.assertTrue((probs[f"proba_{split}"].argmax(axis=1)
                             == stage5[f"pred_m6_{split}"]).all())

    def test_calibrated_interval_lines_up_with_the_predictions(self):
        path = MODELS / "stage5_intervals_calibrated.npz"
        if not path.exists():
            self.skipTest("interval calibration not merged")
        cal, probs = np.load(path), np.load(MODELS / "ordinal_probs_val_test.npz")
        for split in ("val", "test"):
            self.assertTrue((cal[f"ts_{split}"] == probs[f"ts_{split}"]).all())
            self.assertTrue((cal[f"station_id_{split}"] == probs[f"station_id_{split}"]).all())
            interval = cal[f"interval_{split}"]
            self.assertTrue((interval[:, 0] <= interval[:, 1]).all())
            self.assertTrue((interval[:, 1] <= interval[:, 2]).all())

    def test_benchmark_has_the_contract_columns(self):
        with open(MODELS / "all_models_benchmark.csv", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        header = list(rows[0].keys())
        self.assertIn(header, (BENCHMARK_COLUMNS, BENCHMARK_COLUMNS_V32))
        if "feature_set" in header:          # M8 is the only v2 row (KD-01)
            for row in rows:
                self.assertEqual(row["feature_set"], "v2" if row["model"] == "M8" else "v1")

    def test_m8_script_reads_the_registered_v2_files(self):
        m8 = load_module("m8_v2_control", "src/07_m8_v2_control.py")
        manifest = json.loads((ROOT / m8.MANIFEST).read_text(encoding="utf-8"))
        self.assertIn(m8.V2_TABLE, manifest["tables"])
        features = json.loads((ROOT / m8.V2_FEATURES).read_text(encoding="utf-8"))
        names = features["selected_features"]
        self.assertEqual(len(names), 25)
        self.assertTrue({"hour_sin", "hour_cos"} <= set(names))


@unittest.skipUnless((MODELS / "metrics_stage7.json").exists(), "run src/07_system_analysis.py")
class SavedResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((MODELS / "metrics_stage7.json").read_text(encoding="utf-8"))

    def test_provenance_fields(self):
        for key in ("generated_at", "git_commit", "code_sha256", "input_sha256", "definitions"):
            self.assertIn(key, self.data)
        self.assertEqual(len(self.data["git_commit"]), 40)

    def test_summary_matches_the_event_list(self):
        for split in ("val", "test"):
            block = self.data["early_warning"][split]
            events, summary = block["events"], block["summary"]
            self.assertEqual(summary["n_onsets"], len(events))
            for rule, stats in summary["rules"].items():
                counts = stats["lead_counts"]
                self.assertEqual(sum(counts.values()), len(events))
                missed = sum(e["model_lead_h"][rule] is None for e in events)
                self.assertEqual(counts["missed"], missed)
                self.assertTrue(all(e["model_lead_h"][rule] is None
                                    or e["model_lead_h"][rule] <= analysis.HORIZON_H
                                    for e in events))

    def test_calibration_bins_cover_every_row(self):
        for split, rows in self.data["rows_by_split"].items():
            for block in self.data["calibration"][split].values():
                self.assertEqual(sum(b["n"] for b in block["bins"]), rows)


if __name__ == "__main__":
    unittest.main()
