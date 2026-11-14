HazeGuard-KL / Member 05
======================

Local project
  This folder contains main plus selected read-only dependencies from members 03/04.
  The original stage-05 delivery was merged through PR #5.
  The interval-calibration supplement below is delivered through a separate PR for review.
  The stage-05 delivery ZIP contains only member-05 files, not upstream files.

Requirements
  Python 3.11-3.13, using the existing requirements.txt pins.
  Inputs from member 03:
    data/features_final.parquet
    models/selected_features.json
  Shared upstream files:
    src/config.py
    src/utils_metrics.py

Commands (from repository root)
  python src/05_ordinal_rf.py --validate-only
  python -m unittest discover -s tests -p test_stage5.py -v
  python tools/verify_stage5.py

The experiment has already been run. Use the saved models and predictions.
The verifier checks saved results without making new test predictions.
Training entry for an independently documented clean reproduction:
  python src/05_ordinal_rf.py --n-jobs 8
  The alternative entry src/05_reg_to_cls.py runs the same complete workflow.
  Do NOT run both entry points consecutively. Existing audit records block repeats.

This desktop session's prepared Python environment (from repository root):
  ..\.venv-haze\Scripts\python.exe
Example:
  ..\.venv-haze\Scripts\python.exe tools\verify_stage5.py

Read first
  docs/handover/05_ordinal_and_interval.md
  reports/sec2_6_and_2_7.md

Main outputs
  models/rf_regressor.joblib             M5: concentration regression
  models/ordinal_rf.joblib               M6: four-class probabilities/levels
  models/ordinal_probs_val_test.npz      downstream probability interface
  models/metrics_stage5.json             results, seeds, selection and hashes
  models/stage5_selection.json           choices frozen before test
  models/stage5_predictions.npz          diagnostics without predicting again
  figures/fig_ordinal_error_distribution.png
  figures/fig_interval_band.png

Interface cautions
  Add src to sys.path before loading ordinal_rf.joblib.
  Pass 25 engineered features in the exact selected_features_ order.
  Probability keys are proba_val/proba_test, as specified in spec_v3.
  Timestamps are Kuala Lumpur local feature times; target time is timestamp + 3h.
  Row identity is timestamp plus station_id, not timestamp alone.
  Tree prediction percentiles are uncalibrated spread, not guaranteed 90% coverage.
  For calibrated display, use interval_calibration.json and 05_interval_calibration.py.
  Target coverage is 90%; observed test coverage is 83.41%, not a 90% guarantee.
  The target is a CAMS model product, not ground monitoring truth.

Submission preparation
  The reports file contains evidence, equations and writing prompts for the student.
  Write the assessed prose yourself using your understanding and verified results.
  See the handover's spec clarifications before integrating with members 06/07.

Readability update
  Model parameters, feature order, data splits and saved predictions are unchanged.
  Read main() in src/05_ordinal_rf.py for the five training steps.
  See models/metrics_stage5.json -> refactor for the old/new comparison checks.
  Original training hashes remain in code_sha256; refactor.code_sha256 describes current code.
  The updated delivery archives have _readable in their filenames.

Interval calibration supplement (2026-11-13)
  python -m unittest discover -s tests -p "test_stage5*.py" -v
  python src/05_interval_calibration.py
  Calibration has already run; the second command blocks overwriting its records.
  No retraining or new model predictions on test are needed.
  New files:
    src/05_interval_calibration.py
    tests/test_stage5_interval.py
    models/interval_calibration.json
    models/metrics_stage5_interval_calibrated.json
    models/stage5_intervals_calibrated.npz
  Calibration half-width: 13.820877192982458 ug/m3, fitted only on validation.
  Interval: [max(0, tree median - half_width), tree median + half_width].
  Validation PICP: 71.80% -> 90.01%; test PICP: 71.83% -> 83.41%.
  Test mean width: 20.79 -> 27.43 ug/m3.
  See handover section 9 for the member-07 interface and reporting limits.
  Original metrics, model weights, NPZ predictions and figures are retained.

Adaptive interval supplement (2026-11-13, recommended over fixed width)
  src/05_interval_adaptive.py supplies fit_calibration / apply_calibration.
  Selected on three chronological validation blocks, then frozen before test scoring.
  models/interval_calibration_adaptive.json stores frozen parameters and selection evidence.
  models/metrics_stage5_interval_adaptive.json stores overall and per-node results.
  models/stage5_intervals_adaptive.npz stores interval_val/test and row identity.
  The tests are in tests/test_stage5_interval_adaptive.py.
  Test PICP: raw tree spread 71.83%, fixed-width calibration 83.41%, adaptive width 87.97%.
  Test mean width: 20.79, 27.43, 32.11 ug/m3 respectively.
  Adaptive width improves observed coverage, but is not guaranteed 90%.
  Original weights, level thresholds and classification results remain unchanged.
  Integrate the adaptive supplement into main after reviewing its PR.
  See handover section 10 for the integration snippet and validation limits.
