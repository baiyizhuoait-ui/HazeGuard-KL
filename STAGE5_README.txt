HazeGuard-KL / Member 05
======================

Local project
  This folder contains main plus selected read-only dependencies from members 03/04.
  No remote branch has been pushed or merged.
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
