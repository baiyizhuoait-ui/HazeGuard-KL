
from pathlib import Path
from importlib import import_module
import json
import numpy as np

from utils_metrics import (
    DEFAULT_COST_MATRIX,
    evaluate_predictions,
    expected_cost
)

# Project paths
ROOT = Path(__file__).resolve().parents[1]

NPZ_PATH = ROOT / "models" / "ordinal_probs_val_test.npz"
THRESHOLD_PATH = ROOT / "models" / "cost_thresholds.json"
OUTPUT_PATH = ROOT / "models" / "m7_test_metrics.json"

# Reuse the already-tested threshold prediction function
m7_module = import_module("06_cost_thresholds")
predict_with_thresholds = m7_module.predict_with_thresholds

METRICS = [
    "Macro_F1",
    "Balanced_Accuracy",
    "High_Risk_Recall",
    "MAOE",
    "Severe_Jumping_Error_Rate",
    "Avg_Asymmetric_Cost"
]


def main():

    print("===== M7 FINAL TEST EVALUATION =====")

    # Prevent accidental repeated final evaluation
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            "Final test results already exist. "
            "Do not rerun or retune using the test set."
        )

    # 1. Load frozen thresholds selected on validation
    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        saved = json.load(f)

    if saved.get("selected_on") != "validation":
        raise ValueError("Thresholds must be selected on validation.")

    tau = np.asarray(saved["thresholds"], dtype=float)

    if tau.shape != (3,) or not np.all((tau > 0) & (tau < 1)):
        raise ValueError("Invalid saved thresholds.")

    print("Frozen M7 thresholds:", tau)

    # 2. Load test predictions
    with np.load(NPZ_PATH, allow_pickle=False) as data:
        proba_test = data["proba_test"].astype(float)
        y_test = data["y_test"].astype(int)

    assert proba_test.ndim == 2
    assert proba_test.shape[1] == 4
    assert len(proba_test) == len(y_test)
    assert np.isfinite(proba_test).all()
    assert np.all((proba_test >= 0) & (proba_test <= 1))
    assert np.allclose(proba_test.sum(axis=1), 1.0, atol=1e-5)
    assert np.isin(y_test, [0, 1, 2, 3]).all()

    print("Test samples:", len(y_test))

    # 3. Three prediction methods
    predictions = {
        "Default": predict_with_thresholds(
            proba_test,
            np.array([0.5, 0.5, 0.5])
        ),
        "M7_Grid_Search": predict_with_thresholds(
            proba_test,
            tau
        ),
        "Bayesian": expected_cost(
            proba_test,
            DEFAULT_COST_MATRIX
        )
    }

    # 4. Unified six-metric evaluation
    results = {}

    for method, pred in predictions.items():

        metrics = evaluate_predictions(
            y_true=y_test,
            y_pred=pred,
            cost_matrix=DEFAULT_COST_MATRIX
        )

        results[method] = {
            name: float(metrics[name])
            for name in METRICS
        }

    # 5. Print comparison
    print("\n===== TEST SET COMPARISON =====")

    for metric in METRICS:
        print(f"\n{metric}")

        for method, scores in results.items():
            print(f"  {method}: {scores[metric]:.4f}")

    # 6. Save final test results without overwriting
    output = {
        "model": "M7",
        "split": "test",
        "threshold_source": "validation",
        "thresholds": tau.tolist(),
        "test_samples": int(len(y_test)),
        "metrics": results
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "x", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print("\nFinal test results saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
