
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix
)

from utils_metrics import (
    DEFAULT_COST_MATRIX,
    evaluate_predictions,
    expected_cost
)

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Member 5 prediction file
DATA_PATH = (
    PROJECT_ROOT / "models" / "ordinal_probs_val_test.npz"
)


def load_data():
    """
    Load Member 5's ordinal prediction probabilities.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {DATA_PATH}"
        )

    with np.load(DATA_PATH, allow_pickle=False) as data:
        proba_val = data["proba_val"].astype(float)
        y_val = data["y_val"].astype(int)

    # Validate probability matrix
    assert proba_val.ndim == 2
    assert proba_val.shape[1] == 4
    assert len(proba_val) == len(y_val)

    assert np.isfinite(proba_val).all()
    assert np.all(proba_val >= 0)
    assert np.all(proba_val <= 1)
    assert np.allclose(
        proba_val.sum(axis=1), 1.0, atol=1e-5
    )

    assert np.isin(y_val, [0, 1, 2, 3]).all()

    return proba_val, y_val

def predict_with_thresholds(proba, tau):
    """
    Convert 4-class probabilities into ordinal levels.

    tau contains three cumulative probability thresholds.
    """
    proba = np.asarray(proba, dtype=float)
    tau = np.asarray(tau, dtype=float)

    # P(Y > 0), P(Y > 1), P(Y > 2)
    q1 = proba[:, 1:].sum(axis=1)
    q2 = proba[:, 2:].sum(axis=1)
    q3 = proba[:, 3]

    # Sequential ordinal decisions
    pass1 = q1 >= tau[0]
    pass2 = pass1 & (q2 >= tau[1])
    pass3 = pass2 & (q3 >= tau[2])

    return (
        pass1.astype(int)
        + pass2.astype(int)
        + pass3.astype(int)
    )


def average_asymmetric_cost(y_true, y_pred):
    """
    Calculate average cost using the team's matrix.
    """
    cm = np.asarray(DEFAULT_COST_MATRIX, dtype=float)

    return float(np.mean(cm[y_true, y_pred]))


def grid_search_thresholds(proba_val, y_val):
    """
    Search three ordinal thresholds using validation data.

    Objective: minimize Avg_Asymmetric_Cost.
    No GPSO or evolutionary algorithm is used.
    """

    # First use a coarse grid to verify the implementation
    grid = np.round(np.arange(0.05, 0.951, 0.05), 2)

    best_cost = float("inf")
    best_tau = None
    count = 0

    for tau1 in grid:
        for tau2 in grid:
            for tau3 in grid:

                tau = np.array([tau1, tau2, tau3])

                y_pred = predict_with_thresholds(
                    proba_val, tau
                )

                cost = average_asymmetric_cost(
                    y_val, y_pred
                )

                count += 1

                if cost < best_cost:
                    best_cost = cost
                    best_tau = tau.copy()

    return {
        "thresholds": best_tau,
        "validation_cost": best_cost,
        "combinations": count
    }



if __name__ == "__main__":

    print("===== M7 Cost-Sensitive Grid Search =====")

    proba_val, y_val = load_data()

    print("Validation probabilities:", proba_val.shape)
    print("Validation labels:", y_val.shape)

    # 1. Default threshold predictions
    default_tau = np.array([0.5, 0.5, 0.5])

    default_pred = predict_with_thresholds(
        proba_val, default_tau
    )

    default_cost = average_asymmetric_cost(
        y_val, default_pred
    )

    # 2. Grid search on validation data
    print("\nRunning Grid Search...")

    result = grid_search_thresholds(
        proba_val, y_val
    )

    optimized_pred = predict_with_thresholds(
        proba_val,
        result["thresholds"]
    )

    # 3. Bayesian minimum expected cost comparison
    bayes_pred = expected_cost(
        proba_val,
        DEFAULT_COST_MATRIX
    )

    bayes_cost = average_asymmetric_cost(
        y_val, bayes_pred
    )

    # 4. Compare validation costs
    print("\n===== Cost Comparison (Validation) =====")
    print("Default cost:", default_cost)
    print("Grid Search cost:", result["validation_cost"])
    print("Bayesian decision cost:", bayes_cost)
    print("Best thresholds:", result["thresholds"])
    print("Combinations searched:", result["combinations"])

    # 5. Unified six-metric evaluation
    metrics = evaluate_predictions(
        y_true=y_val,
        y_pred=optimized_pred,
        cost_matrix=DEFAULT_COST_MATRIX
    )

    print("\n===== Grid Search Metrics =====")
    for name in [
        "Macro_F1",
        "Balanced_Accuracy",
        "High_Risk_Recall",
        "MAOE",
        "Severe_Jumping_Error_Rate",
        "Avg_Asymmetric_Cost"
    ]:
        print(f"{name}: {metrics[name]}")

    # 6. Save the final validation grid-search result

    if result["combinations"] != 6859:
        raise ValueError(
            "Expected 6859 combinations. Check grid settings."
        )

    output_path = PROJECT_ROOT / "models" / "cost_thresholds.json"

    metric_names = [
        "Macro_F1",
        "Balanced_Accuracy",
        "High_Risk_Recall",
        "MAOE",
        "Severe_Jumping_Error_Rate",
        "Avg_Asymmetric_Cost"
    ]

    output_data = {
        "model": "M7",
        "method": "Grid Search",
        "objective": "Avg_Asymmetric_Cost",
        "selected_on": "validation",
        "thresholds": [
            float(x) for x in result["thresholds"]
        ],
        "grid_step": 0.05,
        "combinations_searched": int(result["combinations"]),
        "validation_samples": int(len(y_val)),
        "validation_cost": float(result["validation_cost"]),
        "default_cost": float(default_cost),
        "bayesian_cost": float(bayes_cost),
        "validation_metrics": {
            name: float(metrics[name])
            for name in metric_names
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print("\n===== File Saved Successfully =====")
    print("Saved to:", output_path)


    # 7. High-risk ROC analysis (validation only)

    print("\n===== High-Risk ROC Analysis =====")

    # Convert four levels into binary high-risk labels
    # Level 0/1 = 0, Level 2/3 = 1
    y_binary = (y_val >= 2).astype(int)

    # Probability of high risk: P(Level 2) + P(Level 3)
    high_risk_scores = proba_val[:, 2] + proba_val[:, 3]

    if len(np.unique(y_binary)) != 2:
        raise ValueError("ROC requires both binary classes.")

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(
        y_binary,
        high_risk_scores
    )

    auc_score = roc_auc_score(
        y_binary,
        high_risk_scores
    )

    # Grid-search operating point
    optimized_binary = (optimized_pred >= 2).astype(int)

    positive = y_binary == 1
    negative = y_binary == 0

    operating_tpr = np.mean(optimized_binary[positive] == 1)
    operating_fpr = np.mean(optimized_binary[negative] == 1)

    print(f"ROC AUC: {auc_score:.4f}")
    print(f"Operating TPR: {operating_tpr:.4f}")
    print(f"Operating FPR: {operating_fpr:.4f}")

    # Verify agreement with the team's High-Risk Recall
    assert np.isclose(
        operating_tpr,
        metrics["High_Risk_Recall"],
        atol=1e-4
    )

    # Draw ROC curve
    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"High-Risk ROC (AUC = {auc_score:.3f})",
        linewidth=2
    )

    plt.scatter(
        operating_fpr,
        operating_tpr,
        s=100,
        color="red",
        label="Grid Search Operating Point",
        zorder=5
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("High-Risk ROC - Validation Set")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    figure_path = (
        PROJECT_ROOT / "figures" / "fig_roc_high_risk.png"
    )

    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(figure_path, dpi=300)
    plt.close()

    print("ROC figure saved to:", figure_path)


    # 8. Confusion Matrix Comparison (Validation only)

    print("\n===== Confusion Matrix Comparison =====")

    comparison_models = {
        "Default Thresholds": default_pred,
        "M7 Grid Search": optimized_pred,
        "Bayesian Decision": bayes_pred
    }

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 5)
    )

    for ax, (name, predictions) in zip(
        axes,
        comparison_models.items()
    ):

        # Calculate the confusion matrix
        cm = confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1, 2, 3]
        )

        # Normalize each row
        row_sums = cm.sum(axis=1, keepdims=True)

        cm_normalized = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0
        )

        # Draw confusion matrix
        image = ax.imshow(
            cm_normalized,
            cmap="Blues",
            vmin=0,
            vmax=1
        )

        ax.set_title(name)
        ax.set_xlabel("Predicted Level")
        ax.set_ylabel("True Level")

        ax.set_xticks(range(4))
        ax.set_yticks(range(4))

        # Display counts and percentages
        for i in range(4):
            for j in range(4):

                count = cm[i, j]
                percentage = cm_normalized[i, j] * 100

                ax.text(
                    j, i,
                    f"{count}\n{percentage:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=(
                        "white"
                        if cm_normalized[i, j] > 0.5
                        else "black"
                    )
                )

        # Count high-risk missed alerts
        missed = np.sum(
            (y_val >= 2) & (predictions < 2)
        )

        print(f"{name}:")
        print(f"  Missed high-risk alerts: {missed}")

    fig.colorbar(
        image,
        ax=axes.tolist(),
        shrink=0.75,
        label="Row-normalized proportion"
    )

    # Save figure
    output_figure = (
        PROJECT_ROOT
        / "figures"
        / "fig_confusion_matrices_comparison.png"
    )

    output_figure.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        output_figure,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nConfusion matrix figure saved to:")
    print(output_figure)
