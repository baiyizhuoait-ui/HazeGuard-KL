
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# 1. Project paths
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "features_final.parquet"
MODEL_PATH = ROOT / "models" / "ordinal_rf.joblib"
FEATURE_PATH = ROOT / "models" / "selected_features.json"
NPZ_PATH = ROOT / "models" / "ordinal_probs_val_test.npz"


def load_interpretability_data():
    """
    Load the ordinal model and align validation rows
    with Member 5's saved prediction data.
    """

    print("Loading model and features...")

    model = joblib.load(MODEL_PATH)

    with open(FEATURE_PATH, "r", encoding="utf-8") as f:
        feature_info = json.load(f)

    feature_names = feature_info["selected_features"]

    df = pd.read_parquet(DATA_PATH)

    # Only use validation data
    val_df = df[df["split"] == "val"].copy()

    val_df = val_df.set_index(
        ["station_id", "timestamp"]
    )

    # Recover Member 5's exact validation row order
    with np.load(NPZ_PATH, allow_pickle=False) as z:
        stations = z["station_id_val"]
        timestamps = pd.to_datetime(z["ts_val"])
        y_val = z["y_val"].astype(int)
        saved_proba = z["proba_val"].copy()

    keys = pd.MultiIndex.from_arrays(
        [stations, timestamps],
        names=["station_id", "timestamp"]
    )

    if not val_df.index.is_unique:
        raise ValueError("Validation keys are not unique.")

    if not keys.isin(val_df.index).all():
        raise ValueError("Some validation rows are missing.")

    X_val = val_df.loc[keys, feature_names].to_numpy(
        dtype=np.float32
    )

    if X_val.shape != (len(y_val), 25):
        raise ValueError("Unexpected feature shape.")

    # Check prediction alignment before explanation
    predicted_proba = model.predict_proba(X_val[:20])

    max_diff = np.max(
        np.abs(predicted_proba - saved_proba[:20])
    )

    if max_diff > 1e-6:
        raise ValueError(
            f"Model/probability mismatch: {max_diff}"
        )

    print("Model loaded successfully!")
    print("Validation samples:", len(y_val))
    print("Features:", len(feature_names))
    print("Prediction alignment difference:", max_diff)

    return model, X_val, y_val, feature_names


def test_permutation_importance(
    model,
    X_val,
    y_val,
    feature_names
):
    """
    Small diagnostic run of permutation importance.

    Uses validation data only.
    """

    print("\nRunning Permutation Importance test...")

    # Stratified, reproducible validation sample
    indices, _ = train_test_split(
        np.arange(len(y_val)),
        train_size=300,
        random_state=42,
        stratify=y_val
    )

    X_sample = X_val[indices]
    y_sample = y_val[indices]

    baseline_pred = model.predict(X_sample)

    baseline_f1 = f1_score(
        y_sample,
        baseline_pred,
        average="macro",
        labels=[0, 1, 2, 3],
        zero_division=0
    )

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="f1_macro",
        n_repeats=2,
        random_state=42,
        n_jobs=1
    )

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    })

    importance = importance.sort_values(
        "importance_mean",
        ascending=False
    )

    print("\nBaseline Macro-F1:", round(baseline_f1, 4))
    print("\nTop 10 features:")
    print(importance.head(10).to_string(index=False))

def generate_permutation_figure(
    model,
    X_val,
    y_val,
    feature_names
):
    """
    Generate permutation importance figure using
    validation data and Macro-F1 scoring.
    """

    print("\nGenerating Permutation Importance figure...")

    # 1. Select 1000 validation samples
    indices, _ = train_test_split(
        np.arange(len(y_val)),
        train_size=1000,
        random_state=42,
        stratify=y_val
    )

    X_sample = X_val[indices]
    y_sample = y_val[indices]

    # 2. Use fixed four-class Macro-F1
    scorer = make_scorer(
        f1_score,
        average="macro",
        labels=[0, 1, 2, 3],
        zero_division=0
    )

    # 3. Calculate permutation importance
    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring=scorer,
        n_repeats=5,
        random_state=42,
        n_jobs=1
    )

    # 4. Create results table
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    })

    importance_df = importance_df.sort_values(
        "importance_mean",
        ascending=True
    )

    # 5. Plot all 25 features
    fig, ax = plt.subplots(figsize=(10, 9))

    ax.barh(
        importance_df["feature"],
        importance_df["importance_mean"],
        xerr=importance_df["importance_std"],
        capsize=2
    )

    ax.axvline(
        x=0,
        color="gray",
        linestyle="--",
        linewidth=1
    )

    ax.set_xlabel("Decrease in Macro-F1 after permutation")
    ax.set_ylabel("Feature")

    ax.set_title(
        "Permutation Feature Importance\n"
        "Ordinal Random Forest - Validation Set"
    )

    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()

    # 6. Save figure
    output_path = (
        ROOT / "figures" / "fig_perm_importance.png"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nFigure saved to:", output_path)

    print("\nTop 10 important features:")
    print(
        importance_df.tail(10)
        .sort_values("importance_mean", ascending=False)
        .to_string(index=False)
    )

def generate_baseline_pdp():
    """
    Generate Partial Dependence Plots for wind speed
    and relative humidity using the baseline decision tree.

    Uses validation receptor data only.
    """

    print("\n===== Baseline Decision Tree PDP =====")

    # 1. Load Member 1's baseline model
    baseline_path = ROOT / "models" / "baseline_dt.joblib"

    bundle = joblib.load(baseline_path)

    model = bundle["model"]
    features = list(bundle["features"])

    # 2. Load Member 1's original cleaned data
    data_path = ROOT / "data" / "haze_panel_clean.parquet"

    df = pd.read_parquet(data_path)

    # Use validation data from receptor nodes
    df = df.loc[
        (df["split"] == "val") &
        (df["role"] == "receptor")
    ].copy()

    # 3. Recreate wind direction features
    # Exactly as Member 1 implemented
    rad = np.deg2rad(df["wind_direction_10m"])

    df["wd_sin"] = np.sin(rad)
    df["wd_cos"] = np.cos(rad)

    # 4. Remove rows with missing model inputs
    df = df.dropna(
        subset=features + ["target_level_3h"]
    )

    if df.empty:
        raise ValueError("No valid validation samples.")

    # Preserve the original model feature order
    X = df[features].astype(np.float32)

    # 5. Sample validation rows reproducibly
    X = X.sample(
        n=min(2000, len(X)),
        random_state=42
    )

    print("Model:", type(model).__name__)
    print("Features:", len(features))
    print("PDP samples:", len(X))

    # 6. Identify high-risk probability columns
    classes = np.asarray(model.classes_)
    high_risk_cols = np.isin(classes, [2, 3])

    if not np.any(high_risk_cols):
        raise ValueError("Model has no high-risk classes.")

    # 7. Define features for PDP
    pdp_features = [
        ("wind_speed_ms", "Wind Speed (m/s)"),
        ("relative_humidity_2m", "Relative Humidity (%)")
    ]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5)
    )

    X_array = X.to_numpy(dtype=np.float32)

    for ax, (feature, label) in zip(axes, pdp_features):

        if feature not in features:
            raise ValueError(
                f"Feature not found in baseline model: {feature}"
            )

        feature_index = features.index(feature)

        # Use 5th to 95th percentile to avoid extremes
        grid = np.unique(
            np.quantile(
                X[feature],
                np.linspace(0.05, 0.95, 25)
            )
        )

        if len(grid) < 2:
            raise ValueError(
                f"Insufficient variation: {feature}"
            )

        pdp_values = []

        # 8. Calculate partial dependence
        for value in grid:

            X_modified = X_array.copy()

            X_modified[:, feature_index] = value

            probabilities = model.predict_proba(
                X_modified
            )

            high_risk_probability = probabilities[
                :, high_risk_cols
            ].sum(axis=1)

            pdp_values.append(
                high_risk_probability.mean()
            )

        # 9. Plot PDP
        ax.plot(
            grid,
            pdp_values,
            linewidth=2
        )

        ax.set_xlabel(label)
        ax.set_ylabel("Average P(High Risk)")
        ax.set_title(f"PDP: {label}")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Partial Dependence - Baseline Decision Tree (M1)"
    )

    fig.tight_layout()

    # 10. Save figure
    output_path = (
        ROOT / "figures" / "fig_pdp_wind_humidity.png"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nPDP figure saved successfully!")
    print("Saved to:", output_path)




if __name__ == "__main__":

    import sys

    if "--pdp" in sys.argv:
        # Generate PDP using Member 1's baseline model
        generate_baseline_pdp()

    else:
        # Generate permutation importance using Member 5's model
        model, X_val, y_val, feature_names = (
            load_interpretability_data()
        )

        generate_permutation_figure(
            model,
            X_val,
            y_val,
            feature_names
        )