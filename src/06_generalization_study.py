
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from utils_metrics import evaluate_predictions
# Project root
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "haze_panel_clean.parquet"


def load_verified_data():
    """
    Load data and verify exact 3-hour prediction targets.

    Original Parquet data is never modified.
    """

    print("===== Generalization Data Preparation =====")

    df = pd.read_parquet(DATA_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Prevent ambiguous station-time matching
    if df.duplicated(["station_id", "timestamp"]).any():
        raise ValueError("Duplicate station-time keys found.")

    # Prepare observations at t+3h
    future = df[
        ["station_id", "timestamp", "pm2_5"]
    ].copy()

    future["timestamp"] -= pd.Timedelta(hours=3)

    future = future.rename(
        columns={"pm2_5": "actual_future_pm25"}
    )

    # Join each row with the exact future observation
    checked = df.merge(
        future,
        on=["station_id", "timestamp"],
        how="left",
        validate="one_to_one"
    )

    # Verify the stored prediction target
    future_exists = checked["actual_future_pm25"].notna()

    target_matches = np.isclose(
        checked["target_pm25_3h"],
        checked["actual_future_pm25"],
        rtol=1e-6,
        atol=1e-6,
        equal_nan=False
    )

    verified = (
        future_exists
        & target_matches
        & checked["target_level_3h"].notna()
    )

    # Verify level encoding: [15, 35, 55]
    expected_levels = np.searchsorted(
        [15, 35, 55],
        checked.loc[verified, "target_pm25_3h"],
        side="right"
    )

    verified_index = checked.index[verified]

    verified.loc[verified_index] = (
        checked.loc[verified, "target_level_3h"].to_numpy()
        == expected_levels
    )

    clean = checked.loc[verified].copy()

    # Remove temporary audit column
    clean = clean.drop(columns=["actual_future_pm25"])

    print("Original rows:", len(df))
    print("Verified rows:", len(clean))
    print("Unverified rows:", len(df) - len(clean))
    print("Total nodes:", clean["station_id"].nunique())

    print("\nNodes by role:")
    print(
        clean.groupby("role")["station_id"].nunique()
    )

    return clean


def run_loso_pilot(df, heldout_station="KL_City"):
    """
    Run one Leave-One-Station-Out experiment.

    Train: other stations, train split only.
    Evaluate: held-out station, validation split only.
    """

    print("\n===== LOSO Pilot =====")
    print("Held-out station:", heldout_station)

    # 1. Load baseline model configuration
    bundle = joblib.load(
        ROOT / "models" / "baseline_dt.joblib"
    )

    features = list(bundle["features"])

    # Create a fresh, unfitted model
    model = clone(bundle["model"])

    # 2. Prepare features
    data = df.copy()

    rad = np.deg2rad(data["wind_direction_10m"])

    data["wd_sin"] = np.sin(rad)
    data["wd_cos"] = np.cos(rad)

    missing = [
        col for col in features
        if col not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    # 3. Select training and validation rows
    train_mask = (
        (data["station_id"] != heldout_station)
        & (data["split"] == "train")
    )

    val_mask = (
        (data["station_id"] == heldout_station)
        & (data["split"] == "val")
    )

    required = features + ["target_level_3h"]

    train_df = data.loc[train_mask].dropna(
        subset=required
    )

    val_df = data.loc[val_mask].dropna(
        subset=required
    )

    if train_df.empty or val_df.empty:
        raise ValueError("Training or validation data is empty.")

    # 4. Build model inputs in exact feature order
    X_train = train_df[features].to_numpy(
        dtype=np.float32
    )
    y_train = train_df["target_level_3h"].to_numpy(
        dtype=int
    )

    X_val = val_df[features].to_numpy(
        dtype=np.float32
    )
    y_val = val_df["target_level_3h"].to_numpy(
        dtype=int
    )

    print("Training samples:", len(y_train))
    print("Validation samples:", len(y_val))
    print("Training stations:",
          train_df["station_id"].nunique())

    # 5. Train a fresh model
    print("Training LOSO model...")
    model.fit(X_train, y_train)

    # 6. Predict the held-out station
    y_pred = model.predict(X_val)

    # 7. Evaluate using the team's unified metrics
    metrics = evaluate_predictions(
        y_true=y_val,
        y_pred=y_pred
    )

    print("\n===== LOSO Pilot Results =====")

    for name in [
        "Macro_F1",
        "Balanced_Accuracy",
        "High_Risk_Recall",
        "MAOE",
        "Severe_Jumping_Error_Rate",
        "Avg_Asymmetric_Cost"
    ]:
        value = metrics[name]
        print(f"{name}: {value}")

    return metrics

def run_loso_batch(df, station_names, output_filename):
    """
    Run LOSO for multiple stations.

    Each station is held out from model training.
    Save all six metrics and node roles.
    """

    metric_names = [
        "Macro_F1",
        "Balanced_Accuracy",
        "High_Risk_Recall",
        "MAOE",
        "Severe_Jumping_Error_Rate",
        "Avg_Asymmetric_Cost"
    ]

    results = []

    output_path = ROOT / "models" / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for fold, station in enumerate(station_names, start=1):

        print(f"\n========== Fold {fold}/{len(station_names)} ==========")

        # Find the role of the held-out station
        roles = df.loc[
            df["station_id"] == station, "role"
        ].unique()

        if len(roles) != 1:
            raise ValueError(
                f"Unexpected roles for station {station}: {roles}"
            )

        # Train and evaluate a fresh model
        metrics = run_loso_pilot(
            df,
            heldout_station=station
        )

        # Record results for this fold
        row = {
            "model": "M1_DecisionTree",
            "fold": fold,
            "station_id": station,
            "role": str(roles[0]),
            "n_samples": int(metrics["n_samples"]),
            "n_high_risk": int(metrics["n_high_risk"])
        }

        for name in metric_names:
            row[name] = metrics.get(name)

        results.append(row)

        # Save progress after each fold
        pd.DataFrame(results).to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

    print("\n===== LOSO Batch Completed =====")
    print("Completed folds:", len(results))
    print("Saved to:", output_path)

    return pd.DataFrame(results)


def run_temporal_holdout(df):
    """
    Temporal generalization experiment.

    Train: 2025 and earlier.
    Evaluate: 2026.

    Only receptor stations are used.
    No 2026 samples are used for training.
    """

    print("\n===== Temporal Holdout: 2025 -> 2026 =====")

    # 1. Load original baseline model configuration
    bundle = joblib.load(
        ROOT / "models" / "baseline_dt.joblib"
    )

    features = list(bundle["features"])

    # Fresh model, not the previously trained model
    model = clone(bundle["model"])

    # 2. Prepare features
    data = df.loc[df["role"] == "receptor"].copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    # Recreate wind-direction features
    rad = np.deg2rad(data["wind_direction_10m"])

    data["wd_sin"] = np.sin(rad)
    data["wd_cos"] = np.cos(rad)

    # 3. Define temporal boundary
    cutoff = pd.Timestamp("2026-01-01")

    # Training labels must refer to time before 2026
    train_mask = (
        data["timestamp"] + pd.Timedelta(hours=3)
        < cutoff
    )

    # Evaluate on 2026 only
    eval_mask = (
        (data["timestamp"] >= cutoff)
        & (data["timestamp"].dt.year == 2026)
    )

    required = features + ["target_level_3h"]

    train_df = data.loc[train_mask].dropna(
        subset=required
    )

    eval_df = data.loc[eval_mask].dropna(
        subset=required
    )

    if train_df.empty or eval_df.empty:
        raise ValueError("Training or evaluation data is empty.")

    # 4. Verify no temporal overlap
    assert (
        train_df["timestamp"].max()
        + pd.Timedelta(hours=3)
        < cutoff
    )

    assert eval_df["timestamp"].min() >= cutoff

    # 5. Prepare feature matrices
    X_train = train_df[features].to_numpy(
        dtype=np.float32
    )

    y_train = train_df["target_level_3h"].to_numpy(
        dtype=int
    )

    X_eval = eval_df[features].to_numpy(
        dtype=np.float32
    )

    y_eval = eval_df["target_level_3h"].to_numpy(
        dtype=int
    )

    print("Training samples:", len(y_train))
    print("2026 evaluation samples:", len(y_eval))
    print("Training stations:",
          train_df["station_id"].nunique())
    print("Evaluation stations:",
          eval_df["station_id"].nunique())

    # 6. Train on historical data only
    print("\nTraining temporal holdout model...")

    model.fit(X_train, y_train)

    # 7. Predict 2026 data
    y_pred = model.predict(X_eval)

    # 8. Team's unified six-metric evaluation
    metrics = evaluate_predictions(
        y_true=y_eval,
        y_pred=y_pred
    )

    metric_names = [
        "Macro_F1",
        "Balanced_Accuracy",
        "High_Risk_Recall",
        "MAOE",
        "Severe_Jumping_Error_Rate",
        "Avg_Asymmetric_Cost"
    ]

    print("\n===== Temporal Holdout Results (2026) =====")

    for name in metric_names:
        print(f"{name}: {metrics[name]}")

    # 9. Seasonal evaluation using the same 2026 predictions

    print("\n===== Seasonal Generalization (2026) =====")

    # Get evaluation months
    months = eval_df["timestamp"].dt.month.to_numpy()

    # Define the two periods in spec_v3
    monsoon_mask = np.isin(months, [6, 7, 8, 9, 10])

    seasonal_groups = {
        "Monsoon_Jun_Oct": monsoon_mask,
        "Other_Months": ~monsoon_mask
    }

    seasonal_results = []

    for season_name, mask in seasonal_groups.items():

        if not np.any(mask):
            raise ValueError(
                f"No evaluation samples for {season_name}"
            )

        # Evaluate predictions from the SAME trained model
        season_metrics = evaluate_predictions(
            y_true=y_eval[mask],
            y_pred=y_pred[mask]
        )

        row = {
            "model": "M1_DecisionTree",
            "evaluation_year": 2026,
            "season": season_name,
            "n_samples": int(np.sum(mask)),
            "n_high_risk": int(np.sum(y_eval[mask] >= 2))
        }

        for name in metric_names:
            row[name] = season_metrics[name]

        seasonal_results.append(row)

        print(f"\n{season_name}")
        print("Samples:", row["n_samples"])
        print("High-risk samples:", row["n_high_risk"])

        for name in metric_names:
            print(f"{name}: {season_metrics[name]}")

    # 10. Save seasonal results
    seasonal_df = pd.DataFrame(seasonal_results)

    if seasonal_df["n_samples"].sum() != len(y_eval):
        raise ValueError("Seasonal sample counts do not match.")

    output_path = ROOT / "models" / "seasonal_metrics.csv"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    seasonal_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== Seasonal Evaluation Completed =====")
    print("Groups evaluated:", len(seasonal_df))
    print("Total samples:", seasonal_df["n_samples"].sum())
    print("Saved to:", output_path)

    return metrics


if __name__ == "__main__":

    import sys

    data = load_verified_data()

    print("\nData preparation completed!")

    all_stations = sorted(
        data["station_id"].unique().tolist()
    )

    if len(all_stations) != 26:
        raise ValueError("Expected 26 spatial nodes.")

    if "--temporal" in sys.argv:

        # 2025 and earlier -> 2026
        run_temporal_holdout(data)

    elif "--all" in sys.argv:

        # Complete 26-node LOSO
        run_loso_batch(
            data,
            station_names=all_stations,
            output_filename="loso_results.csv"
        )

    else:

        # Two-node LOSO pilot
        run_loso_batch(
            data,
            station_names=["KL_City", "Alor_Setar"],
            output_filename="loso_pilot_results.csv"
        )