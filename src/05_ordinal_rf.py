"""第 5 部分训练入口：调参、验证、最终测试、保存结果。

运行：python src/05_ordinal_rf.py --n-jobs 8
"""

import argparse
import importlib
import json
import subprocess
import time
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ordinal_forest import OrdinalRandomForest, cumulative_to_proba
from utils_metrics import evaluate_predictions

# 文件名以数字开头，用 importlib 导入回归分级函数。
R = importlib.import_module("05_reg_to_cls")
ROOT = R.ROOT
MAIN_METRICS = ("Macro_F1", "Balanced_Accuracy", "High_Risk_Recall", "MAOE",
                "Severe_Jumping_Error_Rate", "Avg_Asymmetric_Cost")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def search_model(kind, train, val, features, folds, jobs):
    """两个模型用相同的候选结构和时间折；回归预测要先转成等级。"""
    X = train[features].to_numpy(dtype=np.float32)
    XV = val[features].to_numpy(dtype=np.float32)
    yt = train.target_level_3h.to_numpy()
    yv = val.target_level_3h.to_numpy()
    target = train.target_pm25_3h.to_numpy() if kind == "M5" else yt
    model_class = RandomForestRegressor if kind == "M5" else OrdinalRandomForest
    trace, fitted = [], []
    for i, params in enumerate(R.CANDIDATES):
        start = time.perf_counter()
        scores = []
        for tr, va in folds:
            model = model_class(**params, random_state=42, n_jobs=jobs)
            model.fit(X[tr], target[tr])
            prediction = model.predict(X[va])
            if kind == "M5":
                prediction = R.levels(prediction)
            scores.append(evaluate_predictions(yt[va], prediction))
        model = model_class(**params, random_state=42, n_jobs=jobs)
        model.fit(X, target)
        prediction = model.predict(XV)
        if kind == "M5":
            prediction = R.levels(prediction)
        metric = evaluate_predictions(yv, prediction)
        record = {"params": params, "cv_fold_metrics": scores,
                  "cv_mean_macro_f1": float(np.mean([s["Macro_F1"] for s in scores])),
                  "validation_metrics_seed42": metric,
                  "elapsed_seconds": round(time.perf_counter() - start, 3)}
        trace.append(record)
        fitted.append(model)
        print(f"{kind} {i+1}/{len(R.CANDIDATES)}: CV F1={record['cv_mean_macro_f1']:.4f}, "
              f"val F1={metric['Macro_F1']:.4f}, {record['elapsed_seconds']:.1f}s", flush=True)
    selected = max(range(len(trace)), key=lambda i: (
        trace[i]["validation_metrics_seed42"]["Macro_F1"],
        -trace[i]["validation_metrics_seed42"]["MAOE"],
        trace[i]["cv_mean_macro_f1"], -trace[i]["params"]["max_depth"]))
    return fitted[selected], trace[selected]["params"], trace


def clipping_diagnostics(raw):
    bad = np.any(np.diff(raw, axis=1) > 1e-12, axis=1)
    corrected = np.minimum.accumulate(raw, axis=1)
    return {"violating_samples_before": int(bad.sum()),
            "violation_rate_before": float(bad.mean()),
            "violating_samples_after": int(np.any(np.diff(corrected, axis=1) > 1e-12, axis=1).sum()),
            "max_downward_correction": float(np.max(raw - corrected)),
            "n_samples": len(raw)}


def summary(runs):
    result = {}
    for name in runs[0]:
        result[name] = {}
        for key in MAIN_METRICS:
            values = [run[name][key] for run in runs]
            result[name][key] = {"mean": float(np.mean(values)),
                                 "std": float(np.std(values, ddof=1))}
    return result


def error_figure(y, predictions, output):
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, axes = plt.subplots(1, len(predictions), figsize=(12, 4.1),
                                 sharey=True, layout="constrained")
        for ax, (name, pred) in zip(axes, predictions.items()):
            err = pred.astype(int) - y.astype(int)
            counts = np.array([(err == k).sum() for k in range(-3, 4)])
            bars = ax.bar(np.arange(-3, 4), counts / len(y) * 100,
                          color="#0072B2", edgecolor="#333333", linewidth=0.5)
            for k, bar in zip(range(-3, 4), bars):
                if abs(k) >= 2:
                    bar.set_facecolor("#D55E00")
                    bar.set_hatch("//")
            ax.set(title=f"{name}\nSevere jumps: {np.mean(np.abs(err)>=2):.2%}",
                   xlabel="Predicted level - true level", xticks=range(-3, 4))
            ax.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel("Share of test samples (%)")
        fig.suptitle(f"Fixed test split; seed 42; n={len(y):,}. Hatched bars: |error| >= 2")
        fig.savefig(output, dpi=300, facecolor="white")
        plt.close(fig)


def interval_figure(test, point, quantiles, overall, output):
    # 固定展示 Sepang 最早的 168 个测试小时，保留缺失小时。
    station = "Sepang"
    data = test[["station_id", "timestamp", "target_pm25_3h"]].copy()
    data["mean"] = point
    data[["q05", "q50", "q95"]] = quantiles
    data = data.loc[data.station_id.eq(station)].set_index("timestamp")
    start = data.index.min()
    data = data.reindex(pd.date_range(start, periods=168, freq="h"))
    horizon = data.index + pd.Timedelta(hours=3)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(11.5, 4.6), layout="constrained")
        ax.fill_between(horizon, data.q05.to_numpy(), data.q95.to_numpy(),
                        color="#56B4E9", alpha=0.4, label="5th-95th tree prediction percentiles")
        ax.plot(horizon, data.target_pm25_3h, color="#222222", linewidth=1.4,
                label="CAMS target (not ground observation)")
        ax.plot(horizon, data["mean"], color="#0072B2", linewidth=1.2,
                linestyle="--", label="RF mean")
        ax.plot(horizon, data.q50, color="#D55E00", linewidth=1,
                linestyle=":", label="Tree median")
        ax.set(xlabel="Target time t+3h (Kuala Lumpur local time)",
               ylabel="PM2.5 (µg/m³)", ylim=(0, None),
               title=f"M5, {station}: first 168 test hours; gaps retained\n"
                     f"Full-test PICP={overall['PICP']:.1%}; MPIW={overall['MPIW']:.2f} µg/m³. "
                     "Band is uncalibrated.")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=8, ncols=2)
        fig.savefig(output, dpi=300, facecolor="white")
        plt.close(fig)


def main():
    # 1. 读数据，检查输入和时间划分。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    parts, features, hashes = R.load_data()
    folds = R.chronological_folds(parts["train"].timestamp)
    print("Validated rows:", {s: len(p) for s, p in parts.items()}, flush=True)
    if args.validate_only:
        print("25 features, targets, receptor nodes, split boundaries and five purged CV folds: OK")
        return
    model_dir, figure_dir = ROOT / "models", ROOT / "figures"
    metric_path = model_dir / "metrics_stage5.json"
    frozen_path = model_dir / "stage5_selection.json"
    if metric_path.exists() or frozen_path.exists():
        raise FileExistsError("Stage 05 already has an audit record. Do not repeat test evaluation. "
                              "Use the saved predictions for checks and plots.")
    figure_dir.mkdir(exist_ok=True)
    train, val, test = (parts[s] for s in ("train", "val", "test"))
    X, XV = (p[features].to_numpy(dtype=np.float32) for p in (train, val))
    yv = val.target_level_3h.to_numpy()
    y_train = train.target_level_3h.to_numpy()
    # 2. 分别比较三组参数，按验证集结果选定。
    reg, reg_params, reg_trace = search_model("M5", train, val, features, folds, args.n_jobs)
    ordinal, ord_params, ord_trace = search_model("M6", train, val, features, folds, args.n_jobs)
    val_trees = R.tree_predictions(reg, XV)
    reg_val = val_trees.mean(axis=1)
    q_val = np.quantile(val_trees, [0.05, 0.5, 0.95], axis=1).T
    # 3. 只用验证集调切点，再检查三个随机种子的稳定性。
    cuts, cut_trace = R.search_cut_points(reg_val, yv)
    raw_val = ordinal.predict_cumulative(XV)
    p_val = cumulative_to_proba(raw_val)
    validation_predictions = {"M5_fixed": R.levels(reg_val),
                              "M5_val_tuned": R.levels(reg_val, cuts),
                              "M6_Ordinal_RF": p_val.argmax(axis=1).astype(np.int8)}
    val_metrics = {name: evaluate_predictions(yv, pred,
                   p_val if name == "M6_Ordinal_RF" else None)
                   for name, pred in validation_predictions.items()}
    seed_runs = [{"seed": 42, "validation_metrics": val_metrics}]
    for seed in (43, 44):
        print(f"Validation stability: seed {seed}", flush=True)
        reg_seed = RandomForestRegressor(**reg_params, random_state=seed, n_jobs=args.n_jobs)
        reg_seed.fit(X, train.target_pm25_3h)
        ord_seed = OrdinalRandomForest(**ord_params, random_state=seed,
                                      n_jobs=args.n_jobs).fit(X, y_train)
        pred = reg_seed.predict(XV)
        prob = ord_seed.predict_proba(XV)
        seed_runs.append({"seed": seed, "validation_metrics": {
            "M5_fixed": evaluate_predictions(yv, R.levels(pred)),
            "M5_val_tuned": evaluate_predictions(yv, R.levels(pred, cuts)),
            "M6_Ordinal_RF": evaluate_predictions(yv, prob.argmax(axis=1), prob)}})
    stability = summary([r["validation_metrics"] for r in seed_runs])
    # 4. 保存选定模型和参数，之后才预测测试集。
    reg.selected_features_ = features
    reg.level_thresholds_fixed_ = R.THRESHOLDS.tolist()
    reg.level_thresholds_val_tuned_ = cuts
    ordinal.selected_features_ = features
    for model, name in ((reg, "rf_regressor.joblib"), (ordinal, "ordinal_rf.joblib")):
        path = model_dir / name
        joblib.dump(model, path, compress=("xz", 6))
        if path.stat().st_size >= 20 * 1024**2:
            raise ValueError(f"model exceeds 20 MB: {path}")
    sources = {name: R.sha256(ROOT / "src" / name) for name in
               ("05_reg_to_cls.py", "05_ordinal_rf.py", "ordinal_forest.py", "utils_metrics.py", "config.py")}
    frozen = {"frozen_at": datetime.now(timezone.utc).isoformat(),
              "input_sha256": hashes, "code_sha256": sources,
              "selected_regressor_params": reg_params, "selected_ordinal_params": ord_params,
              "ordinal_class_weight": "balanced_subsample", "final_seed": 42,
              "m5_fixed_thresholds": R.THRESHOLDS.tolist(), "m5_val_tuned_thresholds": cuts,
              "test_evaluation_started": True,
              "note": "All choices fixed before test predictions. No selection uses test metrics."}
    write_json(frozen_path, frozen)
    print("All choices frozen. Starting the single final test pass.", flush=True)
    # 测试集只做这一轮，后续核对和画图读取保存的预测。
    XT = test[features].to_numpy(dtype=np.float32)
    yt = test.target_level_3h.to_numpy()
    test_trees = R.tree_predictions(reg, XT)
    reg_test = test_trees.mean(axis=1)
    q_test = np.quantile(test_trees, [0.05, 0.5, 0.95], axis=1).T
    raw_test = ordinal.predict_cumulative(XT)
    p_test = cumulative_to_proba(raw_test)
    test_predictions = {"M5_fixed": R.levels(reg_test),
                        "M5_val_tuned": R.levels(reg_test, cuts),
                        "M6_Ordinal_RF": p_test.argmax(axis=1).astype(np.int8)}
    test_metrics = {name: evaluate_predictions(yt, pred,
                    p_test if name == "M6_Ordinal_RF" else None)
                    for name, pred in test_predictions.items()}
    np.savez_compressed(model_dir / "ordinal_probs_val_test.npz",
        proba_val=p_val, proba_test=p_test, y_val=yv, y_test=yt,
        ts_val=val.timestamp.to_numpy(), ts_test=test.timestamp.to_numpy(),
        station_id_val=val.station_id.to_numpy(dtype=str),
        station_id_test=test.station_id.to_numpy(dtype=str))
    # 保存中间预测，后续可直接重画图。
    np.savez_compressed(model_dir / "stage5_predictions.npz",
        reg_val=reg_val, reg_test=reg_test, quantiles_val=q_val, quantiles_test=q_test,
        raw_cumulative_val=raw_val, raw_cumulative_test=raw_test,
        pred_m5_fixed_val=validation_predictions["M5_fixed"],
        pred_m5_fixed_test=test_predictions["M5_fixed"],
        pred_m5_tuned_val=validation_predictions["M5_val_tuned"],
        pred_m5_tuned_test=test_predictions["M5_val_tuned"],
        pred_m6_val=validation_predictions["M6_Ordinal_RF"],
        pred_m6_test=test_predictions["M6_Ordinal_RF"])
    interval_val = R.interval_metrics(val.target_pm25_3h, q_val)
    interval_test = R.interval_metrics(test.target_pm25_3h, q_test)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    # 5. 记录结果并画图。指标统一使用组内的 utils_metrics。
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit, "code_sha256": sources, "input_sha256": hashes,
        "training_domain": "role == receptor", "feature_names": features,
        "rows_by_split": {s: len(p) for s, p in parts.items()}, "random_seeds": list(R.SEEDS),
        "selection_rule": "val Macro_F1, then lower MAOE, then CV Macro_F1, then shallower tree; no test selection",
        "cv": {"n_splits": 5, "unit": "unique timestamp; nodes kept together",
               "purge_rule": "max(train timestamp)+3h < min(validation timestamp)",
               "upstream_selection_limitation": "Top-25 fixed upstream using all train labels; internal CV is not fully nested",
               "folds": [{"train_rows": len(a), "validation_rows": len(b),
                          "train_end": str(train.timestamp.iloc[a].max()),
                          "validation_start": str(train.timestamp.iloc[b].min())} for a,b in folds]},
        "parameter_candidates": R.CANDIDATES,
        "regression_search_trace": reg_trace, "ordinal_search_trace": ord_trace,
        "m5_threshold_search": {"selection_split": "val", "seed": 42,
                                "selected_thresholds": cuts, "trace": cut_trace},
        "seed_runs": seed_runs, "validation_mean_std": stability,
        "models": {name: {"validation_metrics_seed42": val_metrics[name],
                           "test_metrics_seed42": test_metrics[name],
                           "validation_mean_std": stability[name]}
                   for name in val_metrics},
        "monotonicity": {"val": clipping_diagnostics(raw_val), "test": clipping_diagnostics(raw_test)},
        "interval": {"source_model": "M5 RandomForestRegressor",
                     "method": "5/50/95 percentiles of individual tree point predictions",
                     "warning": "Tree spread is not calibrated outcome uncertainty; not a Quantile Regression Forest",
                     "validation": interval_val, "test": interval_test},
        "acceptance": {name: {"recall_at_least_0_85": stability[name]["High_Risk_Recall"]["mean"] >= 0.85,
                              "maoe_at_most_0_29": stability[name]["MAOE"]["mean"] <= 0.29}
                       for name in stability},
        "test_evaluation_count": 1,
        "artifacts": {name: R.sha256(model_dir / name) for name in (
            "rf_regressor.joblib", "ordinal_rf.joblib", "ordinal_probs_val_test.npz",
            "stage5_predictions.npz", "stage5_selection.json")}}
    write_json(metric_path, payload)
    error_figure(yt, test_predictions, figure_dir / "fig_ordinal_error_distribution.png")
    interval_figure(test, reg_test, q_test, interval_test, figure_dir / "fig_interval_band.png")
    print(json.dumps({"test": test_metrics, "interval_test": interval_test,
                      "monotonicity": payload["monotonicity"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
