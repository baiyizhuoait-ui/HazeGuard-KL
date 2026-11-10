"""Adversarial checks of stage 2's availability and physical conventions.

Run with: python -m pytest tests/test_stage2_contract.py
Fixtures are synthetic and are never used in the reported EDA or parquet.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F = load("features", "02_spatiotemporal_features.py")
E = load("eda", "02_eda_analysis.py")


@pytest.fixture
def panel():
    records = []
    for i, sid in enumerate(["West", "Center", "East"]):
        n = 75
        x = np.arange(n) * 2.0 + 10 * i
        for t in range(n):
            records.append(dict(station_id=sid, timestamp=pd.Timestamp("2024-01-01")+pd.Timedelta(hours=t),
                                split="train" if t < 35 else "val" if t < 55 else "test",
                                lat=0., lon=.1*i, pm2_5=x[t], pm10=x[t]+5, no2=3*x[t],
                                relative_humidity_2m=60., wind_speed_ms=2., wind_speed_kmh=7.2,
                                wind_direction_10m=270., qc_calm_wind=0, qc_interpolated=0, is_missing=0,
                                target_pm25_3h=x[t+3] if t+3 < n else np.nan,
                                target_level_3h=np.searchsorted([15,35,55], x[t+3], side="right") if t+3 < n else np.nan))
    return pd.DataFrame(records)


def select(df, sid, h):
    return df.loc[df.station_id.eq(sid) & df.timestamp.eq(pd.Timestamp("2024-01-01")+pd.Timedelta(hours=h))].iloc[0]


def predictors(df):
    return [c for c in df if c not in panel_original_columns() and not c.startswith("audit_")]


def panel_original_columns():
    return {"station_id", "timestamp", "split", "lat", "lon", "pm2_5", "pm10", "no2", "relative_humidity_2m",
            "wind_speed_ms", "wind_speed_kmh", "wind_direction_10m", "qc_calm_wind", "qc_interpolated",
            "is_missing", "target_pm25_3h", "target_level_3h"}


def test_preserves_rows_order_dtypes_and_inputs(panel):
    panel = panel.sample(frac=1, random_state=42)
    original = panel.copy(deep=True)
    out = F.build_features(panel)
    pd.testing.assert_frame_equal(out[original.columns], original)
    pd.testing.assert_frame_equal(panel, original)
    assert select(out, "Center", 30).pm2_5_lag_24 == 22


def test_lags_and_rolling_are_node_local_and_reset_at_every_split(panel):
    out = F.build_features(panel)
    assert select(out, "Center", 7).pm2_5_lag_6 == 12
    for h in [0, 35, 55]:
        row = select(out, "Center", h)
        assert pd.isna(row.pm2_5_lag_1)
        assert pd.isna(row.pm2_5_mean_3h)
        assert pd.isna(row.upwind_pm25_advection_1h)
        assert pd.isna(row.upwind_pm25_advection_2h)
    assert select(out, "Center", 37).pm2_5_mean_3h == 82


def test_rolling_trend_and_sample_std(panel):
    row = select(F.build_features(panel), "West", 10)
    assert row.pm2_5_mean_3h == 18
    assert row.pm2_5_std_3h == 2
    assert row.pm2_5_range_3h == 4
    assert row.pm2_5_change_3h == 6
    assert row.pm2_5_slope_6h == 2  # µg/m³ per hour, not per arbitrary row


def test_missing_hour_does_not_become_a_row_lag(panel):
    hole = panel[~(panel.station_id.eq("Center") & panel.timestamp.eq(pd.Timestamp("2024-01-01 05:00")))]
    out = F.build_features(hole)
    row = select(out, "Center", 6)
    assert pd.isna(row.pm2_5_lag_1)
    assert row.pm2_5_lag_3 == 16
    assert pd.isna(row.pm2_5_mean_3h)
    assert pd.isna(row.pm2_5_slope_6h)


def test_future_perturbation_cannot_change_issue_time_features(panel):
    base = F.build_features(panel)
    changed = panel.copy()
    future = changed.timestamp > pd.Timestamp("2024-01-01 20:00")
    changed.loc[future, ["pm2_5", "pm10", "no2", "relative_humidity_2m"]] = 987654.
    changed.loc[future, "wind_direction_10m"] = 90.
    altered = F.build_features(changed)
    cols = predictors(base)
    pd.testing.assert_frame_equal(base.loc[~future, cols], altered.loc[~future, cols])


def test_validation_test_cannot_change_training_predictors(panel):
    base = F.build_features(panel)
    changed = panel.copy()
    changed.loc[changed.split.ne("train"), "pm2_5"] = 999999.
    altered = F.build_features(changed)
    train = panel.split.eq("train")
    pd.testing.assert_frame_equal(base.loc[train, predictors(base)], altered.loc[train, predictors(base)])


def test_target_columns_never_feed_predictors(panel):
    base = F.build_features(panel)
    changed = panel.copy()
    changed[["target_pm25_3h", "target_level_3h"]] = 0
    altered = F.build_features(changed)
    pd.testing.assert_frame_equal(base[predictors(base)], altered[predictors(base)])


@pytest.mark.parametrize("direction,u,v", [(0,0,-2), (90,-2,0), (180,0,2), (270,2,0), (360,0,-2)])
def test_meteorological_wind_from_convention(panel, direction, u, v):
    panel.wind_direction_10m = direction
    a, b = F.wind_vectors(panel)
    np.testing.assert_allclose(a, u, atol=1e-12)
    np.testing.assert_allclose(b, v, atol=1e-12)


def test_unit_conversion_exactly_once_and_rejects_ambiguity(panel):
    np.testing.assert_allclose(F.wind_speed_ms(panel), 2)
    no_ms = panel.drop(columns="wind_speed_ms")
    np.testing.assert_allclose(F.wind_speed_ms(no_ms), 2)
    raw = no_ms.rename(columns={"wind_speed_kmh": "wind_speed_10m"})
    with pytest.raises(ValueError, match="explicit"):
        F.wind_speed_ms(raw)
    np.testing.assert_allclose(F.wind_speed_ms(raw, "km/h"), 2)
    panel.wind_speed_kmh = 72
    with pytest.raises(ValueError, match="disagrees"):
        F.build_features(panel)


def test_upwind_direction_self_exclusion_and_time_alignment(panel):
    out = F.build_features(panel)
    row = select(out, "Center", 10)
    assert row.upwind_pm25_advection_1h == 18  # West(t-1), not Center or West(t)
    assert row.upwind_pm25_advection_2h == 16
    assert row.spatial_pm25_gradient == 10  # Center(t) minus West(t)
    assert row.upwind_neighbor_count_1h == 1
    assert pd.isna(select(out, "West", 10).upwind_pm25_advection_1h)
    # Flip from westerly to easterly: only the eastern cell may contribute to Center.
    panel.wind_direction_10m = 90
    east = select(F.build_features(panel), "Center", 10)
    assert east.upwind_pm25_advection_1h == 38
    assert east.spatial_pm25_gradient == -10


def test_calm_zero_denominator_and_missing_neighbors(panel):
    panel[["wind_speed_ms", "wind_speed_kmh"]] = 0
    out = F.build_features(panel)
    assert out.u_wind.eq(0).all() and out.v_wind.eq(0).all()
    assert out.upwind_pm25_advection_1h.isna().all()
    assert select(out, "Center", 10).upwind_weight_sum_1h == 0
    assert select(out, "Center", 10).upwind_neighbor_count_1h == 0


def test_gaussian_weights_use_km_and_normalize_only_valid_neighbors(panel):
    row = select(F.build_features(panel, sigma_km=15), "East", 10)
    distance, _, _ = F.geometry([[0,0],[0,.1],[0,.2]])
    weights = 2 * np.exp(-distance[:2,2]**2 / (2*15**2))
    expected = np.dot(weights, [18,28]) / weights.sum()
    assert row.upwind_pm25_advection_1h == pytest.approx(expected, rel=1e-6)
    assert row.upwind_weight_sum_1h == pytest.approx(weights.sum(), rel=1e-6)
    assert row.upwind_neighbor_count_1h == 2
    # A remote node with a negligible kernel must not become a confident sole neighbor.
    panel.loc[panel.station_id.eq("West"), "lon"] = -10
    panel.loc[panel.station_id.eq("East"), "wind_direction_10m"] = 270
    assert pd.isna(select(F.build_features(panel), "Center", 10).upwind_pm25_advection_1h)


def test_calm_flag_threshold_and_colocated_nodes(panel):
    west = panel.station_id.eq("West")
    panel.loc[west, "qc_calm_wind"] = 1
    assert pd.isna(select(F.build_features(panel), "Center", 10).upwind_pm25_advection_1h)
    panel.loc[west, "qc_calm_wind"] = 0
    panel.loc[west, ["wind_speed_ms", "wind_speed_kmh"]] = [0.499, 0.499*3.6]
    assert pd.isna(select(F.build_features(panel), "Center", 10).upwind_pm25_advection_1h)
    panel.loc[west, ["wind_speed_ms", "wind_speed_kmh"]] = [0.5, 1.8]
    assert select(F.build_features(panel), "Center", 10).upwind_pm25_advection_1h == 18
    panel.loc[west, "lon"] = .1
    assert pd.isna(select(F.build_features(panel), "Center", 10).upwind_pm25_advection_1h)


def test_interpolated_rows_unavailable_without_changing_original(panel):
    mask = panel.station_id.eq("West") & panel.timestamp.eq(pd.Timestamp("2024-01-01 09:00"))
    panel.loc[mask, "qc_interpolated"] = 1
    out = F.build_features(panel)
    assert select(out, "West", 9).pm2_5 == 18
    assert not select(out, "West", 9).audit_source_clean
    assert pd.isna(select(out, "West", 10).pm2_5_lag_1)
    assert pd.isna(select(out, "Center", 10).upwind_pm25_advection_1h)


def test_spatial_source_must_be_same_split(panel):
    panel.loc[panel.station_id.eq("West") & (panel.timestamp >= pd.Timestamp("2024-01-02 10:00")), "split"] = "val"
    # Center(34h) is train; West(33h) is still train, but West(34h) is already val.
    row = select(F.build_features(panel), "Center", 34)
    assert row.upwind_pm25_advection_1h == 66
    assert pd.isna(row.spatial_pm25_gradient)


def test_sparse_spatial_history_keeps_exact_hour_alignment(panel):
    # The absence of West(9h) must not silently promote West(8h) to a 1h lag.
    missing = panel.station_id.eq("West") & panel.timestamp.eq(pd.Timestamp("2024-01-01 09:00"))
    out = F.build_features(panel.loc[~missing].sample(frac=1, random_state=9))
    row = select(out, "Center", 10)
    assert pd.isna(row.upwind_pm25_advection_1h)
    assert row.upwind_pm25_advection_2h == 16
    assert row.upwind_neighbor_count_1h == 0


def test_source_missingness_and_future_wind_cannot_change_past_predictors(panel):
    base = F.build_features(panel)
    changed = panel.copy()
    future = changed.timestamp > pd.Timestamp("2024-01-01 20:00")
    changed.loc[future, ["is_missing", "qc_interpolated", "qc_calm_wind"]] = 1
    changed.loc[future, ["wind_speed_ms", "wind_speed_kmh"]] = [20., 72.]
    changed.loc[future, ["target_pm25_3h", "target_level_3h"]] = [-999., -9.]
    altered = F.build_features(changed)
    pd.testing.assert_frame_equal(base.loc[~future, predictors(base)], altered.loc[~future, predictors(base)])


def test_git_revision_archive_provenance_and_no_parent_git(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "ROOT", tmp_path)
    # A source ZIP can sit inside another checkout. Its commit must never be inherited.
    def unexpected_git(*args, **kwargs):
        raise AssertionError("Git must not run without root .git metadata")
    monkeypatch.setattr(F.subprocess, "check_output", unexpected_git)
    assert F.git_revision() is None
    commit = "a1" * 20
    (tmp_path / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    assert F.git_revision() == commit
    (tmp_path / "SOURCE_COMMIT.txt").write_text("unverified", encoding="utf-8")
    assert F.git_revision() is None


def test_features_cli_from_extracted_source_without_git(panel, tmp_path):
    # Execute a copied script, not the imported repository module.
    source = tmp_path / "src"
    source.mkdir()
    shutil.copy2(ROOT / "src/02_spatiotemporal_features.py", source)
    shutil.copy2(ROOT / "src/stage2_storage.py", source)
    panel.to_parquet(tmp_path / "input.parquet", index=False)
    commit = "b2" * 20
    (tmp_path / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(source / "02_spatiotemporal_features.py"),
         "--input", str(tmp_path / "input.parquet"), "--output", str(tmp_path / "features.parquet"),
         "--audit", str(tmp_path / "audit.json")],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8",
        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["git_commit"] == commit
    assert audit["new_column_count"] == 63
    assert audit["new_predictor_count"] == 51
    assert sum(audit["feature_family_counts"].values()) == 63
    assert audit["integrity"]["duplicate_primary_keys"] == 0
    assert audit["integrity"]["original_columns_values_dtypes_and_row_order_preserved"]
    assert audit["integrity"]["non_hourly_consecutive_intervals"] == 0
    assert all(not c.startswith(("audit_", "target_")) for c in audit["model_feature_columns"])
    for name, count in audit["nan_counts"].items():
        assert audit["nan_fractions"][name] == count / len(panel)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "features.parquet")[panel.columns], panel)


def test_label_split_crossings_exact_horizon_and_thresholds(panel):
    out = F.build_features(panel)
    assert not select(out, "Center", 34).audit_target_within_split_3h
    assert not select(out, "Center", 34).audit_model_eligible
    assert not select(out, "Center", 74).audit_target_safe_3h
    assert select(out, "Center", 30).audit_target_safe_3h
    bad = panel.copy()
    bad.loc[bad.index[30], "target_pm25_3h"] += 1
    assert not F.build_features(bad).iloc[30].audit_target_aligned_3h
    # Exact class boundaries use left-closed bins.
    for value, level in [(14.999,0),(15.,1),(35.,2),(55.,3)]:
        x = panel.copy()
        idx = x.index[5]
        x.loc[idx, ["target_pm25_3h", "target_level_3h"]] = [value, level]
        x.loc[x.index[8], "pm2_5"] = value
        assert F.build_features(x).loc[idx, "audit_target_aligned_3h"]


def test_geometry_km_and_source_bearing():
    d, e, n = F.geometry([[0,0],[0,1],[1,0]])
    assert d[0,1] == pytest.approx(111.19508, rel=1e-6)
    assert e[0,1] == pytest.approx(1)
    assert n[0,2] == pytest.approx(1)
    assert e[1,0] == pytest.approx(-1)


def test_timezones_local_clock_and_hour_boundary(panel):
    local = F.build_features(panel)
    aware = panel.copy()
    aware.timestamp = aware.timestamp.dt.tz_localize(F.TIMEZONE).dt.tz_convert("UTC")
    out = F.build_features(aware)
    np.testing.assert_allclose(local.hour_sin, out.hour_sin)
    assert select(local, "West", 0).month_sin == 0
    assert select(local, "West", 0).hour_cos == 1
    panel.timestamp = panel.timestamp + pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="exact hours"):
        F.build_features(panel)


@pytest.mark.parametrize("kind", ["duplicate", "coords", "split", "negative_wind", "direction", "sigma", "overwrite"])
def test_rejects_unsafe_inputs(panel, kind):
    kw = {}
    if kind == "duplicate": panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    if kind == "coords": panel.loc[0,"lat"] = 91
    if kind == "split": panel.loc[0,"split"] = "test"
    if kind == "negative_wind": panel.loc[0,["wind_speed_ms","wind_speed_kmh"]] = [-1,-3.6]
    if kind == "direction": panel.loc[0,"wind_direction_10m"] = 361
    if kind == "sigma": kw["sigma_km"] = 0
    if kind == "overwrite": panel["u_wind"] = 0
    with pytest.raises(ValueError):
        F.build_features(panel, **kw)


def test_ccf_sign_uses_hours_and_not_row_offsets():
    rng = np.random.default_rng(42)
    clock = pd.date_range("2024-01-01", periods=200, freq="h")
    source = pd.Series(rng.normal(size=len(clock)), index=clock)
    destination = source.shift(3)
    points = E.lag_correlations(source.drop(clock[80]), destination)
    peak = max([p for p in points if p["r"] is not None], key=lambda p: p["r"])
    assert peak["lag_h"] == 3
    assert peak["r"] == pytest.approx(1)


def test_bootstrap_reproducible():
    x = np.arange(24*40).reshape(40,24)
    np.testing.assert_array_equal(E.block_diurnal_ci(x), E.block_diurnal_ci(x))


def test_feature_design_eda_rejects_heldout_or_empty_rows(panel):
    train = panel.loc[panel.split.eq("train")]
    E.require_training(train)
    with pytest.raises(ValueError, match="train-only"):
        E.require_training(panel)
    with pytest.raises(ValueError, match="train-only"):
        E.require_training(train.iloc[:0])


def test_grid_similarity_uses_timestamp_overlap_instead_of_row_order():
    clock = pd.date_range("2024-01-01", periods=5, freq="h")
    records = []
    for sid, hours, values in [("A", [0, 1, 2, 3, 4], [0., 11., 7., 20., 2.]),
                               ("B", [0, 2, 3, 4], [0., 7., 20., 2.])]:
        records.extend({"station_id": sid, "_time": clock[h], "pm2_5": v, "split": "train"}
                       for h, v in zip(hours, values))
    result = E.spatial_similarity(pd.DataFrame(records).sample(frac=1, random_state=31))
    assert result["nodes"] == 2 and result["pair_count"] == 1
    assert result["max_pair"]["n_pairs"] == 4
    assert result["max_pair"]["pearson_r"] == pytest.approx(1)
    assert result["max_pair"]["identical_on_overlap"]
    assert not result["max_pair"]["complete_series_identical"]
    assert result["exact_duplicate_pairs"] == []
    complete = pd.DataFrame(records)
    complete = complete[complete._time.ne(clock[1])]
    assert len(E.spatial_similarity(complete)["exact_duplicate_pairs"]) == 1


def test_legacy_v2_input_keeps_raw_units_and_columns(panel):
    legacy = panel.drop(columns="wind_speed_ms").rename(columns={"wind_speed_kmh": "wind_speed_10m"})
    out = F.build_features(legacy, raw_wind_unit="km/h")
    pd.testing.assert_frame_equal(out[legacy.columns], legacy)
    np.testing.assert_allclose(out.u_wind, 2, atol=1e-7)


def test_delivered_parquet_and_figure_contract():
    path = ROOT / "data/feat_spatiotemporal.parquet"
    assert path.is_dir(), "The release must contain the ordered feature dataset"
    audit = json.loads((ROOT / "models/metrics_stage2_features.json").read_text(encoding="utf-8"))
    original = pd.read_parquet(ROOT / "data/haze_panel_clean.parquet")
    out = pd.read_parquet(path)
    pd.testing.assert_frame_equal(out[original.columns], original)
    assert len(out) == audit["rows"]
    assert F.digest(path) == audit["output_sha256"]
    assert F.digest(ROOT / "data/haze_panel_clean.parquet") == audit["input_sha256"]
    assert set(out.columns) - set(original.columns) == set(audit["new_columns"])
    for c in audit["model_feature_columns"]:
        assert c in out and not c.startswith(("target_", "audit_", "b_proto_"))
        assert not np.isinf(out[c]).any()
    from PIL import Image
    slugs = ["diurnal_seasonal", "correlation_matrix", "class_imbalance", "wind_rose_pollution", "spatial_crosscorr"]
    for slug in slugs:
        im = Image.open(ROOT / f"figures/fig_eda_{slug}.png")
        assert im.size[0] >= 3000
        np.testing.assert_allclose(im.info["dpi"], [300,300], atol=.01)
