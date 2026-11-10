"""Exact-hour and physical checks for the two spatial layers."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("spatial_layers", ROOT / "src/02_spatiotemporal_features.py")
F = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(F)
START = pd.Timestamp("2024-01-01")


def node(station, distance, region, offset, *, speed=2., direction=270., periods=80):
    hours = np.arange(periods)
    return pd.DataFrame({"station_id": station, "timestamp": pd.date_range(START, periods=periods, freq="h"),
                         "split": "train", "lat": 0., "lon": np.rad2deg(distance / 6371.0088),
                         "region": region, "pm2_5": offset + hours, "pm10": offset + hours + 10.,
                         "no2": 5., "relative_humidity_2m": 70., "wind_speed_ms": speed,
                         "wind_direction_10m": direction, "qc_interpolated": 0, "is_missing": 0,
                         "qc_calm_wind": 0})


def panel(*nodes):
    return pd.concat(nodes, ignore_index=True)


def at(frame, station, hour):
    return frame.loc[frame.station_id.eq(station) & frame.timestamp.eq(START + pd.Timedelta(hours=hour))].iloc[0]


def test_near_radius_150_km_and_gaussian_60_km():
    data = panel(node("Target", 0, "KL", 10.), node("Near", -30, "Peninsula", 20.),
                 node("Boundary", -150, "Peninsula", 80.), node("Outside", -151, "Peninsula", 1000.))
    row = at(F.build_features(data), "Target", 30)
    weights = 2 * np.exp(-.5 * (np.array([30., 150.]) / 60.) ** 2)
    expected = np.average([49., 109.], weights=weights)
    assert row.upwind_weight_sum_1h == pytest.approx(weights.sum(), rel=1e-6)
    assert row.upwind_neighbor_count_1h == 2
    assert row.upwind_pm25_advection_1h == pytest.approx(expected, rel=1e-6)
    diagnosis = F.spatial_training_diagnostics(data)
    assert diagnosis["radius_km"] == 150 and diagnosis["sigma_km"] == 60
    assert diagnosis["by_node"]["Target"]["geometric_neighbors"] == ["Boundary", "Near"]


def test_offshore_background_does_not_enter_domestic_near_field():
    data = panel(node("Target", 0, "KL", 10.), node("Domestic", -30, "Peninsula", 20.),
                 node("Offshore", -100, "Sumatra", 1000.))
    row = at(F.build_features(data), "Target", 35)
    assert row.upwind_neighbor_count_1h == 1
    assert row.upwind_pm25_advection_1h == 54.
    assert row.far_source_pm25_lag_24h == 1011.


def test_wind_reversal_changes_the_source_not_the_target_clock():
    data = panel(node("West", -30, "Peninsula", 20.), node("Target", 0, "KL", 10.),
                 node("East", 30, "Peninsula", 200.))
    west_wind = at(F.build_features(data), "Target", 30)
    east_wind = at(F.build_features(data.assign(wind_direction_10m=90.)), "Target", 30)
    assert west_wind.upwind_pm25_advection_1h == 49.
    assert east_wind.upwind_pm25_advection_1h == 229.
    assert west_wind.upwind_neighbor_count_1h == east_wind.upwind_neighbor_count_1h == 1
    assert west_wind.spatial_pm25_gradient == -10.
    assert east_wind.spatial_pm25_gradient == -190.


def test_calm_wind_is_zero_support_while_unavailable_inputs_are_null():
    target = node("Target", 0, "KL", 10.)
    calm = panel(target, node("Near", -30, "Peninsula", 20., speed=.1, direction=np.nan))
    missing = panel(target, node("Near", -30, "Peninsula", 20., direction=np.nan))
    for data, cause, expected in [(calm, "calm_sources_only", 0.), (missing, "missing_source_inputs", np.nan)]:
        row = at(F.build_features(data), "Target", 30)
        assert pd.isna(row.upwind_pm25_advection_1h)
        if np.isnan(expected):
            assert pd.isna(row.upwind_weight_sum_1h) and pd.isna(row.upwind_neighbor_count_1h)
        else:
            assert row.upwind_weight_sum_1h == row.upwind_neighbor_count_1h == expected
        diagnosed = F.spatial_training_diagnostics(data)["by_node"]["Target"]
        assert diagnosed["support_reason_counts"][cause] == len(target)
        assert diagnosed["zero_weight_fraction"] == 1.


def test_near_lags_take_each_hours_own_pm25_and_wind_and_keep_the_hole():
    source = node("Near", -30, "Peninsula", 20.)
    # At target issue 30h, reverse wind at t-2 and omit t-3 entirely.
    source.loc[source.index == 28, "wind_direction_10m"] = 90.
    data = panel(node("Target", 0, "KL", 10.), source.drop(index=27))
    row = at(F.build_features(data.sample(frac=1, random_state=4)), "Target", 30)
    assert row.upwind_pm25_advection_1h == 49.
    assert pd.isna(row.upwind_pm25_advection_2h) and row.upwind_weight_sum_2h == 0.
    assert pd.isna(row.upwind_pm25_advection_3h) and pd.isna(row.upwind_weight_sum_3h)
    # An ordinary 3h lag reads t-3, which need not equal either shorter lag.
    following = at(F.build_features(data), "Target", 32)
    assert following.upwind_pm25_advection_3h == 49.


def test_far_field_windows_exclude_issue_hour_and_use_exact_pooled_history():
    data = panel(node("Target", 0, "KL", 10000.), node("Sum", -300, "Sumatra", 20.),
                 node("Bor", 500, "Borneo", 60.), node("Local", 70, "Peninsula", 10000.))
    out = F.build_features(data)
    row = at(out, "Target", 60)
    for window in (12, 24, 48):
        assert row[f"far_source_pm25_mean_{window}h"] == pytest.approx(40 + np.arange(60-window, 60).mean())
    assert row.far_source_pm25_lag_24h == 76.
    changed = data.copy()
    changed.loc[changed.timestamp.ge(START + pd.Timedelta(hours=60)), "pm2_5"] = 99999.
    pd.testing.assert_series_equal(at(F.build_features(changed), "Target", 60)[F.FAR_COLUMNS], row[F.FAR_COLUMNS])


def test_near_and_far_history_restart_at_split_and_qc_gaps_stay_missing():
    data = panel(node("Target", 0, "KL", 10.), node("Near", -30, "Peninsula", 20.),
                 node("Sum", -300, "Sumatra", 30.), node("Bor", 500, "Borneo", 70.))
    data.loc[data.timestamp.ge(START + pd.Timedelta(hours=48)), "split"] = "val"
    out = F.build_features(data)
    assert pd.isna(at(out, "Target", 48).upwind_pm25_advection_1h)
    assert pd.isna(at(out, "Target", 49).upwind_pm25_advection_2h)
    assert pd.isna(at(out, "Target", 50).upwind_pm25_advection_3h)
    assert at(out, "Target", 51).upwind_pm25_advection_3h == 68.
    row = at(out, "Target", 60)
    assert row.far_source_pm25_mean_12h == 103.5
    assert row[["far_source_pm25_mean_24h", "far_source_pm25_mean_48h", "far_source_pm25_lag_24h"]].isna().all()
    data.loc[data.region.isin(F.FAR_REGIONS) & data.timestamp.eq(START + pd.Timedelta(hours=50)), "qc_interpolated"] = 1
    assert pd.isna(at(F.build_features(data), "Target", 60).far_source_pm25_mean_12h)


def test_future_perturbation_does_not_change_the_new_spatial_predictors():
    data = panel(node("Target", 0, "KL", 10.), node("Near", -30, "Peninsula", 20.),
                 node("Sum", -300, "Sumatra", 30.), node("Bor", 500, "Borneo", 70.))
    cutoff = START + pd.Timedelta(hours=60)
    future = data.timestamp.gt(cutoff)
    changed = data.copy()
    changed.loc[future, ["pm2_5", "wind_speed_ms", "wind_direction_10m"]] = [9999., 3.5, 90.]
    columns = F.NEAR_COLUMNS + F.FAR_COLUMNS
    pd.testing.assert_frame_equal(F.build_features(data).loc[~future, columns],
                                  F.build_features(changed).loc[~future, columns])


def test_legacy_and_near_support_keep_same_denominator_and_explain_lost_sources():
    data = panel(node("Target", 0, "KL", 10.), node("Near", -130, "Peninsula", 20.),
                 node("Unconnected", 1000, "Peninsula", 30.))
    near = F.spatial_training_diagnostics(data)
    old = F.spatial_training_diagnostics(data, 15., radius_km=None)
    assert near["overall"]["rows"] == old["overall"]["rows"] == len(data)
    assert old["by_node"]["Target"]["support_reason_counts"]["below_weight_floor"] == 80
    assert near["by_node"]["Target"]["support_reason_counts"]["supported"] == 80
    assert near["by_node"]["Unconnected"]["support_reason_counts"]["no_geometric_neighbors"] == 80
    assert near["overall"]["zero_weight_fraction"] < old["overall"]["zero_weight_fraction"]
    summary = F.combine_spatial_support({"train": near})
    assert summary["overall"]["rows"] == sum(x["rows"] for x in summary["by_target_wind_speed"].values())


def test_layer_contract_preserves_input_and_marks_unavailable_far_sources():
    data = panel(node("Target", 0, "KL", 10.), node("Near", -30, "Peninsula", 20.))
    out = F.build_features(data)
    pd.testing.assert_frame_equal(out[data.columns], data, check_exact=True)
    assert len(set(out) - set(data)) == 63
    assert set(F.AUDIT_COLUMNS) <= set(out)
    assert out[F.FAR_COLUMNS].isna().all().all()
    assert not out.duplicated(["station_id", "timestamp"]).any()
    assert not any(name.startswith("audit_") for name in F.NEAR_COLUMNS + F.FAR_COLUMNS)
    with pytest.raises(ValueError, match="duplicate"):
        F.build_features(pd.concat([data, data.iloc[[0]]], ignore_index=True))
    invalid = data.copy()
    invalid.loc[0, "pm2_5"] = np.inf
    with pytest.raises(ValueError, match="finite or missing"):
        F.build_features(invalid)
