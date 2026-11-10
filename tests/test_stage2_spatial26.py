"""Spatial design diagnostics. Small synthetic fixtures never become project data."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("spatial26_features", ROOT / "src/02_spatiotemporal_features.py")
F = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(F)


@pytest.fixture
def train_panel():
    records = []
    for j, node in enumerate(("West", "Center", "East")):
        for hour in range(5):
            records.append({"station_id": node, "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=hour),
                            "split": "train", "lat": 0., "lon": .1 * j,
                            "pm2_5": 10. * (j + 1) + hour, "pm10": 30. + hour,
                            "no2": 5., "relative_humidity_2m": 60.,
                            "wind_speed_ms": 2., "wind_direction_10m": 270.,
                            "qc_interpolated": 0, "is_missing": 0, "qc_calm_wind": 0})
    return pd.DataFrame(records)


def test_fixed_sigma_geometry_and_concentration_are_physically_consistent(train_panel):
    audit = F.spatial_training_diagnostics(train_panel, sigma_km=15, radius_km=None)
    assert audit["node_count"] == 3 and audit["rows"] == 15
    assert audit["sigma_km"] == 15 and audit["coordinate_pair_count"] == 3
    assert audit["coordinate_pairs_within_sigma"] == 2
    assert audit["coordinate_pairs_within_3sigma"] == 3
    assert audit["duplicate_coordinate_pairs"] == []
    assert audit["nearest_neighbor_distance_km"]["median"] == pytest.approx(11.119508, rel=1e-6)
    west, center, east = (audit["by_node"][node] for node in ("West", "Center", "East"))
    assert west["zero_weight_fraction"] == 1
    assert west["effective_neighbor_count_when_available"]["median"] is None
    assert center["zero_weight_fraction"] == 0
    assert center["max_normalized_weight_when_available"]["median"] == 1
    assert center["effective_neighbor_count_when_available"]["median"] == 1
    distance = np.array([22.239016, 11.119508])
    weight = 2. * np.exp(-.5 * (distance / 15.) ** 2)
    probability = weight / weight.sum()
    assert east["positive_weight_neighbor_count"]["median"] == 2
    assert east["max_normalized_weight_when_available"]["median"] == pytest.approx(probability.max(), rel=1e-6)
    assert east["effective_neighbor_count_when_available"]["median"] == pytest.approx(1 / np.sum(probability ** 2), rel=1e-6)
    assert audit["overall"]["zero_weight_fraction"] == pytest.approx(1 / 3)
    json.dumps(audit, allow_nan=False)


def test_training_only_guard_and_labels_do_not_influence_diagnostics(train_panel):
    baseline = F.spatial_training_diagnostics(train_panel)
    with_labels = train_panel.assign(target_pm25_3h=np.inf, target_level_3h=-999)
    assert F.spatial_training_diagnostics(with_labels) == baseline
    heldout = train_panel.copy()
    heldout.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="train-only"):
        F.spatial_training_diagnostics(heldout)
    with pytest.raises(ValueError, match="train-only"):
        F.spatial_training_diagnostics(train_panel.iloc[:0])
    for bad in (0., -15., np.nan, np.inf):
        with pytest.raises(ValueError, match="positive and finite"):
            F.spatial_training_diagnostics(train_panel, sigma_km=bad)


def test_coordinate_duplicates_are_exposed_without_claiming_model_cell_centers(train_panel):
    twin = train_panel.loc[train_panel.station_id.eq("Center")].assign(station_id="Twin")
    duplicate = pd.concat([train_panel, twin], ignore_index=True)
    audit = F.spatial_training_diagnostics(duplicate)
    assert audit["duplicate_coordinate_pairs"] == [{"node_a": "Center", "node_b": "Twin"}]
    assert audit["nearest_neighbor_by_node"]["Center"]["nearest_km"] == 0
    assert audit["nearest_neighbor_by_node"]["Center"]["nearest_distinct_km"] > 11
    assert "not verified" in audit["coordinate_basis"]
    # The co-located source is excluded for the target at that coordinate.
    assert audit["by_node"]["Center"]["positive_weight_neighbor_count"]["median"] == 1


def test_missing_direction_cannot_supply_advection(train_panel):
    train_panel.loc[train_panel.station_id.eq("West"), "wind_direction_10m"] = np.nan
    out = F.build_features(train_panel)
    assert out.loc[out.station_id.eq("West"), ["u_wind", "v_wind"]].isna().all().all()
    assert out.loc[out.station_id.eq("Center"), "upwind_pm25_advection_1h"].isna().all()
    audit = F.spatial_training_diagnostics(train_panel)
    assert audit["source_quality"]["missing_wind_direction_rows"] == 5
    assert audit["source_quality"]["unavailable_wind_vector_rows"] == 5
    assert audit["by_node"]["Center"]["zero_weight_fraction"] == 1


def test_calm_flag_does_not_invent_missing_speed_but_known_calm_needs_no_direction(train_panel):
    west = train_panel.station_id.eq("West")
    train_panel.loc[west, "qc_calm_wind"] = 1
    train_panel.loc[west, "wind_speed_ms"] = np.nan
    u, v = F.wind_vectors(train_panel)
    assert u[west].isna().all() and v[west].isna().all()
    train_panel.loc[west, "wind_speed_ms"] = 0.
    train_panel.loc[west, "wind_direction_10m"] = np.nan
    u, v = F.wind_vectors(train_panel)
    assert u[west].eq(0).all() and v[west].eq(0).all()
    assert F.spatial_training_diagnostics(train_panel)["by_node"]["Center"]["zero_weight_fraction"] == 1


def test_diagnostics_distinguish_missing_history_from_zero_weight_and_exact_hour_gaps(train_panel):
    hole = train_panel.station_id.eq("West") & train_panel.timestamp.eq(pd.Timestamp("2024-01-01 01:00"))
    sparse = train_panel.loc[~hole].sample(frac=1, random_state=42)
    audit = F.spatial_training_diagnostics(sparse)["by_node"]["Center"]
    assert audit["zero_weight_rows"] == 1
    lag = audit["advection_availability_by_lag_h"]["1"]
    assert lag == {"positive_weight_rows": 3, "zero_weight_rows": 1,
                   "missing_history_rows": 1, "missing_source_input_rows": 0, "coverage_fraction": .6}
    out = F.build_features(sparse)
    issue = out.loc[out.station_id.eq("Center") & out.timestamp.eq(pd.Timestamp("2024-01-01 02:00"))].iloc[0]
    assert pd.isna(issue.upwind_pm25_advection_1h)
    assert issue.upwind_pm25_advection_2h == 10.


def test_reversed_wind_reverses_usable_neighbors_and_features_keep_contract(train_panel):
    base = F.build_features(train_panel)
    forward = F.spatial_training_diagnostics(train_panel)
    reversed_panel = train_panel.assign(wind_direction_10m=90.)
    backward = F.spatial_training_diagnostics(reversed_panel)
    assert forward["by_node"]["West"]["zero_weight_fraction"] == 1
    assert backward["by_node"]["East"]["zero_weight_fraction"] == 1
    assert backward["by_node"]["West"]["zero_weight_fraction"] == 0
    assert base.shape[1] - train_panel.shape[1] == 63
    assert set(F.NEAR_COLUMNS + F.FAR_COLUMNS).issubset(base)
    pd.testing.assert_frame_equal(base[train_panel.columns], train_panel)


def test_single_node_has_no_neighbors_or_invented_distances(train_panel):
    single = train_panel.loc[train_panel.station_id.eq("Center")]
    audit = F.spatial_training_diagnostics(single)
    assert audit["coordinate_pair_count"] == 0
    assert audit["pair_distance_km"]["median"] is None
    assert audit["nearest_neighbor_distance_km"]["median"] is None
    assert audit["overall"]["zero_weight_fraction"] == 1
    assert audit["by_node"]["Center"]["all_other_source_observations_unavailable_rows"] == len(single)
    json.dumps(audit, allow_nan=False)


def test_default_path_remains_the_confirmed_panel_when_file_is_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "ROOT", tmp_path)
    assert F.default_input() == tmp_path / "data/haze_panel_clean.parquet"
