"""Check 26-cell EDA scope, geometry selection, and descriptive time slices."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eda26", ROOT / "src/02_eda_analysis.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


@pytest.fixture
def network():
    nodes = [("R_w", .4, "receptor", "Local"), ("R_e", .5, "receptor", "Local"),
             ("C_near", .6, "corridor", "Local"), ("C_far", 20., "corridor", "Local"),
             ("S_w", -1., "source", "West"), ("S_e", 2., "source", "East")]
    return pd.DataFrame([dict(station_id=s, lat=0., lon=lon, role=role, region=region)
                         for s, lon, role, region in nodes])


def test_geometry_selection_is_reproducible_without_pollution_columns(network):
    first = E.geometry_pair_candidates(network)
    shuffled = E.geometry_pair_candidates(network.sample(frac=1, random_state=73))
    assert first == shuffled
    assert first["candidate_count"] == 10
    selected = first["selected_pairs"]
    assert len(selected) == 8
    for destination in ("R_w", "R_e"):
        pairs = [p for p in selected if p["destination"] == destination]
        assert {p["source"] for p in pairs} == {"R_e" if destination == "R_w" else "R_w",
                                               "C_near", "S_w", "S_e"}
        assert all(p["distance_km"] > 0 for p in pairs)
    assert all(p["source"] != "C_far" for p in selected)


def test_heldout_pollution_or_targets_cannot_choose_ccf_pairs(network):
    panel = pd.concat([network.assign(split=s, pm2_5=float(i), target_pm25_3h=float(i))
                       for i, s in enumerate(("train", "val", "test"))], ignore_index=True)
    before = E.geometry_pair_candidates(panel)
    heldout = panel.split.ne("train")
    panel.loc[heldout, ["pm2_5", "target_pm25_3h"]] = 999999.
    assert E.geometry_pair_candidates(panel) == before


def test_equal_distance_selection_uses_identifier_tie_break(network):
    other = network.loc[network.station_id.eq("C_near")].assign(station_id="C_aaa")
    selected = E.geometry_pair_candidates(pd.concat([network, other]))["selected_pairs"]
    assert {p["source"] for p in selected if p["category"] == "local_corridor"} == {"C_aaa"}


def test_current_registry_covers_three_receptors_and_source_regions():
    config_spec = importlib.util.spec_from_file_location("config26", ROOT / "src/config.py")
    config = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config)
    metadata = pd.DataFrame(config.NODES).rename(columns={"id": "station_id"})
    result = E.geometry_pair_candidates(metadata)
    assert len(result["network_nodes"]) == 26
    assert result["candidate_count"] == 75
    selected = result["selected_pairs"]
    assert len(selected) == 12
    assert {p["destination"] for p in selected} == set(config.RECEPTORS)
    regional = [p for p in selected if p["category"] == "regional_source"]
    assert {p["source_region"] for p in regional} == {"Sumatra", "Borneo"}
    assert all(p["source"] != p["destination"] for p in selected)


def test_class_split_and_year_changes_preserve_four_ordered_levels(monkeypatch):
    frame = pd.DataFrame({"station_id": ["R"]*4 + ["S"],
                          "role": ["receptor"]*4 + ["source"],
                          "_time": pd.to_datetime(["2023-01-01", "2023-02-01", "2024-01-01", "2024-02-01", "2024-02-01"]),
                          "split": ["train", "train", "val", "test", "test"],
                          "target_level_3h": [0, 1, 2, 3, 0]})
    monkeypatch.setattr(E, "save", lambda fig, *_: E.plt.close(fig))
    result = E.class_distribution(frame, Path("."))
    assert result["receptor_by_split"]["train"]["counts"] == {"0": 1, "1": 1, "2": 0, "3": 0}
    assert result["receptor_by_split"]["test"]["proportions"]["3"] == 1.
    assert result["receptor_by_year"]["2024"]["n_labeled"] == 2
    assert result["node_counts"]["S"]["0"] == 1
    assert "not official hourly AQI" in result["risk_definition"]


def test_empty_interval_does_not_fabricate_class_proportions():
    frame = pd.DataFrame({"target_level_3h": pd.Series(dtype="float")})
    summary = E.risk_summary(frame)
    assert summary["n_labeled"] == 0
    assert list(summary["counts"].values()) == [0]*4
    assert list(summary["proportions"].values()) == [None]*4


def test_wind_sw_calendar_and_direction_are_distinct_and_qc_filtered(monkeypatch):
    frame = pd.DataFrame({"station_id": ["R"]*5, "role": ["receptor"]*5,
                          "split": ["train"]*5,
                          "_time": pd.to_datetime(["2024-01-01", "2024-06-01", "2024-07-01", "2024-08-01", "2024-09-01"]),
                          "wind_direction_10m": [225., 225., 90., 225., 225.],
                          "wind_speed_ms": [2., 2., 2., 0., 2.],
                          "pm2_5": [10., 20., 60., 999., 900.], "qc_interpolated": [0, 0, 0, 0, 1]})
    monkeypatch.setattr(E, "save", lambda fig, *_: E.plt.close(fig))
    result = E.wind_rose(frame, Path("."))
    assert result["n_noncalm"] == 3
    assert result["n_excluded_calm_or_missing"] == 2
    assert result["sw_flow_count"] == 2
    sw = result["sw_monsoon"]
    assert sw["n_noncalm"] == 2 and sw["sw_flow_count"] == 1
    assert sw["sw_flow_mean_pm25"] == 20.
    assert sw["sector_mean_pm25"][10] == 20.  # 225 degrees
    assert sw["sector_mean_pm25"][4] == 60.   # 90 degrees
    assert sum(sw["sector_frequency_percent"]) == pytest.approx(100)
    assert result["by_node"]["R"]["sw_monsoon"] == sw


def test_climatology_fit_is_invariant_to_test_concentration():
    frame = pd.DataFrame({"station_id": ["R"]*3, "split": ["train", "train", "test"],
                          "_month": [6]*3, "_hour": [0]*3, "pm2_5": [10., 30., 50.]})
    before = E.deseasonalized(frame)
    frame.loc[2, "pm2_5"] = 9000.
    after = E.deseasonalized(frame)
    np.testing.assert_allclose(before.iloc[:2], after.iloc[:2])
    assert before.iloc[2] == 30. and after.iloc[2] == 8980.
