"""Check 26-cell EDA scope, geometry selection, and descriptive time slices."""
import importlib.util
import json
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


def test_extended_training_lags_find_a_known_regional_lead_across_a_gap():
    clock = pd.date_range("2025-06-01", periods=120, freq="h")
    values = np.random.default_rng(58).normal(size=len(clock))
    source = pd.DataFrame({"station_id": "S", "_time": clock, "pm2_5": values})
    destination = pd.DataFrame({"station_id": "R", "_time": clock,
                                "pm2_5": pd.Series(values).shift(24)})
    frame = pd.concat([source.drop(index=30), destination], ignore_index=True)
    frame = frame.assign(split="train", _month=frame._time.dt.month, _hour=frame._time.dt.hour)
    pairs = [{"source": "S", "destination": "R", "category": "regional_source"}]
    result = E.extended_training_peaks(frame, pairs)
    index, series, best, in_band = result["records"][0]
    assert index == 0 and series == "raw"
    assert best == in_band
    lag, correlation, sample_count = best
    assert lag == 24 and correlation == pytest.approx(1) and sample_count == 95
    frame.loc[frame.index[-1], "split"] = "test"
    with pytest.raises(ValueError, match="train-only"):
        E.extended_training_peaks(frame, pairs)


def test_published_metrics_fit_budget_and_retain_report_evidence():
    artifact = ROOT / "models/metrics_stage2_eda.json"
    assert 0 < artifact.stat().st_size <= 100_000
    metrics = json.loads(artifact.read_text(encoding="utf-8"))
    assert metrics["format_version"] == 2
    assert metrics["diurnal_seasonal"]["seasons"]["SW: Jun–Sep"]["median"] == 26.6
    assert metrics["correlations"]["pearson"]["target_pm25_3h"]["pm2_5"] == pytest.approx(.73704, abs=1e-5)
    assert metrics["class_distribution"]["receptor_by_split"]["test"]["counts"]["3"] == 3154

    wind = metrics["wind_rose"]
    assert wind["sw_monsoon"]["n_noncalm"] == 24_528
    assert wind["sw_monsoon"]["sw_flow_fraction"] == pytest.approx(.10148, abs=1e-5)
    assert len(wind["by_node"]) == 26
    for periods in wind["by_node"].values():
        assert set(periods) == {"all_months", "sw_monsoon"}
        for stats in periods.values():
            counts = np.array(stats["speed_direction_counts"])
            assert counts.shape == (16, 4) and counts.sum() == stats["n_noncalm"]
            assert len(stats["sector_mean_pm25"]) == 16

    crosscorr = metrics["crosscorr"]
    selection = crosscorr["pair_selection"]
    assert len(selection["candidates"]) == selection["candidate_count"] == 75
    assert len(selection["selected_pairs"]) == 12
    lags = crosscorr["lag_hours"]
    assert lags == list(range(-12, 13))
    assert len(crosscorr["records"]) == 24
    for record in crosscorr["records"]:
        points = record["points"]
        assert len(points["r"]) == len(points["n_pairs"]) == len(lags)
        peak = record["peak"]
        i = lags.index(peak["lag_h"])
        assert points["r"][i] == peak["r"] and points["n_pairs"][i] == peak["n_pairs"]

    extended = crosscorr["extended_training_peaks"]
    assert extended["lag_range_hours"] == [-48, 48]
    assert extended["candidate_bands_hours"] == {"local": [2, 4], "regional": [18, 36]}
    assert {(i, series) for i, series, *_ in extended["records"]} == {
        (i, series) for i in range(12) for series in ("raw", "deseasonalized")}
    for i, _, maximum, band_peak in extended["records"]:
        group = "regional" if selection["selected_pairs"][i]["category"] == "regional_source" else "local"
        lower, upper = extended["candidate_bands_hours"][group]
        assert -48 <= maximum[0] <= 48 and lower <= band_peak[0] <= upper
        assert maximum[1] >= band_peak[1] and maximum[2] > 25_000 and band_peak[2] > 25_000

    similarity = metrics["spatial_similarity"]
    assert similarity["pair_count"] == 325
    assert similarity["max_pair"]["pearson_r"] == pytest.approx(.9086)
    assert similarity["max_pair"]["n_pairs"] == 26_034
    assert similarity["identical_on_overlap_pairs"] == []
    assert len(metrics["figure_sha256"]) == 5
