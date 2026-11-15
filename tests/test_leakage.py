"""Leakage and correctness checks for Phase 2 features.

Run with:  .venv/bin/python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from phase2.constants import CUTOFF_HOUR, FORBIDDEN_INPUTS, P1_CATEGORICAL, P1_NUMERIC
from phase2.data import build_flight_table, labelled
from phase2.history import build_origin_daily, build_route_daily, lag_features
from phase2.weather import DAILY_WEATHER_COLS, FORECAST_COLS, load_hourly, daily_summaries


@pytest.fixture(scope="module")
def tables():
    flights = build_flight_table()
    lab = labelled(flights)
    origin, route = build_origin_daily(flights), build_route_daily(flights)
    lags = lag_features(lab, origin, route)
    return flights, lab, origin, route, lags


def _cutoff(date):
    return date - pd.Timedelta(hours=24 - CUTOFF_HOUR)  # 18:00 on D-1


def test_phase1_split_sizes_match(tables):
    _, lab, *_ = tables
    assert lab["split"].value_counts().to_dict() == {
        "train": 217_727, "validation": 55_628, "test": 53_991}


def test_no_forbidden_inputs_in_feature_lists():
    from phase2.history import LAG_FEATURES
    for cols in (P1_CATEGORICAL + P1_NUMERIC, LAG_FEATURES, DAILY_WEATHER_COLS):
        assert FORBIDDEN_INPUTS.isdisjoint(cols)
    assert set(FORECAST_COLS).isdisjoint(DAILY_WEATHER_COLS)


def test_outcomes_known_only_after_landing(tables):
    flights, *_ = tables
    arrived = flights[flights["status"].eq("arrived")]
    dep = arrived["sched_dep"] + pd.to_timedelta(arrived["dep_delay"], unit="min")
    assert (arrived["known_time"] >= dep + pd.to_timedelta(arrived["air_time"], unit="min")).all()


@pytest.mark.parametrize("w", [1, 7, 14])
def test_origin_lags_match_brute_force(tables, w):
    flights, lab, _, _, lags = tables
    rng = np.random.default_rng(0)
    sample = lab.iloc[rng.choice(len(lab), 300, replace=False)]
    arrived = flights[flights["status"].eq("arrived")]
    for idx, row in sample.iterrows():
        cut = _cutoff(row["date"])
        past = arrived[(arrived["origin"] == row["origin"])
                       & (arrived["known_time"] < cut)
                       & (arrived["known_time"] >= cut - pd.Timedelta(days=w))]
        expected = past["y"].mean() if len(past) else np.nan
        got = lags.at[idx, f"o_rate_{w}d"]
        assert (np.isnan(expected) and np.isnan(got)) or got == pytest.approx(expected), idx


def test_route_lags_match_brute_force(tables):
    flights, lab, _, _, lags = tables
    rng = np.random.default_rng(1)
    sample = lab.iloc[rng.choice(len(lab), 300, replace=False)]
    arrived = flights[flights["status"].eq("arrived")]
    for idx, row in sample.iterrows():
        cut = _cutoff(row["date"])
        past = arrived[(arrived["route"] == row["route"])
                       & (arrived["known_time"] < cut)
                       & (arrived["known_time"] >= cut - pd.Timedelta(days=14))]
        assert lags.at[idx, "r_n_done_14d"] == len(past)


def test_history_never_uses_post_cutoff_outcomes(tables):
    """Strong check: zeroing every outcome known at/after the cutoff for day D
    must leave day-D features unchanged."""
    flights, lab, *_ = tables
    day = 200
    rows = lab[lab["day_idx"] == day]
    cut = _cutoff(rows["date"].iloc[0])
    pruned = flights[~(flights["known_time"] >= cut)]  # keep only pre-cutoff outcomes
    pruned = pruned[pd.to_datetime(pruned["sched_dep"]) < cut + pd.Timedelta(days=30)]
    a = lag_features(rows, build_origin_daily(flights), build_route_daily(flights))
    b = lag_features(rows, build_origin_daily(pruned), build_route_daily(pruned))
    rate_cols = [c for c in a.columns if "cancel" not in c]
    pd.testing.assert_frame_equal(a[rate_cols], b[rate_cols])


def test_weather_days_end_at_cutoff():
    h = load_hourly()
    d = daily_summaries(h)
    from phase2.data import obs_day
    h = h.assign(k=obs_day(h["ts"]))
    start = pd.Timestamp("2013-01-01") + pd.to_timedelta(h["k"], unit="D")
    # observation day k covers [18:00 on k-1, 18:00 on k)
    assert (h["ts"] < start + pd.Timedelta(hours=CUTOFF_HOUR)).all()
    assert (h["ts"] >= start - pd.Timedelta(hours=24 - CUTOFF_HOUR)).all()
    assert d.groupby("origin").size().to_dict() == {"EWR": 365, "JFK": 365, "LGA": 365}
