"""Sequence windows must end on day D-1 and agree with the lag features."""
import numpy as np
import pytest

from phase2.constants import MAX_WINDOW
from phase2.data import build_flight_table, labelled
from phase2.history import build_origin_daily, build_route_daily, lag_features
from phase2.sequences import origin_day_features, route_day_features
from phase2.features import weather_table
from phase2.weather import daily_summaries
from phase2.history import DailyTable


@pytest.fixture(scope="module")
def setup():
    flights = build_flight_table()
    lab = labelled(flights)
    origin, route = build_origin_daily(flights), build_route_daily(flights)
    return lab, origin, route, lag_features(lab, origin, route)


def _raw_windows(lab, origin, route, w):
    o = DailyTable("o", origin.keys, ["x"] * 6, origin_day_features(origin))
    r = DailyTable("r", route.keys, ["x"] * 4, route_day_features(route))
    d = lab["day_idx"].to_numpy()
    return (o.window(o.index_of(lab["origin"]), d, w),
            r.window(r.index_of(lab["route"]), d, w))


def test_last_step_is_yesterday(setup):
    """Unstandardised last step (D-1) delay rate equals the 1-day lag feature."""
    lab, origin, route, lags = setup
    ow, _ = _raw_windows(lab, origin, route, 14)
    last_rate = ow[:, -1, 3]
    expected = np.nan_to_num(lags["o_rate_1d"].to_numpy(), nan=0.0)
    np.testing.assert_allclose(last_rate, expected)


def test_route_window_counts_match_lags(setup):
    lab, origin, route, lags = setup
    _, rw = _raw_windows(lab, origin, route, 14)
    n_done = np.expm1(rw[:, :, 0]).sum(axis=1)
    np.testing.assert_allclose(n_done, lags["r_n_done_14d"].to_numpy(), rtol=1e-9)


def test_padding_before_january(setup):
    lab, origin, route, _ = setup
    first = lab[lab["day_idx"] == 3]
    ow, _ = _raw_windows(first, origin, route, 14)
    # day 3 sees days -11..2: 11 padded days with has_data = 0
    assert (ow[:, :11, -1] == 0).all() and (ow[:, :11] == 0).all()


def test_weather_window_alignment(setup):
    _, origin, *_ = setup
    wt = weather_table(daily_summaries(), origin.keys)
    # padding is NaN (no weather), real days 0..364 are filled
    assert np.isnan(wt.values[:, :MAX_WINDOW, 0]).all()
    assert not np.isnan(wt.values[:, MAX_WINDOW:, 0]).any()
