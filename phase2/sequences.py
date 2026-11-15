"""Turn daily history tables into per-flight sequence tensors.

Each flight on day d gets, per branch, a [W, F] window of observation days
d-W .. d-1 (all known by 18:00 on d-1). Day-level inputs:

  origin: log1p(scheduled), cancellation share, log1p(completed),
          delay rate, mean arrival delay, has-data flag
          (+ daily weather columns from Stage 3 onwards)
  route:  log1p(completed), delay rate, mean arrival delay, has-data flag

Rates/means are 0 when a day has no completed flights; the has-data flag
tells the model those zeros mean "no information", and also marks the
left padding before 1 January.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import MAX_WINDOW
from .history import DailyTable

TRAIN_LAST_DAY = 242  # 31 Aug 2013; normalisation statistics use days <= this

ORIGIN_SEQ_COLS = ["log_sched", "cancel_share", "log_done", "delay_rate", "mean_delay", "has_data"]
ROUTE_SEQ_COLS = ["log_done", "delay_rate", "mean_delay", "has_data"]


def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b > 0)


def origin_day_features(t: DailyTable) -> np.ndarray:
    c = {k: t.values[..., i] for i, k in enumerate(t.columns)}
    has = (c["n_done"] > 0).astype(float)
    return np.stack([
        np.log1p(c["n_sched"]), _safe_div(c["n_cancel"], c["n_sched"]),
        np.log1p(c["n_done"]), _safe_div(c["n_delayed"], c["n_done"]),
        _safe_div(c["sum_arr_delay"], c["n_done"]), has,
    ], axis=-1)


def route_day_features(t: DailyTable) -> np.ndarray:
    c = {k: t.values[..., i] for i, k in enumerate(t.columns)}
    has = (c["n_done"] > 0).astype(float)
    return np.stack([
        np.log1p(c["n_done"]), _safe_div(c["n_delayed"], c["n_done"]),
        _safe_div(c["sum_arr_delay"], c["n_done"]), has,
    ], axis=-1)


def standardise(x: np.ndarray, skip_last: int = 1) -> tuple[np.ndarray, dict]:
    """Standardise each day feature with statistics from training-period days
    only (real days 0..TRAIN_LAST_DAY, padding excluded). NaNs (missing
    weather) become 0 after scaling, i.e. the training mean. The trailing
    `skip_last` columns (flags) are left as they are."""
    train = x[:, MAX_WINDOW:MAX_WINDOW + TRAIN_LAST_DAY + 1].reshape(-1, x.shape[-1])
    n = x.shape[-1] - skip_last
    mean = np.nanmean(train[:, :n], axis=0)
    std = np.nanstd(train[:, :n], axis=0)
    std[std == 0] = 1.0
    out = x.copy()
    out[..., :n] = (x[..., :n] - mean) / std
    out = np.nan_to_num(out, nan=0.0)
    out[:, :MAX_WINDOW] = 0.0  # padding stays exactly zero (has_data = 0)
    return out.astype(np.float32), {"mean": mean, "std": std}


def build_sequences(flights: pd.DataFrame, tables: dict[str, DailyTable], window: int,
                    use_weather: bool = False) -> dict[str, np.ndarray]:
    """Return {'origin': [n, W, F_o], 'route': [n, W, F_r]} for `flights` rows."""
    o_raw = origin_day_features(tables["origin"])
    cols = list(ORIGIN_SEQ_COLS)
    if use_weather:
        w = tables["weather"]
        w_obs = (~np.isnan(w.values[..., 0])).astype(float)[..., None]
        # weather columns go before the flags so they are standardised
        o_raw = np.concatenate([o_raw[..., :-1], w.values, w_obs, o_raw[..., -1:]], axis=-1)
        cols = cols[:-1] + list(w.columns) + ["weather_obs", "has_data"]
        o_std, _ = standardise(o_raw, skip_last=2)
    else:
        o_std, _ = standardise(o_raw)
    r_std, _ = standardise(route_day_features(tables["route"]))

    day = flights["day_idx"].to_numpy()
    o_tab = DailyTable("origin_seq", tables["origin"].keys, cols, o_std)
    r_tab = DailyTable("route_seq", tables["route"].keys, ROUTE_SEQ_COLS, r_std)
    return {
        "origin": o_tab.window(o_tab.index_of(flights["origin"]), day, window),
        "route": r_tab.window(r_tab.index_of(flights["route"]), day, window),
        "origin_cols": cols,
    }
