"""Delay-history features for the origin airport and route levels.

Each past flight contributes to the observation day in which its outcome
became known (`known_obs_day`, rules R5/R6). A flight on calendar day d may
only use observation days d-W .. d-1, all of which end by 18:00 on d-1.

Daily tables are stored as dense arrays of shape [n_entities, n_days, F] with
MAX_WINDOW days of zero padding on the left, so a window for day d is
`arr[e, d : d + W]` in padded coordinates (covering original days d-W .. d-1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import MAX_WINDOW, N_DAYS
from .data import obs_day

ARR_DELAY_CLIP = (-60.0, 300.0)

# Raw daily sums (additive, so windows can be pooled exactly).
ORIGIN_SUMS = ["n_sched", "n_cancel", "n_done", "n_delayed", "sum_arr_delay"]
ROUTE_SUMS = ["n_done", "n_delayed", "sum_arr_delay"]


@dataclass
class DailyTable:
    """Dense daily sums for one entity level."""
    level: str
    keys: list[str]            # entity names, index = entity id
    columns: list[str]
    values: np.ndarray         # [n_entities, MAX_WINDOW + N_DAYS, F], left-padded

    def index_of(self, names: pd.Series) -> np.ndarray:
        lookup = {k: i for i, k in enumerate(self.keys)}
        return names.map(lookup).to_numpy()

    def window(self, ent: np.ndarray, day: np.ndarray, w: int) -> np.ndarray:
        """[n, w, F] window covering days day-w .. day-1 for each row."""
        if w > MAX_WINDOW:
            raise ValueError(f"window {w} > MAX_WINDOW {MAX_WINDOW}")
        start = day + MAX_WINDOW - w
        offs = start[:, None] + np.arange(w)[None, :]
        return self.values[ent[:, None], offs]

    def window_sum(self, ent: np.ndarray, day: np.ndarray, w: int) -> np.ndarray:
        return self.window(ent, day, w).sum(axis=1)


def _dense(level: str, frame: pd.DataFrame, key: str, cols: list[str]) -> DailyTable:
    keys = sorted(frame[key].dropna().unique())
    lookup = {k: i for i, k in enumerate(keys)}
    vals = np.zeros((len(keys), MAX_WINDOW + N_DAYS, len(cols)), dtype=np.float64)
    f = frame[frame["day"].between(0, N_DAYS - 1)]
    e = f[key].map(lookup).to_numpy()
    d = f["day"].to_numpy() + MAX_WINDOW
    vals[e, d] = f[cols].to_numpy(dtype=np.float64)
    return DailyTable(level, keys, cols, vals)


def build_origin_daily(flights: pd.DataFrame) -> DailyTable:
    sched = (flights.assign(day=obs_day(flights["sched_dep"]))
             .groupby(["origin", "day"]).size().rename("n_sched"))
    known = flights[flights["known_obs_day"].notna()].assign(
        day=lambda x: x["known_obs_day"].astype(int),
        cancel=lambda x: x["status"].eq("cancelled").astype(int),
        done=lambda x: x["status"].eq("arrived").astype(int),
        delayed=lambda x: x["y"].fillna(0).astype(int),
        arr=lambda x: x["arr_delay"].clip(*ARR_DELAY_CLIP).fillna(0.0),
    )
    g = known.groupby(["origin", "day"])
    sums = pd.DataFrame({
        "n_cancel": g["cancel"].sum(), "n_done": g["done"].sum(),
        "n_delayed": g["delayed"].sum(), "sum_arr_delay": g["arr"].sum(),
    })
    frame = pd.concat([sched, sums], axis=1).fillna(0).reset_index()
    return _dense("origin", frame, "origin", ORIGIN_SUMS)


def build_route_daily(flights: pd.DataFrame) -> DailyTable:
    known = flights[flights["status"].eq("arrived")].assign(
        day=lambda x: x["known_obs_day"].astype(int),
        delayed=lambda x: x["y"].astype(int),
        arr=lambda x: x["arr_delay"].clip(*ARR_DELAY_CLIP),
    )
    g = known.groupby(["route", "day"])
    frame = pd.DataFrame({
        "n_done": g.size(), "n_delayed": g["delayed"].sum(), "sum_arr_delay": g["arr"].sum(),
    }).reset_index()
    return _dense("route", frame, "route", ROUTE_SUMS)


def _ratio(num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def lag_features(flights: pd.DataFrame, origin: DailyTable, route: DailyTable) -> pd.DataFrame:
    """Hand-made lag features for the Stage 0 baseline (pooled window rates)."""
    day = flights["day_idx"].to_numpy()
    oi, ri = origin.index_of(flights["origin"]), route.index_of(flights["route"])
    oc = {c: i for i, c in enumerate(origin.columns)}
    rc = {c: i for i, c in enumerate(route.columns)}
    out = {}
    for w in (1, 7, 14):
        s = origin.window_sum(oi, day, w)
        out[f"o_rate_{w}d"] = _ratio(s[:, oc["n_delayed"]], s[:, oc["n_done"]])
        out[f"o_mean_delay_{w}d"] = _ratio(s[:, oc["sum_arr_delay"]], s[:, oc["n_done"]])
        out[f"o_cancel_share_{w}d"] = _ratio(s[:, oc["n_cancel"]], s[:, oc["n_sched"]])
    for w in (7, 14):
        s = route.window_sum(ri, day, w)
        out[f"r_rate_{w}d"] = _ratio(s[:, rc["n_delayed"]], s[:, rc["n_done"]])
        out[f"r_mean_delay_{w}d"] = _ratio(s[:, rc["sum_arr_delay"]], s[:, rc["n_done"]])
        out[f"r_n_done_{w}d"] = s[:, rc["n_done"]]
    return pd.DataFrame(out, index=flights.index)


LAG_FEATURES = (
    [f"o_{k}_{w}d" for w in (1, 7, 14) for k in ("rate", "mean_delay", "cancel_share")]
    + [f"r_{k}_{w}d" for w in (7, 14) for k in ("rate", "mean_delay", "n_done")]
)
