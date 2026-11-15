"""Build and load the Phase 2 feature store (artifacts/features/).

The store is regenerated from Datasets/ by `scripts/build_features.py` and is
not committed (it is deterministic and large).
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from .constants import FEATURES_DIR, MAX_WINDOW, N_DAYS, P1_CATEGORICAL, P1_NUMERIC
from .data import build_flight_table, labelled
from .history import (
    DailyTable, LAG_FEATURES, build_origin_daily, build_route_daily, lag_features,
)
from .weather import DAILY_WEATHER_COLS, daily_summaries, load_hourly, perfect_forecast

KEY_COLS = ["day_idx", "date", "origin", "dest", "route", "split", "y", "month"]


def weather_table(daily: pd.DataFrame, origin_keys: list[str]) -> DailyTable:
    """Daily weather as a DailyTable aligned with the origin history table."""
    vals = np.full((len(origin_keys), MAX_WINDOW + N_DAYS, len(DAILY_WEATHER_COLS)), np.nan)
    lookup = {k: i for i, k in enumerate(origin_keys)}
    e = daily["origin"].map(lookup).to_numpy()
    d = daily["obs_day"].to_numpy() + MAX_WINDOW
    vals[e, d] = daily[DAILY_WEATHER_COLS].to_numpy(dtype=float)
    return DailyTable("weather", origin_keys, DAILY_WEATHER_COLS, vals)


def build_store() -> dict:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    flights = build_flight_table()
    lab = labelled(flights)
    origin, route = build_origin_daily(flights), build_route_daily(flights)
    hourly = load_hourly()
    weather = weather_table(daily_summaries(hourly), origin.keys)

    table = pd.concat([
        lab[KEY_COLS + P1_CATEGORICAL + P1_NUMERIC],
        lag_features(lab, origin, route),
        perfect_forecast(lab, hourly),
    ], axis=1)
    table = table.loc[:, ~table.columns.duplicated()].reset_index(drop=True)
    table["y"] = table["y"].astype(int)
    table.to_parquet(FEATURES_DIR / "flights.parquet", index=False)
    with open(FEATURES_DIR / "daily_tables.pkl", "wb") as f:
        pickle.dump({"origin": origin, "route": route, "weather": weather}, f)
    return {"rows": len(table), "columns": table.shape[1],
            "lag_features": len(LAG_FEATURES)}


def load_flights() -> pd.DataFrame:
    path = FEATURES_DIR / "flights.parquet"
    if not path.exists():
        raise FileNotFoundError("Feature store missing: run scripts/build_features.py")
    return pd.read_parquet(path)


def load_daily_tables() -> dict[str, DailyTable]:
    with open(FEATURES_DIR / "daily_tables.pkl", "rb") as f:
        return pickle.load(f)
