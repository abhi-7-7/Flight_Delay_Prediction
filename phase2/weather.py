"""Weather features.

Hourly rows come from nycflights13 `weather.csv`. The `hour` column is local
time and refers to the METAR report taken at about hh:51, so hour 17 is known
before 18:00 and hour 18 is not (rules R7 and R9).

Missing hours are NOT forward-filled: forward-filling rainfall would count the
same rain twice. Instead each summary records how many hours were observed,
and summaries with no observations are left NaN for the model's
training-split imputer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import DATASETS, YEAR_START
from .data import obs_day

FREEZING_F = 32.0
LOW_VISIBILITY_MI = 3.0

DAILY_WEATHER_COLS = [
    "w_obs_hours", "w_precip_sum", "w_precip_hours", "w_snow_risk_hours",
    "w_visib_min", "w_low_vis_hours", "w_wind_max", "w_gust_any",
    "w_temp_min", "w_temp_max", "w_humid_mean", "w_pressure_min",
    "w_pressure_change", "w_wind_dir_sin", "w_wind_dir_cos",
]

FORECAST_COLS = [
    "pf_obs_hours", "pf_precip_sum", "pf_snow_risk_hours", "pf_visib_min",
    "pf_wind_max", "pf_gust_any", "pf_temp_min",
]


def load_hourly() -> pd.DataFrame:
    w = pd.read_csv(DATASETS / "weather.csv")
    w = w.drop(columns=["time_hour"])  # UTC; never join on it (rule R9)
    w["ts"] = pd.to_datetime(w[["year", "month", "day"]]) + pd.to_timedelta(w["hour"], unit="h")
    # A few duplicated local hours exist at the DST change; keep one row each.
    w = w.sort_values(["origin", "ts"]).drop_duplicates(["origin", "ts"], keep="last")
    # Known source error: one EWR reading of 1,048 mph. Treat impossible
    # values as missing.
    for col in ["wind_speed", "wind_gust"]:
        w.loc[w[col] > 100, col] = np.nan
    w["precip"] = w["precip"].fillna(0.0)
    w["rain"] = (w["precip"] > 0).astype(float)
    w["snow_risk"] = ((w["precip"] > 0) & (w["temp"] <= FREEZING_F)).astype(float)
    w.loc[w["temp"].isna(), "snow_risk"] = np.nan
    w["low_vis"] = (w["visib"] < LOW_VISIBILITY_MI).astype(float)
    w["gust"] = w["wind_gust"].notna().astype(float)
    calm = w["wind_speed"].fillna(0) <= 0
    rad = np.deg2rad(w["wind_dir"])
    w["dir_sin"] = np.where(calm, np.nan, np.sin(rad))
    w["dir_cos"] = np.where(calm, np.nan, np.cos(rad))
    return w.reset_index(drop=True)


def daily_summaries(hourly: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (origin, observation day). Uses only hours known by 18:00."""
    w = load_hourly() if hourly is None else hourly
    w = w.assign(obs_day=obs_day(w["ts"]))

    def pressure_change(s: pd.Series) -> float:
        s = s.dropna()
        return float(s.iloc[-1] - s.iloc[0]) if len(s) >= 2 else np.nan

    g = w.sort_values("ts").groupby(["origin", "obs_day"])
    out = pd.DataFrame({
        "w_obs_hours": g.size(),
        "w_precip_sum": g["precip"].sum(),
        "w_precip_hours": g["rain"].sum(),
        "w_snow_risk_hours": g["snow_risk"].sum(),
        "w_visib_min": g["visib"].min(),
        "w_low_vis_hours": g["low_vis"].sum(),
        "w_wind_max": g["wind_speed"].max(),
        "w_gust_any": g["gust"].max(),
        "w_temp_min": g["temp"].min(),
        "w_temp_max": g["temp"].max(),
        "w_humid_mean": g["humid"].mean(),
        "w_pressure_min": g["pressure"].min(),
        "w_pressure_change": g["pressure"].apply(pressure_change),
        "w_wind_dir_sin": g["dir_sin"].mean(),
        "w_wind_dir_cos": g["dir_cos"].mean(),
    }).reset_index()
    return out[out["obs_day"].between(0, 364)].reset_index(drop=True)[
        ["origin", "obs_day"] + DAILY_WEATHER_COLS]


def perfect_forecast(flights: pd.DataFrame, hourly: pd.DataFrame | None = None,
                     half_window_h: int = 2) -> pd.DataFrame:
    """Rule R8 upper-bound features: actual origin weather in the
    [-2 h, +2 h] window around each flight's scheduled departure hour.

    Returns a frame aligned to `flights.index`. NEVER use in main models.
    """
    w = load_hourly() if hourly is None else hourly
    grid_idx = pd.date_range(YEAR_START, "2013-12-31 23:00", freq="h")
    frames = []
    win = 2 * half_window_h + 1
    for origin, g in w.groupby("origin"):
        h = g.set_index("ts").reindex(grid_idx)
        obs = h["temp"].notna().astype(float)
        r = lambda s, how: getattr(s.rolling(win, center=True, min_periods=1), how)()
        f = pd.DataFrame({
            "pf_obs_hours": r(obs, "sum"),
            "pf_precip_sum": r(h["precip"], "sum"),
            "pf_snow_risk_hours": r(h["snow_risk"], "sum"),
            "pf_visib_min": r(h["visib"], "min"),
            "pf_wind_max": r(h["wind_speed"], "max"),
            "pf_gust_any": r(h["gust"], "max"),
            "pf_temp_min": r(h["temp"], "min"),
        }, index=grid_idx)
        f.loc[f["pf_obs_hours"] == 0, FORECAST_COLS[1:]] = np.nan
        f["origin"] = origin
        frames.append(f.rename_axis("dep_hour").reset_index())
    pf = pd.concat(frames, ignore_index=True)
    key = flights[["origin"]].assign(dep_hour=flights["sched_dep"].dt.floor("h"))
    merged = key.merge(pf, on=["origin", "dep_hour"], how="left")
    merged.index = flights.index
    return merged[FORECAST_COLS]
