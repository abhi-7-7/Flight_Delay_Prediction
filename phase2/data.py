"""Flight table: Phase 1 features (reproduced exactly) plus the timing fields
needed to apply the 6 PM D-1 cutoff to history features.

All timestamps are naive New York local time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import (
    CUTOFF_HOUR, DATASETS, DELAY_THRESHOLD_MIN, NY_TZ, SPLIT_BY_MONTH, YEAR_START,
)

# Destinations missing from nycflights13 airports.csv.
EXTRA_TZONES = {
    "BQN": "America/Puerto_Rico",
    "SJU": "America/Puerto_Rico",
    "PSE": "America/Puerto_Rico",
    "STT": "America/St_Thomas",
}


def load_raw_flights() -> pd.DataFrame:
    raw = pd.read_csv(DATASETS / "flights.zip")
    assert len(raw) == 336_776, f"Unexpected raw row count {len(raw):,}"
    return raw


def load_tzones() -> dict[str, str]:
    a = pd.read_csv(DATASETS / "airports.csv", usecols=["faa", "tzone"])
    tz = dict(zip(a["faa"], a["tzone"]))
    tz.update(EXTRA_TZONES)
    return tz


def obs_day(ts: pd.Series) -> pd.Series:
    """Observation-day index (rule R6): 18:00 on day k-1 .. 17:59 on day k -> k.

    Day index 0 is 2013-01-01.
    """
    shifted = ts + pd.Timedelta(hours=24 - CUTOFF_HOUR)
    return (shifted.dt.normalize() - YEAR_START).dt.days


def _hhmm(col: pd.Series) -> tuple[pd.Series, pd.Series]:
    v = col.astype(int)
    return v // 100, v % 100


def _to_ny(local_naive: pd.Series, dest: pd.Series, tzones: dict[str, str]) -> pd.Series:
    out = pd.Series(pd.NaT, index=local_naive.index, dtype="datetime64[ns]")
    tz_of = dest.map(tzones)
    if tz_of.isna().any():
        raise ValueError(f"No time zone for: {sorted(dest[tz_of.isna()].unique())}")
    for tz, idx in tz_of.groupby(tz_of).groups.items():
        s = local_naive.loc[idx]
        loc = s.dt.tz_localize(tz, ambiguous=np.zeros(len(s), dtype=bool),
                               nonexistent="shift_forward")
        out.loc[idx] = loc.dt.tz_convert(NY_TZ).dt.tz_localize(None)
    return out


def build_flight_table(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per scheduled flight (labelled and unlabelled).

    Adds:
      status         arrived / cancelled / diverted
      sched_dep      scheduled departure (NY time)
      known_time     when the flight's outcome became known (NY time):
                     actual arrival for arrived flights, scheduled departure
                     for cancellations (known no later than that), NaT for
                     diversions (never used)
      known_obs_day  observation day of known_time (rule R6)
      day_idx        calendar day index of scheduled departure (0 = 1 Jan)
      y, split, and the Phase 1 features
    """
    raw = load_raw_flights() if raw is None else raw
    tzones = load_tzones()
    df = raw.copy()

    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df["day_idx"] = (df["date"] - YEAR_START).dt.days
    dep_h, dep_m = _hhmm(df["sched_dep_time"])
    arr_h, arr_m = _hhmm(df["sched_arr_time"])
    df["sched_dep"] = df["date"] + pd.to_timedelta(dep_h * 60 + dep_m, unit="min")

    df["status"] = np.select(
        [df["dep_time"].isna(), df["arr_delay"].isna()],
        ["cancelled", "diverted"], default="arrived",
    )

    # Scheduled arrival in NY time, rolled to the next day when needed.
    sched_arr_local = df["date"] + pd.to_timedelta(arr_h * 60 + arr_m, unit="min")
    sched_arr_ny = _to_ny(sched_arr_local, df["dest"], tzones)
    rolled = sched_arr_ny < df["sched_dep"]
    sched_arr_ny = sched_arr_ny.where(~rolled, sched_arr_ny + pd.Timedelta(days=1))
    df["sched_arr_ny"] = sched_arr_ny

    actual_arr = sched_arr_ny + pd.to_timedelta(df["arr_delay"], unit="min")
    # A handful of source rows (8 of 327k) imply negative taxi time. Never let
    # the outcome be "known" before actual departure + air time.
    floor = (df["sched_dep"] + pd.to_timedelta(df["dep_delay"], unit="min")
             + pd.to_timedelta(df["air_time"], unit="min"))
    actual_arr = actual_arr.where(~(floor > actual_arr), floor)
    df["known_time"] = np.where(
        df["status"].eq("arrived"), actual_arr,
        np.where(df["status"].eq("cancelled"), df["sched_dep"], pd.NaT),
    )
    df["known_time"] = pd.to_datetime(df["known_time"])
    df["known_obs_day"] = obs_day(df["known_time"]).astype("Int64")

    df["route"] = df["origin"] + "-" + df["dest"]
    df["y"] = (df["arr_delay"] >= DELAY_THRESHOLD_MIN).astype("Int8")
    df.loc[df["arr_delay"].isna(), "y"] = pd.NA
    df["split"] = df["month"].map(SPLIT_BY_MONTH)

    _add_phase1_features(df, dep_h, dep_m, arr_h, arr_m)
    return df


def _add_phase1_features(df, dep_h, dep_m, arr_h, arr_m) -> None:
    """Exact reproduction of Notebooks/Preprocessing.ipynb feature code."""
    df["sched_hour"] = dep_h
    df["sched_minute"] = dep_m
    df["sched_arr_hour"] = arr_h
    df["sched_arr_minute"] = arr_m
    # Counted over all scheduled rows (labelled or not), as in Phase 1.
    df["scheduled_origin_hour_count"] = df.groupby(
        ["date", "origin", "sched_hour"])["flight"].transform("size")

    weekday = df["date"].dt.dayofweek
    doy = df["date"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["sched_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["sched_hour"] / 24)
    df["arr_hour_sin"] = np.sin(2 * np.pi * df["sched_arr_hour"] / 24)
    df["arr_hour_cos"] = np.cos(2 * np.pi * df["sched_arr_hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    df["year_day_sin"] = np.sin(2 * np.pi * (doy - 1) / 365.25)
    df["year_day_cos"] = np.cos(2 * np.pi * (doy - 1) / 365.25)

    for col in ["carrier", "origin", "dest"]:
        df[col] = df[col].astype("string").fillna("MISSING").astype(str)
    df["flight"] = df["flight"].astype(str)

    # Destination grouping learned from train-month schedule rows only.
    counts = df.loc[df["split"].eq("train"), "dest"].value_counts()
    frequent = set(counts[counts >= 100].index)
    df["dest_grouped"] = df["dest"].where(df["dest"].isin(frequent), "OTHER")


def labelled(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["y"].notna()].copy()
