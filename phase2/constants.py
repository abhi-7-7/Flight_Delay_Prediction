"""Fixed design rules for Phase 2.

These are constraints, not tuned choices. They are justified in DECISIONS.md
and must not be changed by experiments.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "Datasets"
ARTIFACTS = ROOT / "artifacts"
FEATURES_DIR = ARTIFACTS / "features"
EXPERIMENTS_DIR = ROOT / "experiments"
RESULTS_CSV = EXPERIMENTS_DIR / "results.csv"

NY_TZ = "America/New_York"
YEAR_START = pd.Timestamp("2013-01-01")
N_DAYS = 365

# Prediction cutoff: 6:00 PM local time on the day before departure (D-1).
CUTOFF_HOUR = 18

# Target: arrival delay of at least 15 minutes (US DOT on-time definition).
DELAY_THRESHOLD_MIN = 15

# Chronological splits (calendar month of scheduled departure).
SPLIT_BY_MONTH = {m: "train" for m in range(1, 9)}
SPLIT_BY_MONTH.update({9: "validation", 10: "validation", 11: "test", 12: "test"})
SPLITS = ("train", "validation", "test")

# Longest history window used by any experiment (days before D).
MAX_WINDOW = 14

# Phase 1 feature set, reproduced exactly so Phase 2 comparisons are fair.
P1_CATEGORICAL = ["carrier", "origin", "dest_grouped", "flight"]
P1_NUMERIC = [
    "distance", "sched_minute", "sched_arr_minute", "scheduled_origin_hour_count",
    "hour_sin", "hour_cos", "arr_hour_sin", "arr_hour_cos",
    "weekday_sin", "weekday_cos", "year_day_sin", "year_day_cos",
]

# Columns that are only known after departure (or are identifiers / labels).
FORBIDDEN_INPUTS = {
    "dep_time", "dep_delay", "arr_time", "air_time", "arr_delay",
    "id", "name", "tailnum", "y", "date", "split",
    "actual_dep", "actual_arr", "known_time",
}

RANDOM_STATE = 42
