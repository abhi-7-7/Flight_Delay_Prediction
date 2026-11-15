"""Tabular models (Phase 1 setup) used as Stage 0 baselines."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .constants import P1_CATEGORICAL, P1_NUMERIC
from .history import LAG_FEATURES
from .weather import DAILY_WEATHER_COLS, FORECAST_COLS

FEATURE_SETS = {
    "phase1": (P1_CATEGORICAL, P1_NUMERIC),
    "phase1_lags": (P1_CATEGORICAL, P1_NUMERIC + LAG_FEATURES),
}


def preprocessor(categorical, numeric) -> ColumnTransformer:
    """Same as Phase 1: median impute + scale; most-frequent impute + one-hot.
    `add_indicator` flags missing history values (e.g. a route with no recent
    flights) without changing Phase 1 behaviour, since Phase 1 features have
    no missing values."""
    num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))])
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)])


def make_model(name: str, params: dict, categorical, numeric, seed: int) -> Pipeline:
    if name == "dummy":
        est = DummyClassifier(strategy="prior")
    elif name == "logreg":
        est = LogisticRegression(**{"max_iter": 1000, **params}, random_state=seed)
    elif name == "xgboost":
        est = XGBClassifier(**{
            "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "objective": "binary:logistic", "eval_metric": "logloss",
            "tree_method": "hist", "n_jobs": -1, **params,
        }, random_state=seed)
    else:
        raise ValueError(name)
    return Pipeline([("prep", preprocessor(categorical, numeric)), ("model", est)])


assert set(FORECAST_COLS).isdisjoint(c for _, (cat, num) in FEATURE_SETS.items() for c in cat + num)
assert set(DAILY_WEATHER_COLS).isdisjoint(c for _, (cat, num) in FEATURE_SETS.items() for c in cat + num)
