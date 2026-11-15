"""Perfect-forecast inputs: gaps filled from train only, flag marks them."""
import numpy as np
import pandas as pd

from phase2.weather import FORECAST_COLS, forecast_inputs


def test_fill_uses_train_means_only():
    train = pd.DataFrame({c: [1.0, 3.0] for c in FORECAST_COLS})
    val = pd.DataFrame({c: [np.nan, 100.0] for c in FORECAST_COLS})
    tr, va = forecast_inputs(train, val)
    assert va[FORECAST_COLS].iloc[0].eq(2.0).all()  # train mean, not val's 100
    assert va["pf_observed"].tolist() == [0.0, 1.0]
    assert tr["pf_observed"].eq(1.0).all()
    assert val[FORECAST_COLS].iloc[0].isna().all()  # input not modified
