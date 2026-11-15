"""Stage 8: score the selected model on the test split (Nov-Dec 2013), once.

Usage:  .venv/bin/python scripts/run_final.py configs/stage8/final.yaml

Models scored: the selected network (Stage 6), stacking (secondary result
declared in Stage 7), XGBoost + lags (Stage 0 target) and Phase 1's
logistic regression. Everything is trained exactly as during selection (training split, early
stopping and decision threshold from validation). The test split is only
predicted, never used for any choice. The Stage 0 baselines are scored
alongside so the final numbers can be compared on the same months.

Writes experiments/final_test.csv (one row per model and seed, plus the
3-seed average "ensemble") and experiments/final_test_by_month.csv.
"""
import argparse
import os
import sys

# PyTorch and XGBoost each bring their own OpenMP runtime on macOS, and using
# both in one process with default threading crashes (segfault). Starting
# single-threaded and then setting thread counts explicitly avoids it.
os.environ.setdefault("OMP_NUM_THREADS", "1")
N_THREADS = os.cpu_count() or 1
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase2.baselines import FEATURE_SETS, make_model  # noqa: E402
from phase2.constants import EXPERIMENTS_DIR  # noqa: E402
from phase2.features import load_flights  # noqa: E402
from phase2.metrics import f1_threshold  # noqa: E402
from phase2.nn import predict  # noqa: E402
from phase2.tracking import git_commit, load_config  # noqa: E402
from run_hybrid import EPS, FOLDS, fit_net, subset, xgb_fit_predict  # noqa: E402
from run_nn import DEFAULTS, prepare  # noqa: E402


def score(y_val, p_val, y_test, p_test) -> dict:
    """Test metrics. The threshold is chosen on validation (rule R10)."""
    thr, *_ = f1_threshold(y_val, p_val)
    pred = (p_test >= thr).astype(int)
    tp = int(((pred == 1) & (y_test == 1)).sum())
    precision = tp / max(pred.sum(), 1)
    recall = tp / max(y_test.sum(), 1)
    return {
        "test_base_rate": y_test.mean(),
        "test_ap": average_precision_score(y_test, p_test),
        "test_ap_lift": average_precision_score(y_test, p_test) / y_test.mean(),
        "test_roc_auc": roc_auc_score(y_test, p_test),
        "threshold_from_val": thr,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "test_accuracy_at_threshold": accuracy_score(y_test, pred),
        "test_accuracy_at_0.5": accuracy_score(y_test, (p_test >= 0.5).astype(int)),
        "test_accuracy_all_on_time": 1 - y_test.mean(),
    }


def main(config_path: str) -> None:
    cfg = {**DEFAULTS, **load_config(config_path)}
    torch.set_num_threads(N_THREADS)
    enc, train, val, dims, test = prepare(cfg, with_test=True)
    df = load_flights()
    parts = {s: df[df["split"] == s].reset_index(drop=True)
             for s in ["train", "validation", "test"]}
    y_val, y_test = parts["validation"]["y"].to_numpy(), parts["test"]["y"].to_numpy()
    month = parts["test"]["month"].to_numpy()

    cat, num = FEATURE_SETS["phase1_lags"]
    cols = cat + num
    tr_month = parts["train"]["month"].to_numpy()
    y_tr = parts["train"]["y"].to_numpy()
    preds = {}  # model -> list of (seed, p_val, p_test)
    for seed in cfg["seeds"]:
        # selected model: the Stage 6 network
        net = fit_net(cfg, enc, dims, train, val, seed)
        nn_v, nn_t = predict(net, val), predict(net, test)
        preds.setdefault("final_network", []).append((seed, nn_v, nn_t))
        # Stage 0 target: XGBoost + lags
        x_v = xgb_fit_predict(parts["train"][cols], y_tr, parts["validation"][cols], seed, num)
        x_t = xgb_fit_predict(parts["train"][cols], y_tr, parts["test"][cols], seed, num)
        preds.setdefault("xgb_phase1_lags", []).append((seed, x_v, x_t))
        # secondary result declared in Stage 7: stacking on out-of-fold logits
        oof_n, oof_x = np.zeros(len(y_tr)), np.zeros(len(y_tr))
        for block in FOLDS:
            held = np.isin(tr_month, block)
            fold_net = fit_net(cfg, enc, dims, subset(train, np.where(~held)[0]), val, seed)
            oof_n[held] = predict(fold_net, subset(train, np.where(held)[0]))
            oof_x[held] = xgb_fit_predict(parts["train"].loc[~held, cols], y_tr[~held],
                                          parts["train"].loc[held, cols], seed, num)
        z = lambda a, b: np.column_stack([logit(np.clip(a, EPS, 1 - EPS)),
                                          logit(np.clip(b, EPS, 1 - EPS))])
        stack = LogisticRegression().fit(z(oof_n, oof_x), y_tr)
        preds.setdefault("hybrid_B_stacking", []).append(
            (seed, stack.predict_proba(z(nn_v, x_v))[:, 1], stack.predict_proba(z(nn_t, x_t))[:, 1]))
        print(f"   seed {seed} done")
    # Phase 1 model, for reference
    cat, num = FEATURE_SETS["phase1"]
    m = make_model("logreg", {}, cat, num, 42).fit(parts["train"][cat + num], parts["train"]["y"])
    preds["logreg_phase1"] = [(42, m.predict_proba(parts["validation"][cat + num])[:, 1],
                               m.predict_proba(parts["test"][cat + num])[:, 1])]

    rows, by_month = [], []
    for name, runs in preds.items():
        for seed, pv, pt in runs:
            rows.append({"model": name, "seed": seed, **score(y_val, pv, y_test, pt)})
        if len(runs) > 1:  # average of the seeds' probabilities
            pv = np.mean([r[1] for r in runs], axis=0)
            pt = np.mean([r[2] for r in runs], axis=0)
            rows.append({"model": name, "seed": "mean_of_seeds", **score(y_val, pv, y_test, pt)})
        for mth in (11, 12):
            k = month == mth
            by_month.append({"model": name, "month": mth, "base_rate": y_test[k].mean(),
                             "test_ap": average_precision_score(y_test[k], pt[k]),
                             "test_roc_auc": roc_auc_score(y_test[k], pt[k])})
        np.save(EXPERIMENTS_DIR.parent / "artifacts" / "preds" / f"8_{name}_test.npy", pt)

    out = pd.DataFrame(rows)
    out.insert(0, "git_commit", git_commit())
    out.to_csv(EXPERIMENTS_DIR / "final_test.csv", index=False, float_format="%.4f")
    pd.DataFrame(by_month).to_csv(EXPERIMENTS_DIR / "final_test_by_month.csv", index=False,
                                  float_format="%.4f")
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    print(pd.DataFrame(by_month).round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    main(ap.parse_args().config)
