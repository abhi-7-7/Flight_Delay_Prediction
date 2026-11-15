"""Stage 7: hybrid models that combine the Stage 6 network with XGBoost.

Usage:  .venv/bin/python scripts/run_hybrid.py configs/stage7/hybrids.yaml

For each seed:
  * the network is trained on 4 time blocks of the training split
    (Jan-Feb, Mar-Apr, May-Jun, Jul-Aug): each block gets predictions and
    representations from a network that never saw it ("out-of-fold");
  * one network is trained on the whole training split (used for validation
    rows, and for the "naive" variant);
  * one network is trained on Jan-May only (for the "split" variant);
  * XGBoost on Phase 1 + lag features gets out-of-fold predictions the same way.

Hybrids (all scored on validation only):
  A  network representation (64 values) + tabular features -> XGBoost
     A_naive : representations of training rows come from the network that
               was trained on those same rows
     A_split : network on Jan-May, XGBoost on Jun-Aug
     A_oof   : out-of-fold representations
  B  stacking: logistic regression on the out-of-fold logits of the network
     and of XGBoost
  C  residual: XGBoost on tabular features, starting from the network's
     out-of-fold logit (base_margin), so the trees model what the network
     got wrong
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase2.baselines import FEATURE_SETS, make_model  # noqa: E402
from phase2.constants import ARTIFACTS  # noqa: E402
from phase2.features import load_flights  # noqa: E402
from phase2.metrics import validation_metrics  # noqa: E402
from phase2.nn import (  # noqa: E402
    Batchable, FlightNet, predict, represent, seed_everything, train_model,
)
from phase2.tracking import append_result, load_config  # noqa: E402
from run_nn import DEFAULTS, prepare  # noqa: E402

FOLDS = [(1, 2), (3, 4), (5, 6), (7, 8)]
SPLIT_NN_MONTHS = (1, 2, 3, 4, 5)
EPS = 1e-6


def subset(b: Batchable, idx: np.ndarray) -> Batchable:
    t = torch.from_numpy(idx)
    return Batchable(b.cat[t], b.num[t], {k: v[t] for k, v in b.seqs.items()}, b.y[t])


def fit_net(cfg, enc, dims, train: Batchable, val: Batchable, seed: int) -> FlightNet:
    seed_everything(seed)
    model = FlightNet(enc.cardinalities(), len(enc.numeric), dims, encoder=cfg["encoder"],
                      hidden=cfg["hidden"], dropout=cfg["dropout"], window=cfg["window"])
    train_model(model, train, val, seed=seed, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                batch_size=cfg["batch_size"], max_epochs=cfg["max_epochs"],
                patience=cfg["patience"], verbose=False)
    return model


def xgb_fit_predict(X_tr, y_tr, X_va, seed, numeric, margin_tr=None, margin_va=None):
    """XGBoost with the Phase 1 preprocessing; optional base_margin (hybrid C)."""
    categorical, _ = FEATURE_SETS["phase1_lags"]
    pipe = make_model("xgboost", {}, categorical, numeric, seed)
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Xt = prep.fit_transform(X_tr)
    model.fit(Xt, y_tr, base_margin=margin_tr)
    return model.predict_proba(prep.transform(X_va), base_margin=margin_va)[:, 1]


def log(cfg, name, seed, y_val, prob, n_train, seconds, notes=""):
    m = validation_metrics(y_val, prob)
    append_result({
        "stage": cfg["stage"], "experiment": name,
        "model": name if name.endswith("_alone") else f"hybrid_{name[0]}", "seed": seed,
        "config_path": cfg["_path"], "config_hash": cfg["_hash"], "n_train": n_train,
        "n_val": len(y_val), "train_seconds": round(seconds, 1), "notes": notes, **m,
    })
    out = ARTIFACTS / "preds" / f"{cfg['stage']}_{name}_s{seed}_val.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, prob)
    print(f"   {name:>14} seed={seed}  AP={m['val_ap']:.4f}  ROC={m['val_roc_auc']:.4f}  "
          f"F1={m['val_f1']:.4f}")


def main(config_path: str) -> None:
    cfg = {**DEFAULTS, **load_config(config_path)}
    enc, train, val, dims = prepare(cfg)
    df = load_flights()
    tr_df = df[df["split"] == "train"].reset_index(drop=True)
    va_df = df[df["split"] == "validation"].reset_index(drop=True)
    categorical, numeric = FEATURE_SETS["phase1_lags"]
    cols = categorical + numeric
    y_tr, y_va = tr_df["y"].to_numpy(), va_df["y"].to_numpy()
    month = tr_df["month"].to_numpy()
    rep_cols = [f"rep_{i}" for i in range(64)]

    for seed in cfg["seeds"]:
        t0 = time.time()
        print(f"== seed {seed}")
        # --- out-of-fold network and XGBoost on the training split
        oof_p = np.zeros(len(tr_df))
        oof_rep = np.zeros((len(tr_df), 64), dtype=np.float32)
        oof_x = np.zeros(len(tr_df))
        for block in FOLDS:
            held = np.isin(month, block)
            net = fit_net(cfg, enc, dims, subset(train, np.where(~held)[0]), val, seed)
            part = subset(train, np.where(held)[0])
            oof_p[held], oof_rep[held] = predict(net, part), represent(net, part)
            oof_x[held] = xgb_fit_predict(tr_df.loc[~held, cols], y_tr[~held],
                                          tr_df.loc[held, cols], seed, numeric)
        # --- full-train models (used for validation rows)
        net = fit_net(cfg, enc, dims, train, val, seed)
        nn_va, rep_va, rep_tr_naive = predict(net, val), represent(net, val), represent(net, train)
        xgb_va = xgb_fit_predict(tr_df[cols], y_tr, va_df[cols], seed, numeric)
        log(cfg, "nn_alone", seed, y_va, nn_va, len(tr_df), 0, "Stage 6 network, same seed")
        log(cfg, "xgb_lags_alone", seed, y_va, xgb_va, len(tr_df), 0, "Stage 0 target, same seed")

        def with_rep(frame, rep):
            out = frame[cols].copy()
            out[rep_cols] = rep
            return out

        # --- A: representation -> XGBoost
        p = xgb_fit_predict(with_rep(tr_df, rep_tr_naive), y_tr, with_rep(va_df, rep_va), seed,
                            numeric + rep_cols)
        log(cfg, "A_naive", seed, y_va, p, len(tr_df), 0,
            "XGBoost on reps of rows the network was trained on")

        early = np.isin(month, SPLIT_NN_MONTHS)
        net_s = fit_net(cfg, enc, dims, subset(train, np.where(early)[0]), val, seed)
        late_rep = represent(net_s, subset(train, np.where(~early)[0]))
        p = xgb_fit_predict(with_rep(tr_df[~early], late_rep), y_tr[~early],
                            with_rep(va_df, represent(net_s, val)), seed, numeric + rep_cols)
        log(cfg, "A_split", seed, y_va, p, int((~early).sum()), 0,
            "network on Jan-May, XGBoost on Jun-Aug")

        p = xgb_fit_predict(with_rep(tr_df, oof_rep), y_tr, with_rep(va_df, rep_va), seed,
                            numeric + rep_cols)
        log(cfg, "A_oof", seed, y_va, p, len(tr_df), 0,
            "out-of-fold reps (from 4 different networks) -> XGBoost")

        # --- B: stacking on out-of-fold logits
        z_tr = np.column_stack([logit(np.clip(oof_p, EPS, 1 - EPS)),
                                logit(np.clip(oof_x, EPS, 1 - EPS))])
        z_va = np.column_stack([logit(np.clip(nn_va, EPS, 1 - EPS)),
                                logit(np.clip(xgb_va, EPS, 1 - EPS))])
        stack = LogisticRegression().fit(z_tr, y_tr)
        p = stack.predict_proba(z_va)[:, 1]
        w = stack.coef_[0]
        log(cfg, "B_stacking", seed, y_va, p, len(tr_df), 0,
            f"LR weights nn={w[0]:.3f} xgb={w[1]:.3f}")

        # --- C: residual boosting from the network's logit
        p = xgb_fit_predict(tr_df[cols], y_tr, va_df[cols], seed, numeric,
                            margin_tr=z_tr[:, 0], margin_va=z_va[:, 0])
        log(cfg, "C_residual", seed, y_va, p, len(tr_df), time.time() - t0,
            "XGBoost starting from out-of-fold network logit")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    for c in ap.parse_args().configs:
        main(c)
