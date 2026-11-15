"""Run one tabular experiment config on train -> validation and log it.

Usage:  .venv/bin/python scripts/run_tabular.py configs/stage0/xgb_phase1.yaml

Only the validation split is scored. The test split is never touched here.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase2.baselines import FEATURE_SETS, make_model  # noqa: E402
from phase2.constants import ARTIFACTS  # noqa: E402
from phase2.features import load_flights  # noqa: E402
from phase2.metrics import validation_metrics  # noqa: E402
from phase2.tracking import append_result, load_config  # noqa: E402


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    df = load_flights()
    train, val = df[df["split"] == "train"], df[df["split"] == "validation"]
    categorical, numeric = FEATURE_SETS[cfg["features"]]
    cols = categorical + numeric

    for seed in cfg.get("seeds", [42]):
        model = make_model(cfg["model"], cfg.get("params", {}), categorical, numeric, seed)
        t = time.time()
        model.fit(train[cols], train["y"])
        seconds = time.time() - t
        prob = model.predict_proba(val[cols])[:, 1]
        m = validation_metrics(val["y"], prob)
        row = {
            "stage": cfg["stage"], "experiment": cfg["experiment"], "model": cfg["model"],
            "seed": seed, "config_path": cfg["_path"], "config_hash": cfg["_hash"],
            "n_train": len(train), "n_val": len(val), "train_seconds": round(seconds, 1),
            "notes": cfg.get("notes", ""), **m,
        }
        append_result(row)
        out = ARTIFACTS / "preds" / f"{cfg['stage']}_{cfg['experiment']}_{cfg['model']}_s{seed}_val.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, prob)
        print(f"{cfg['experiment']:>22} {cfg['model']:>8} seed={seed}  "
              f"AP={m['val_ap']:.4f}  ROC={m['val_roc_auc']:.4f}  F1={m['val_f1']:.4f}  ({seconds:.0f}s)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+")
    for c in p.parse_args().configs:
        main(c)
