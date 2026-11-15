"""Run one neural-network experiment config (all its seeds) and log it.

Usage:  .venv/bin/python scripts/run_nn.py configs/stage1/gru_origin_route.yaml [...]

Trains on the training split, early-stops on validation AP, logs validation
metrics. The test split is never touched here.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase2.constants import ARTIFACTS  # noqa: E402
from phase2.features import load_daily_tables, load_flights  # noqa: E402
from phase2.metrics import validation_metrics  # noqa: E402
from phase2.nn import (  # noqa: E402
    FlightNet, StaticEncoder, make_batchable, predict, seed_everything, train_model,
)
from phase2.sequences import build_sequences  # noqa: E402
from phase2.tracking import append_result, load_config  # noqa: E402

DEFAULTS = {
    "branches": ["origin", "route"], "window": 14, "weather": False,
    "flight_embedding": False, "encoder": "gru", "hidden": 32, "dropout": 0.2,
    "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 1024, "max_epochs": 25,
    "patience": 4, "seeds": [42, 43, 44],
}


def prepare(cfg: dict):
    df = load_flights()
    tables = load_daily_tables()
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "validation"].reset_index(drop=True)
    categorical = ["carrier", "origin", "dest_grouped"] + (["flight"] if cfg["flight_embedding"] else [])
    enc = StaticEncoder(categorical).fit(train)
    out = {}
    for name, part in [("train", train), ("val", val)]:
        cat, num = enc.transform(part)
        seq = build_sequences(part, tables, cfg["window"], use_weather=cfg["weather"])
        seqs = {b: seq[b] for b in cfg["branches"]}
        out[name] = make_batchable(cat, num, seqs, part["y"].to_numpy())
    dims = {b: out["train"].seqs[b].shape[-1] for b in cfg["branches"]}
    return enc, out["train"], out["val"], dims


def main(config_path: str) -> None:
    cfg = {**DEFAULTS, **load_config(config_path)}
    torch.set_num_threads(cfg.get("threads", torch.get_num_threads()))
    enc, train, val, dims = prepare(cfg)
    print(f"== {cfg['experiment']}  branches={cfg['branches']}  window={cfg['window']}  "
          f"weather={cfg['weather']}  encoder={cfg['encoder']}")
    for seed in cfg["seeds"]:
        seed_everything(seed)
        model = FlightNet(enc.cardinalities(), len(enc.numeric), dims, encoder=cfg["encoder"],
                          hidden=cfg["hidden"], dropout=cfg["dropout"])
        info = train_model(model, train, val, seed=seed, lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"], batch_size=cfg["batch_size"],
                           max_epochs=cfg["max_epochs"], patience=cfg["patience"],
                           verbose=cfg.get("verbose", False))
        prob = predict(model, val)
        m = validation_metrics(val.y.numpy(), prob)
        append_result({
            "stage": cfg["stage"], "experiment": cfg["experiment"], "model": f"nn_{cfg['encoder']}",
            "seed": seed, "config_path": cfg["_path"], "config_hash": cfg["_hash"],
            "n_train": len(train), "n_val": len(val), "train_seconds": round(info["seconds"], 1),
            "notes": f"best_epoch={info['best_epoch']}/{info['epochs_run']}; {cfg.get('notes', '')}",
            **m,
        })
        tag = f"{cfg['stage']}_{cfg['experiment']}_nn_{cfg['encoder']}_s{seed}"
        (ARTIFACTS / "preds").mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "models").mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "curves").mkdir(parents=True, exist_ok=True)
        np.save(ARTIFACTS / "preds" / f"{tag}_val.npy", prob)
        torch.save(model.state_dict(), ARTIFACTS / "models" / f"{tag}.pt")
        (ARTIFACTS / "curves" / f"{tag}.json").write_text(json.dumps(info["history"]))
        print(f"   seed={seed}  AP={m['val_ap']:.4f}  ROC={m['val_roc_auc']:.4f}  "
              f"F1={m['val_f1']:.4f}  best_epoch={info['best_epoch']}/{info['epochs_run']}  "
              f"({info['seconds']:.0f}s)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+")
    for c in p.parse_args().configs:
        main(c)
