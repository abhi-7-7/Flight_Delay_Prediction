"""Append-only experiment log.

Every run (including failed or losing ones) appends one row to
experiments/results.csv. Rows are never edited or deleted, so the file is the
evidence trail for how each Phase 2 design choice was made.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .constants import RESULTS_CSV, ROOT

RESULT_COLUMNS = [
    "run_id", "timestamp_utc", "git_commit", "stage", "experiment", "model",
    "seed", "config_path", "config_hash", "n_train", "n_val",
    "val_base_rate", "val_ap", "val_ap_lift", "val_roc_auc",
    "val_threshold", "val_precision", "val_recall", "val_f1",
    "train_seconds", "notes",
]


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(path.resolve().relative_to(ROOT))
    cfg["_hash"] = hashlib.sha1(
        json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")},
                   sort_keys=True).encode()
    ).hexdigest()[:10]
    return cfg


def git_commit() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", "phase2", "scripts", "configs"],
            cwd=ROOT,
        ).returncode != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def append_result(row: dict) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RESULTS_CSV.exists()
    row = {**row}
    row.setdefault("timestamp_utc", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    row.setdefault("git_commit", git_commit())
    row.setdefault(
        "run_id",
        f"{row['stage']}-{row['experiment']}-{row['model']}-s{row['seed']}-"
        f"{row['timestamp_utc'].replace(':', '').replace('-', '')}",
    )
    unknown = set(row) - set(RESULT_COLUMNS)
    if unknown:
        raise KeyError(f"Unknown result columns: {sorted(unknown)}")
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow({k: _fmt(row.get(k, "")) for k in RESULT_COLUMNS})


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6f}"
    return v
