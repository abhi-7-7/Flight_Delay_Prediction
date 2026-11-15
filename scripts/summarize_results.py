"""Summarise experiments/results.csv as mean ± std per configuration.

Usage:  .venv/bin/python scripts/summarize_results.py [--stage 0] [--markdown]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase2.constants import RESULTS_CSV  # noqa: E402


def summarize(stage=None) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    if stage is not None:
        df = df[df["stage"].astype(str) == str(stage)]
    g = df.groupby(["stage", "experiment", "model"], sort=False)
    out = g.agg(
        runs=("seed", "size"),
        val_ap_mean=("val_ap", "mean"), val_ap_std=("val_ap", "std"),
        val_roc_auc_mean=("val_roc_auc", "mean"), val_f1_mean=("val_f1", "mean"),
        val_ap_lift=("val_ap_lift", "mean"),
    ).reset_index()
    return out.fillna({"val_ap_std": 0.0})


def to_markdown(s: pd.DataFrame) -> str:
    lines = ["| Experiment | Model | Runs | Val AP (mean ± std) | AP lift | Val ROC-AUC | Val F1 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in s.itertuples():
        lines.append(f"| {r.experiment} | {r.model} | {r.runs} | {r.val_ap_mean:.4f} ± {r.val_ap_std:.4f} "
                     f"| {r.val_ap_lift:.2f}× | {r.val_roc_auc_mean:.4f} | {r.val_f1_mean:.4f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage")
    p.add_argument("--markdown", action="store_true")
    a = p.parse_args()
    s = summarize(a.stage)
    print(to_markdown(s) if a.markdown else s.round(4).to_string(index=False))
