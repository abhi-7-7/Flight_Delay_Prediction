"""Validation metrics, computed exactly as in Phase 1.

Model selection uses validation Average Precision. The decision threshold is
the one that maximises F1 on validation. The test split is never scored here.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
)


def f1_threshold(y_true, prob) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    i = int(np.argmax(f1))
    return float(thresholds[i]), float(precision[i]), float(recall[i]), float(f1[i])


def validation_metrics(y_true, prob) -> dict:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob, dtype=float)
    base = float(y_true.mean())
    ap = float(average_precision_score(y_true, prob))
    thr, p, r, f1 = f1_threshold(y_true, prob)
    return {
        "val_base_rate": base,
        "val_ap": ap,
        "val_ap_lift": ap / base,
        "val_roc_auc": float(roc_auc_score(y_true, prob)),
        "val_threshold": thr,
        "val_precision": p,
        "val_recall": r,
        "val_f1": f1,
    }
