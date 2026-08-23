"""Evaluation metrics for an imbalanced binary classifier."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


def precision_at_k(y_true, y_score, k: float = 0.10) -> float:
    """Precision among the top k fraction by predicted risk.

    This is the operational metric, iff a care-management team can follow up
    with k% of discharges, what share of those flagged actually readmit?
    """
    n = int(np.ceil(len(y_score) * k))
    top = np.argsort(y_score)[::-1][:n]
    return float(np.asarray(y_true)[top].mean())


def evaluate(y_true, y_score, k: float = 0.10) -> dict:
    """Threshold-free metrics appropriate for ~9% prevalence."""
    prevalence = float(np.mean(y_true))
    return {
        "auroc": roc_auc_score(y_true, y_score),
        "auprc": average_precision_score(y_true, y_score),
        "auprc_baseline": prevalence,
        "auprc_lift": average_precision_score(y_true, y_score) / prevalence,
        "brier": brier_score_loss(y_true, y_score),
        "brier_baseline": float(np.mean((prevalence - np.asarray(y_true)) ** 2)),
        f"precision@{int(k*100)}%": precision_at_k(y_true, y_score, k),
        f"lift@{int(k*100)}%": precision_at_k(y_true, y_score, k) / prevalence,
    }