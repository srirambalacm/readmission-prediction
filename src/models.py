"""Model constructors. Each returns an unfitted sklearn-compatible Pipeline."""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor, RANDOM_STATE
from lightgbm import LGBMClassifier


def logistic_baseline(X: pd.DataFrame, C: float = 1.0) -> Pipeline:
    """L2-regularized logistic regression.

    Preprocessing lives inside the Pipeline so that it is refit on each CV
    training fold. Fitting the scaler on the full dataset first would leak
    test-set statistics into training.
    """
    return Pipeline([
        ("pre", build_preprocessor(X, scale_numeric=True)),
        ("clf", LogisticRegression(
            C=C,
            max_iter=2000,
            random_state=RANDOM_STATE,
        )),
    ])


def lightgbm_model(X: pd.DataFrame, **params) -> Pipeline:
    """Gradient-boosted trees, no scaling: trees are invariant to monotone
    transforms of individual features."""
    defaults = dict(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    defaults.update(params)

    return Pipeline([
        ("pre", build_preprocessor(X, scale_numeric=False)),
        ("clf", LGBMClassifier(**defaults)),
    ])
def lightgbm_native(**params) -> LGBMClassifier:
    """LightGBM with native categorical handling,no one-hot expansion.

    Categorical features must be pandas 'category' dtype. LightGBM then
    partitions levels directly rather than treating each as a binary split,
    which matters for high-cardinality columns like medical_specialty.
    Returns a bare estimator, not a Pipeline, the dtype conversion happens
    outside, and there is no preprocessing to leak.
    """
    defaults = dict(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    defaults.update(params)
    return LGBMClassifier(**defaults)


def to_category_dtype(X: pd.DataFrame) -> pd.DataFrame:
    """Cast object columns to pandas 'category' for LightGBM native handling."""
    X = X.copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category")
    return X

# Selected by RandomizedSearchCV (30 candidates, 5-fold, scoring=average_precision)
# over n_estimators, learning_rate, num_leaves, min_child_samples,
# colsample_bytree, and reg_lambda. CV AUPRC 0.1851 vs 0.0897 baseline.
LGBM_BEST_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.02,
    num_leaves=31,
    min_child_samples=50,
    colsample_bytree=1.0,
    reg_lambda=1,
)