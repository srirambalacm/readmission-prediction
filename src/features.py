"""Feature specification and preprocessing for the readmission model."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


from src.clean import ID_COLS, TARGET

RANDOM_STATE = 42
TEST_SIZE = 0.2


# Integer-coded but categorica, these are lookup keys from IDS_mapping.csv,
# not quantities. Discharge code 11 is not "greater than" discharge code 3.
INT_CODED_CATEGORICAL = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

# '[80-90)' -> 85. Age is monotone in risk here, so a single numeric column
# is more parsimonious than ten dummies, and trees can still split it
# non-linearly if the relationship isn't actually linear.
AGE_MIDPOINTS = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
    "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}




def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Apply type corrections that must happen before the ColumnTransformer."""
    df = df.copy()

    df["age"] = df["age"].map(AGE_MIDPOINTS)
    if df["age"].isna().any():
        raise ValueError("unmapped age bucket encountered")

    for col in INT_CODED_CATEGORICAL:
        df[col] = df[col].astype(str)

    return df

def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, groups). Identifiers are excluded from X but returned
    for group-aware splitting and audit joins."""
    y = df[TARGET]
    groups = df["patient_nbr"]
    X = df.drop(columns=ID_COLS + [TARGET])
    return X, y, groups

def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """One-hot categoricals; optionally scale numerics.

    scale_numeric=True for linear models and the MLP (both are sensitive to
    feature magnitude), False for tree models, which are invariant to
    monotone transforms of individual features.
    """
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(include="object").columns.tolist()

    assert set(numeric) | set(categorical) == set(X.columns), "unhandled dtype"

    numeric_step = StandardScaler() if scale_numeric else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("num", numeric_step, numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
    )

def make_splits(X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE):
    """Single stratified holdout, every row is a distinct patient (first
    encounter only), so no group-aware splitting is required here."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test