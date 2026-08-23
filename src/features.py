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
DISCHARGE_GROUPS = {
    1: "Home", 6: "HomeHealth", 8: "HomeIV",
    2: "Facility", 3: "SNF", 4: "ICF", 5: "Facility",
    15: "Facility", 22: "Rehab", 23: "LongTerm", 24: "SNF",
    27: "Facility", 28: "Facility", 29: "Facility", 30: "Facility",
    7: "AMA",
    9: "Other", 10: "Other", 12: "Other", 16: "Other", 17: "Other",
    18: "Unknown", 25: "Unknown", 26: "Unknown",
}

_MED_VALUES = {"No", "Steady", "Up", "Down"}

def _medication_columns(df: pd.DataFrame) -> list[str]:
    """Detect diabetes medication columns by their value signature."""
    return [
        c for c in df.select_dtypes(include="object").columns
        if set(df[c].dropna().unique()) <= _MED_VALUES
    ]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-motivated derived features.

    Must run BEFORE prepare(): DISCHARGE_GROUPS is keyed on integers, and
    prepare() casts discharge_disposition_id to string.
    """
    df = df.copy()

    if not pd.api.types.is_integer_dtype(df["discharge_disposition_id"]):
        raise TypeError(
            "add_engineered_features must run before prepare(): "
            "discharge_disposition_id has already been cast to string."
        )

    med_cols = _medication_columns(df)

    # Prior acute-care use is the best established readmission predictor.
    # Trees can find thresholds, but explicit flags make the signal cheaper
    # to learn at low prevalence.
    df["prior_acute"] = df["number_inpatient"] + df["number_emergency"]
    df["any_prior_inpatient"] = (df["number_inpatient"] > 0).astype(int)
    df["frequent_prior_inpatient"] = (df["number_inpatient"] >= 2).astype(int)
    df["total_prior_visits"] = df["prior_acute"] + df["number_outpatient"]

    # Active medication management during the stay: a changed regimen implies
    # the admission was managing unstable disease.
    df["n_meds_active"] = (df[med_cols] != "No").sum(axis=1)
    df["n_meds_changed"] = df[med_cols].isin(["Up", "Down"]).sum(axis=1)

    # Care intensity, normalized by length of stay.
    df["meds_per_day"] = df["num_medications"] / df["time_in_hospital"]
    df["procedures_per_day"] = df["num_procedures"] / df["time_in_hospital"]
    df["labs_per_day"] = df["num_lab_procedures"] / df["time_in_hospital"]

    # Discharge destination, grouped.
    df["discharge_group"] = (
        df["discharge_disposition_id"].map(DISCHARGE_GROUPS).fillna("Other")
    )

    return df



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