from __future__ import annotations
import pandas as pd

# Discharge dispositions where readmission is impossible or not meaningful.
# IDs sourced from data/raw/IDS_mapping.csv (discharge_disposition_id section)
# 11 = Expired
# 13 = Hospice / home
# 14 = Hospice /medical facility
# 19 = Expired at home(hospice)
# 20 = Expired in medical facility(hospice)
# 21 = Expired, place unknown (hospice)
DEATH_HOSPICE_IDS = {11, 13, 14, 19, 20, 21}

# Identifier columns: kept in  frame for joins/audits
ID_COLS = ["encounter_id", "patient_nbr"]

TARGET_RAW = "readmitted"
TARGET = "readmitted_30d"


def _filter_rows(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Reduce to one row per patient and remove impossible outcomes."""
    n_start = len(df)
    # first encounter per patient,encounter_id is assumed monotonically
    # increasing in time -- the dataset has no timestamps, so this is the
    # only available ordering.
    df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    n_first = len(df)

    # Patients who died or entered hospice cannot be readmitted; leaving them
    # in labels them as negatives for the wrong reason.
    df = df[~df["discharge_disposition_id"].isin(DEATH_HOSPICE_IDS)]
    n_final = len(df)

    if verbose:
        print(f"rows: {n_start} -> {n_first} (first encounter) -> {n_final} (alive)")

    return df.reset_index(drop=True)

DROP_COLS = ["weight"]

def _drop_uninformative(
    df: pd.DataFrame,
    nzv_threshold: float = 0.995,
    verbose: bool = True,
) -> pd.DataFrame:
    """Dropping constant, near-constant, and explicitly excluded columns."""
    dropped = {}

    for col in DROP_COLS:
        if col in df.columns:
            dropped[col] = "explicit"

    protected = set(ID_COLS) | {TARGET_RAW}

    for col in df.columns:
        if col in protected or col in dropped:
            continue

        counts = df[col].value_counts(dropna=False)
        if len(counts) <= 1:
            dropped[col] = "constant"
        elif counts.iloc[0] / len(df) >= nzv_threshold:
            dropped[col] = f"near-constant ({counts.iloc[0] / len(df):.4f})"

    if verbose:
        print(f"dropping {len(dropped)} columns:")
        for col, reason in dropped.items():
            print(f"  {col}: {reason}")

    return df.drop(columns=list(dropped))

# Lab columns where "missing" means the test was never ordered 
# Strack et al. (2014) found A1C *measurement* (not just its value) predicts readmission, so this should not be imputed away.
NOT_TESTED_COLS = ["A1Cresult", "max_glu_serum"]
NOT_TESTED_FILL = "NotTested"
MISSING_FILL = "Missing"


def _encode_missing(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Convert NaN to explicit categories, nothing here is imputed."""
    df = df.copy()

    for col in NOT_TESTED_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(NOT_TESTED_FILL)

    obj_cols = df.select_dtypes(include="object").columns
    remaining = [c for c in obj_cols if df[c].isna().any()]

    for col in remaining:
        df[col] = df[col].fillna(MISSING_FILL)

    if verbose:
        print(f"missing-as-category: {NOT_TESTED_COLS} -> '{NOT_TESTED_FILL}'")
        print(f"                     {remaining} -> '{MISSING_FILL}'")

    assert df.select_dtypes(include="object").isna().sum().sum() == 0

    return df


def _build_target(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Binarize the outcome and remove raw column so it cannot leak."""
    df = df.copy()

    # '<30' is positive; '>30' and 'NO' are both negative, matches the
    # CMS HRRP 30-day window, but  it collapses late readmissions into
    # the negative class which is a modeling decision, not property of the data.
    df[TARGET] = (df[TARGET_RAW] == "<30").astype(int)
    df = df.drop(columns=[TARGET_RAW])
    if verbose:
        rate = df[TARGET].mean()
        print(f"target '{TARGET}': {df[TARGET].sum()} positives, rate {rate:.4f}")

    return df
def clean(df: pd.DataFrame, nzv_threshold: float = 0.995, verbose: bool = True) -> pd.DataFrame:
    """Full cleaning pipeline: raw dataframe -> modelling-ready frame.

    Rows are filtered before variance is computed
    """
    df = _filter_rows(df, verbose=verbose)
    df = _drop_uninformative(df, nzv_threshold=nzv_threshold, verbose=verbose)
    df = _encode_missing(df, verbose=verbose)
    df = _build_target(df, verbose=verbose)

    assert df[ID_COLS[1]].is_unique, "expected one row per patient"
    assert TARGET in df.columns and TARGET_RAW not in df.columns

    return df