"""Map ICD-9 diagnosis codes to nine clinical categories.

Grouping follows Strack et al. (2014), Table 2, Ranges are on the integer
part of the code: 250.83 -> 250 -> Diabetes.

V-codes (supplementary factors) and E-codes (external causes) are not on the
numeric scale and fall into 'Other'.
"""

from __future__ import annotations
import pandas as pd

DIAG_COLS = ["diag_1", "diag_2", "diag_3"]

OTHER = "Other"
MISSING = "Missing"

# (inclusive_low, inclusive_high, category)
_RANGES = [
    (390, 459, "Circulatory"),
    (785, 785, "Circulatory"),
    (460, 519, "Respiratory"),
    (786, 786, "Respiratory"),
    (520, 579, "Digestive"),
    (787, 787, "Digestive"),
    (580, 629, "Genitourinary"),
    (788, 788, "Genitourinary"),
    (800, 999, "Injury"),
    (710, 739, "Musculoskeletal"),
    (140, 239, "Neoplasms"),
]


def _categorize(code) -> str:
    """Map a single ICD-9 code string to its clinical category."""
    if pd.isna(code) or str(code).strip() in {"", MISSING}:
        return MISSING

    text = str(code).strip()

    # V- and E-codes are supplementary/external cause codes with no place on the numeric ranges above.
    if text[0].upper() in {"V", "E"}:
        return OTHER

    try:
        value = float(text)
    except ValueError:
        return OTHER

    # Diabetes is the 250.xx block and is checked first, since 250 also falls 
    # inside no other range but is the clinically meaningful group here.
    if 250 <= value < 251:
        return "Diabetes"

    whole = int(value)
    for low, high, category in _RANGES:
        if low <= whole <= high:
            return category

    return OTHER

def add_diagnosis_groups(df: pd.DataFrame, drop_raw: bool = True) -> pd.DataFrame:
    """Add grouped versions of diag_1/2/3."""
    df = df.copy()

    for col in DIAG_COLS:
        df[f"{col}_group"] = df[col].map(_categorize)

    if drop_raw:
        df = df.drop(columns=DIAG_COLS)

    return df

