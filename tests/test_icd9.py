import pandas as pd
import pytest

from src.icd9 import _categorize, add_diagnosis_groups, DIAG_COLS


@pytest.mark.parametrize(
    "code,expected",
    [
        ("414", "Circulatory"),
        ("428.0", "Circulatory"),
        ("785", "Circulatory"),
        ("486", "Respiratory"),
        ("786", "Respiratory"),
        ("250", "Diabetes"),
        ("250.83", "Diabetes"),
        ("715", "Musculoskeletal"),
        ("V57", "Other"),
        ("E909", "Other"),
        ("276", "Endocrine"),
        ("038", "Infectious"),
        ("296", "Mental"),
        ("285", "Blood"),
        ("Missing", "Missing"),
    ],
)
def test_categorize(code, expected):
    assert _categorize(code) == expected


def test_nan_is_missing():
    assert _categorize(float("nan")) == "Missing"


def test_unparseable_is_other():
    assert _categorize("???") == "Other"


def test_add_groups_replaces_raw_columns():
    df = pd.DataFrame({c: ["414", "250.83", "V57"] for c in DIAG_COLS})
    out = add_diagnosis_groups(df)
    assert all(c not in out.columns for c in DIAG_COLS)
    assert all(f"{c}_group" in out.columns for c in DIAG_COLS)