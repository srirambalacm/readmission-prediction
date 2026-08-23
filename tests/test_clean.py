import pandas as pd
import pytest

from src.load_data import load_raw
from src.clean import clean, DEATH_HOSPICE_IDS, TARGET, ID_COLS


@pytest.fixture(scope="module")
def cleaned() -> pd.DataFrame:
    return clean(load_raw(), verbose=False)


def test_one_row_per_patient(cleaned):
    assert cleaned["patient_nbr"].is_unique


def test_no_impossible_outcomes(cleaned):
    assert not cleaned["discharge_disposition_id"].isin(DEATH_HOSPICE_IDS).any()


def test_target_is_binary(cleaned):
    assert set(cleaned[TARGET].unique()) == {0, 1}


def test_raw_target_removed(cleaned):
    assert "readmitted" not in cleaned.columns


def test_no_missing_in_categoricals(cleaned):
    assert cleaned.select_dtypes(include="object").isna().sum().sum() == 0


def test_ids_retained(cleaned):
    for col in ID_COLS:
        assert col in cleaned.columns