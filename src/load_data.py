from pathlib import Path
import pandas as pd

def load_raw() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "diabetic_data.csv"
    if not data_path.is_file():
        uci_url = "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008"
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. "
            f"Please download 'diabetic_data.csv' from the UCI Repository: {uci_url}"
        )
    df = pd.read_csv(data_path, na_values="?", low_memory=False)
    expected_cols = 50
    required = {"encounter_id", "patient_nbr", "readmitted", "discharge_disposition_id"}

    if df.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns, got {df.shape[1]}.")

    if df.shape[0] < 100_000:
        raise ValueError(f"Suspiciously few rows ({df.shape[0]}); download may be truncated.")

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df