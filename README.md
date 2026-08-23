# 30-Day Hospital Readmission Prediction

Predicting 30-day hospital readmission for diabetic patients using the
[UCI Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
(101,766 encounters, 130 hospitals, 1999–2008).

Readmission within 30 days is the window used by the CMS Hospital Readmissions
Reduction Program, which penalizes hospitals for excess readmissions, so the
prediction target maps to a real operational and financial decision.

## Status

Work in progress.

- [x] Data loading with schema validation
- [x] Cleaning pipeline (leakage filtering, variance-based column pruning, missing-as-category)
- [x] ICD-9 diagnosis grouping
- [ ] Feature engineering
- [ ] Models: logistic regression, LightGBM, PyTorch MLP
- [ ] Evaluation: AUROC, AUPRC, calibration, precision@k, SHAP, subgroup breakdown

## Key decisions so far

**Patient-level leakage.** The dataset is encounter-level, and 29.7% of rows are
repeat encounters from patients seen before — one patient appears 40 times. A
random train/test split would place the same patient on both sides, letting a
model memorize individuals rather than learn risk. The analysis is restricted to
each patient's first encounter, giving 71,518 independent rows.

**Impossible outcomes.** 1,545 encounters ended in death or hospice transfer.
These patients cannot be readmitted, so they are mislabeled negatives rather
than hard cases, and are removed.

**Class prevalence.** The first-encounter population has an 8.97% positive rate,
meaningfully below the 11.16% rate across all encounters, so patients with repeat
encounters are systematically higher-risk. Metrics here are therefore not
directly comparable to published results computed on all encounters.

**Nothing is imputed.** Every column with missing values is missing for a
structural reason. `A1Cresult` (81.6% absent) and `max_glu_serum` (95.2% absent)
record tests that were never ordered and whether an A1C was ordered is the
central finding of the original Strack et al. (2014) paper. These are encoded as
explicit categories rather than imputed away.

**ICD-9 grouping.** The three diagnosis columns contain 695, 724, and 757
distinct ICD-9 codes respectively, which is too granular to learn from, since most codes
appear only a handful of times. Codes are grouped into clinical categories by
numeric range, following Strack et al. (2014).

Their nine-category scheme left 17.3% of primary diagnoses in a catch-all
`Other` bucket. Inspecting its contents showed four coherent blocks: infectious
disease (septicemia and cellulitis, ~2,500 encounters), endocrine and metabolic
disorders, blood disorders, and mental health diagnoses. Sepsis and psychiatric
comorbidity are both established readmission risk factors, so these were split
into their own categories rather than left pooled. V-codes (supplementary
factors) and E-codes (external causes) have no place on the numeric scale and
remain in `Other`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows
pip install -r requirements.txt
```

Download `diabetic_data.csv` and `IDS_mapping.csv` from the UCI link above into
`data/raw/`. The dataset is not committed to this repository.

## Layout
src/
load_data.py # loading + schema validation
clean.py # filtering, column pruning, target construction
tests/
test_clean.py
notebooks/
01_eda.ipynb

## Running tests

```bash
pytest -v
```