# 30-Day Hospital Readmission Prediction

Predicting 30-day readmission for diabetic inpatients using the
[UCI Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
(101,766 encounters, 130 hospitals, 1999–2008).

Readmission within 30 days is the window used by the CMS Hospital Readmissions
Reduction Program, which penalizes hospitals for excess readmissions so the
target maps to a real operational and financial decision.

## Headline results

Out-of-fold performance, LightGBM, 5-fold patient-grouped cross-validation on
99,343 encounters (11.4% positive):

| Metric | Value | Baseline | Lift |
|---|---|---|---|
| AUROC | 0.680 | 0.500 | — |
| AUPRC | 0.237 | 0.114 | 2.08x |
| Precision @ top 10% | 0.285 | 0.114 | 2.51x |
| Brier score | 0.0955 | 0.1009 | — |

**Operationally:** if a care management team can follow up with 10% of
discharges, targeting by this model finds a group where 28.5% readmit, versus
11.4% selecting at random, 2.5x more readmissions caught per outreach call.

The model separates a 3.7% risk decile from a 28.5% risk decile out-of-fold,
and risk increases monotonically across all ten deciles.

![Readmission rate by risk decile](reports/figures/decile_lift.png)

## Risk factors

![SHAP summary](reports/figures/shap_summary.png)

The five dominant drivers account for roughly 48% of total attributed model
impact:

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | Discharge disposition | 0.202 |
| 2 | Prior inpatient visits | 0.171 |
| 3 | Combined prior acute care | 0.109 |
| 4 | Primary diagnosis group | 0.084 |
| 5 | Admitting specialty | 0.077 |

**Prior utilization dominates.** Readmission rate rises monotonically with
prior inpatient stays, 8.6% at zero, 13.3% at one, 17.9% at two, 31.4% at four
or more. Patients with four or more prior stays readmit at **3.7x** the rate of
those with none.

**Discharge destination is the single strongest feature.** Discharge to
inpatient rehab carries a 27.7% readmission rate versus 9.3% for discharge
home, 3x higher. This is a transition point where an intervention could
plausibly be placed.

**Age matters less than expected**, ranking 12th. What predicts readmission is
not how old a patient is but how much acute care they have recently needed.

**Payer code ranks 6th**, and its predictive power likely reflects access
barriers and socioeconomic position rather than clinical risk. 

## Calibration

![Calibration curve](reports/figures/calibration.png)

Predicted probabilities are accurate to within roughly one percentage point
across the full range, a patient scored 0.25 readmits about 25% of the time.
This is makes the output usable for capacity planning rather than ranking
alone.

No resampling (SMOTE) or class weighting was used. Both improve apparent recall
at a fixed threshold but distort predicted probabilities; calibrated output was
judged more valuable than a better default-threshold confusion matrix.

## Subgroup performance

![AUROC by age band](reports/figures/auroc_by_age.png)

Calibration is stable across race, sex, and age, no subgroup gap exceeds 0.6
percentage points, so no group is systematically over or under-scored.

Discrimination is less uniform. AUROC falls from 0.76 in patients under 40 to
**0.63 in patients 75 and older**, the group with both the highest readmission
prevalence (12.5%) and the largest sample (19,023). The model ranks least
effectively where risk is greatest, likely because utilization-history features
saturate in elderly patients, and because readmission in that population
depends on factors absent from administrative data (frailty, functional status,
caregiver availability).

A deployed targeting policy should allocate follow up capacity within age strata
rather than on a single pooled threshold.

## Methodology notes

**Patient-level leakage was tested, not assumed.** 29.7% of rows are repeat
encounters; one patient appears 40 times. Standard practice is to restrict to
first encounters, on the theory that a random split lets a model memorize
individual patients. Running both a naive random split and a patient-grouped
split on identical data produced **identical results** (AUROC 0.6792 both ways),
and the gap remained zero even with a deliberately over-parameterized model
(1000 trees, 255 leaves, `min_child_samples=1`).

The features are too coarse to fingerprint individuals, a 10-year age bucket,
a 14-category diagnosis group, and a handful of counts are shared by thousands
of patients. Leakage of this kind requires either a fine-grained representation
or target-derived features; neither is present here. The encounter-level
analysis is therefore used as primary, with patient-grouped CV retained as the
correct default.

**Impossible outcomes removed.** 2,423 encounters ended in death or hospice
transfer. These patients cannot be readmitted, making them mislabeled negatives
rather than hard cases.

**Nothing is imputed.** Every column with missing values is missing for a
structural reason. `A1Cresult` (81.6% absent) and `max_glu_serum` (95.2%)
record tests that were never ordered and whether an A1C was ordered is the
central finding of Strack et al. (2014). These are encoded as explicit
categories.

**ICD-9 grouping extended.** 695/724/757 distinct codes across the three
diagnosis columns are grouped by numeric range following Strack et al. Their
nine-category scheme left 17.3% of primary diagnoses in a catch-all `Other`
bucket; inspecting its contents revealed four coherent blocks: infectious
disease, endocrine/metabolic, blood, and mental health. Since sepsis and
psychiatric comorbidity are both established readmission risk factors, these
were split out, reducing `Other` to 9.1%.

**Model comparison.** Logistic regression reaches AUROC 0.646; tuned LightGBM
reaches 0.680. The modest gap suggests the signal is largely additive.
Hyperparameters were selected by randomized search over six parameters
(30 candidates, 5-fold, scoring on average precision).

## Limitations

- **Data vintage.** 1999–2008, ICD-9, pre-ACA. A model trained here would not
  transfer to a current patient population without retraining.
- **Performance ceiling.** Published results on this dataset cluster at
  0.65–0.70 AUROC. Administrative billing data does not contain the factors
  that most plausibly drive readmission: medication adherence, social support,
  housing stability, and follow-up appointment access.
- **Outcome definition.** Late readmissions (>30 days, 35% of encounters) are
  collapsed into the negative class. This matches the CMS penalty window but
  means the negative class mixes genuinely stable patients with later returns.
- **Elderly discrimination gap.** See subgroup section above.

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows
pip install -r requirements.txt
pytest -v
```

Download `diabetic_data.csv` and `IDS_mapping.csv` from the UCI link above into
`data/raw/`. The dataset is not committed.

## Layout

```
src/
  load_data.py   # loading + schema validation
  clean.py       # row filtering, variance pruning, target construction
  icd9.py        # ICD-9 code grouping
  features.py    # feature engineering, preprocessing, splits
  models.py      # logistic regression, LightGBM
  evaluate.py    # AUROC, AUPRC, Brier, precision@k
tests/           # 24 tests
notebooks/
  01_eda.ipynb
reports/figures/
```