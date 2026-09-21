# Flight Delay Prediction (NYC 2013)

Predict whether a flight arrives 15+ minutes late, using only information known
**before departure**, then explore an offline rescheduling rule.

## Folder layout

```
raw_data/flights.csv        <- put the Kaggle file here (not committed)
notebooks/01_eda.ipynb      <- Phase 1: exploratory data analysis (this file)
figures/                    <- charts saved by the notebook, used in the report
```

## How to Run

1. Download `flights.csv` from https://www.kaggle.com/datasets/matinsajadi/flights
2. Put it in `raw_data/`
3. Open `notebooks/01_eda.ipynb` and choose **Run All** (about 15 seconds)

Requires: `pandas numpy matplotlib seaborn scikit-learn`

## Project rules

| Rule | Meaning |
|---|---|
| Predict before departure | Prediction time is 6 PM the day before. Only the schedule is known. |
| No peeking at the future | Historical statistics use earlier days only. |
| Split by date | Train Jan-Aug, validation Sep-Oct, test Nov-Dec. |

## Forbidden columns (known only after departure)

`dep_time`, `dep_delay`, `arr_time`, `air_time`. The target is `arr_delay >= 15`.

## Main findings from the EDA

- 1 flight in 4 is delayed (24.5%), so accuracy is not used. We report PR AUC, lift, ROC AUC and recall at fixed precision.
- Delay rises through the day, and Saturday is calmest.
- The validation months are much calmer (16% delayed) than train and test (26%), so PR AUC is not comparable across splits. We report **lift** (PR AUC divided by base rate).
- No weather data, so storm days cannot be predicted. This caps model performance.
