# ✈️ Flight Delay Prediction — NYC Flights (2013)

**Team T029 · Section E · Machine Learning Project**

Predict whether a flight arrives **15 minutes or more late**, using information intended to be available before departure, and explore an offline rescheduling rule.

## 👥 Team Members

| # | Name | Enrollment Number |
|---:|---|---|
| 1 | Aarsh Bhatnagar | `2401010007` |
| 2 | Anurag Kumar Tiwari | `2401020009` |
| 3 | Nav Prabhat Singh | `2401010293` |
| 4 | Ashish Kumar Yadav | `2401020015` |
| 5 | Shivansh Tiwari | `2401020108` |

## 📁 Folder Layout

```text
Flight_Delay_Prediction/
├── raw_data/
│   └── flights.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   └── Preprocessing.ipynb
├── figures/
├── processed_flights/
└── README.md
```

- `raw_data/flights.csv` — Original dataset; do not commit the raw file.
- `notebooks/01_eda.ipynb` — Exploratory data analysis.
- `notebooks/Preprocessing.ipynb` — Data preprocessing and model evaluation.
- `figures/` — Charts saved for the project report.
- `processed_flights/` — Processed train, validation, and test data, if exported.

*Adjust notebook filenames if they differ in the repository.*

## ▶️ How to Run

1. Download `flights.csv` from [Kaggle — Flights Dataset](https://www.kaggle.com/datasets/matinsajadi/flights).
2. Place the file in the `raw_data/` directory, or upload it when prompted by Google Colab.
3. Open `notebooks/01_eda.ipynb`.
4. Select **Runtime → Run all** in Google Colab.
5. Open the preprocessing notebook and run its cells in order.
6. Review the generated charts, model comparisons, and evaluation metrics.

### Requirements

The EDA notebook uses:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

The preprocessing notebook may also require `xgboost`, depending on the models being run.

## 📊 Dataset Overview

The project uses the **2013 New York flights dataset**, covering flights associated with three airports:

- **JFK** — John F. Kennedy International Airport
- **LGA** — LaGuardia Airport
- **EWR** — Newark Liberty International Airport

| Dataset detail | Reported value |
|---|---:|
| Original records | 336,776 |
| Original columns | 21 |
| Records with missing `arr_delay` | 9,430 |
| Labelled records retained | 327,346 |
| Delayed-flight prevalence | 24.47% |

These values come from the project’s EDA output. Recalculate them if the dataset or cleaning procedure changes.

### Target Definition

A flight is labelled as delayed if its arrival delay is **15 minutes or more**.

```python
y = (arr_delay >= 15).astype(int)
```

- `1` — Arrival delay of at least 15 minutes.
- `0` — Arrival delay of less than 15 minutes.

Rows with missing `arr_delay` are excluded because their arrival-delay target is unknown. Consequently, this project does not predict cancellations or diversions.

## 🧭 Project Rules

| Rule | Meaning |
|---|---|
| Predict before departure | The assumed prediction time is 6:00 PM on the previous day. Only information available by that cutoff should be used. |
| No peeking at the future | Historical statistics must use only information available before the prediction cutoff. |
| Split by date | Train on January–August, validate on September–October, and test on November–December 2013. |
| Prevent leakage | Exclude actual-operation outcomes from model inputs. |
| Keep the test set separate | Use validation data for model and threshold selection; reserve the test period for final evaluation. |

### Forbidden Columns

The following columns must not be used as pre-departure predictors:

```text
dep_time
dep_delay
arr_time
air_time
arr_delay
id
name
tailnum
y
date
split
```

The `arr_delay` column defines the target, but it must not be included in the predictor matrix.

## ⚙️ Feature Engineering

The current preprocessing notebook uses the following feature groups.

**Categorical features**
- `carrier`
- `origin`
- `dest_grouped`
- `flight`

**Numeric and scheduled-time features**
- `distance`
- Scheduled departure and arrival minute representations
- Cyclical hour encodings
- Cyclical weekday and calendar encodings
- `scheduled_origin_hour_count`

The scheduled origin-hour count represents the number of flights scheduled from the same origin in the same hour. It is a **schedule-based count, not a measurement of actual airport congestion**. Its validity depends on the assumption that the relevant schedule is available at the stated prediction cutoff.

The `flight` feature should also be evaluated carefully to determine whether it contributes useful predictive information rather than encouraging memorization.

## 🔬 Main Findings from the EDA

- Approximately **24.5% of labelled flights are delayed** by at least 15 minutes.
- Delay rates rise later in the day.
- Saturday is reported as the calmest day in the EDA.
- The validation period has a substantially lower delayed-flight prevalence than the training and test periods.
- The dataset does not contain direct weather observations.

The validation period was reported to have approximately **16% delayed flights**, compared with approximately **26% in the training and test periods**.

Because delayed-flight prevalence changes between periods, PR-AUC should be interpreted alongside the positive-class prevalence of the same evaluation split.

These are observed associations, not proof of causation.

## 📏 Evaluation Strategy

The project uses chronological evaluation to estimate how a model performs on later observations.

| Split | Period | Purpose |
|---|---|---|
| Training | January–August 2013 | Fit preprocessing and model parameters |
| Validation | September–October 2013 | Compare models and select the decision threshold |
| Testing | November–December 2013 | Evaluate the selected model on later observations |

The project compares a simple baseline with a more complex candidate model, including Logistic Regression and a boosted-tree approach where implemented.

### Evaluation Metrics

- **PR-AUC / Average Precision:** Summarizes the precision–recall trade-off and is useful when the positive class is less prevalent.
- **ROC-AUC:** Measures how well the model ranks delayed flights above non-delayed flights.
- **PR-AUC lift:** Compares average precision with the positive-class prevalence baseline.
- **Precision and recall:** Evaluate the trade-off between correctly flagged flights and delayed flights detected at a selected threshold.
- **F1-score and confusion matrix:** Summarize threshold-specific classification performance.

The lift calculation is:

$$
\text{PR-AUC Lift} =
\frac{\text{Average Precision}}
{\text{Positive-Class Prevalence}}
$$

Average precision and prevalence must be calculated on the **same evaluation split**.

Actual model scores should be reported from the executed notebook rather than assumed or estimated.

## ⚠️ Limitations

- **No direct weather data:** The model cannot directly learn from measured storms or weather conditions.
- **No cancellation prediction:** Flights without observed arrival delays are excluded from the binary classification task.
- **Changing delay prevalence:** Differences across months can affect evaluation metrics.
- **Feature availability:** Schedule-derived features must respect the 6 PM previous-day prediction cutoff.
- **Limited generalization:** Findings from one year and three airports should not automatically be generalized to other years or airports.
- **Offline rescheduling rule:** Any rescheduling experiment is an analytical exercise, not a validated operational recommendation.

## 📚 Resources and References

1. **Dataset:** [Kaggle — Flights Dataset](https://www.kaggle.com/datasets/matinsajadi/flights)
2. **Official delay reporting:** [U.S. Bureau of Transportation Statistics — On-Time Reporting](https://www.bts.gov/topics/airlines-and-airports/number-14-time-reporting)
3. **Systematic review:** Gharibzadeh et al., *Trends and gaps in machine learning based flight delay prediction in spatiotemporal and data-driven approaches: A systematic review*, Applied Soft Computing, 2026. [DOI: 10.1016/j.asoc.2026.115761](https://doi.org/10.1016/j.asoc.2026.115761)
4. **Benchmarking study:** *Machine Learning Methods Benchmarking for Predicting Flight Delays: An Efficiency Meta-Analysis*, Sustainability, 2025. [Read the article](https://www.mdpi.com/2071-1050/17/21/9887)

*Verify publication metadata and reference details before final academic submission.*

---

**Project:** Flight Delay Prediction (NYC 2013)  
**Team:** T029 · **Section:** E
