# ✈️ Flight Delay Prediction — NYC Flights (2013)

**Team T029 · Section E · Machine Learning Project**

Predict whether a flight arrives **15 minutes or more late** using scheduled-flight information intended to be available before departure. This project covers exploratory data analysis (EDA), feature engineering, chronological data splitting, model comparison, evaluation, and an offline rescheduling experiment.

> **Prediction cutoff:** 6:00 PM on the day before departure. Feature availability at this cutoff must be verified before operational use.

## 👥 Team Members

| # | Name | Enrollment Number |
|---:|---|---|
| 1 | Aarsh Bhatnagar | 2401010007 |
| 2 | Anurag Kumar Tiwari | 2401020009 |
| 3 | Nav Prabhat Singh | 2401010293 |
| 4 | Ashish Kumar Yadav | 2401020015 |
| 5 | Shivansh Tiwari | 2401020108 |

## 📁 Project Structure

```text
Flight_Delay_Prediction/
├── raw_data/
│   └── flights.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── Preprocessing.ipynb
│   └── 02_Model_Application.ipynb
├── figures/
├── processed_flights/
├── outputs/
└── README.md
```

*Note: Adjust filenames and directories to match the files actually present in the repository.*

| Directory/File | Description |
|---|---|
| `raw_data/` | Original dataset; raw data should not be committed unless permitted |
| `notebooks/01_eda.ipynb` | Exploratory data analysis |
| `notebooks/Preprocessing.ipynb` | Data cleaning, feature engineering, and chronological splitting |
| `notebooks/02_Model_Application.ipynb` | Model training, validation, threshold selection, and test evaluation |
| `figures/` | Exported project charts |
| `processed_flights/` | Processed datasets, if generated |
| `outputs/` | Model files, metrics, predictions, and evaluation artifacts, if generated |

## 📊 Dataset Overview

The project uses the **2013 NYC flights dataset**, covering flights associated with three airports:

- **JFK** — John F. Kennedy International Airport
- **LGA** — LaGuardia Airport
- **EWR** — Newark Liberty International Airport

### Dataset statistics

| Dataset detail | Reported value |
|---|---:|
| Original rows | 336,776 |
| Original columns | 21 |
| Missing `arr_delay` labels | 9,430 |
| Labelled rows retained | 327,346 |
| Delayed-flight prevalence | 24.47% |

These statistics should be recalculated if the dataset version or preprocessing procedure changes.

### Target Definition

A flight is labelled as delayed if its arrival delay is at least 15 minutes.

```python
y = (arr_delay >= 15).astype(int)
```

- `1` — Arrival delay of 15 minutes or more.
- `0` — Arrival delay of less than 15 minutes.

Rows with missing `arr_delay` are excluded because their target cannot be determined. Consequently, this project does not separately predict flight cancellations or diversions.

## 🧭 Prediction Cutoff and Data Leakage Prevention

The intended prediction time is **6:00 PM on the day before departure**. Features must contain only information available by that cutoff.

The project follows these principles:

- Split observations chronologically rather than randomly.
- Exclude actual operational outcomes from model inputs.
- Fit preprocessing transformations on training data only.
- Use validation data for model comparison and threshold selection.
- Reserve the test period for final evaluation.
- Verify the availability of every schedule-derived feature at prediction time.

The following columns are excluded from the predictor matrix:

`dep_time`, `dep_delay`, `arr_time`, `air_time`, `arr_delay`, `id`, `name`, `tailnum`, `y`, `date`, and `split`.

The `arr_delay` column defines the target but must not be included as a predictor. The actual construction and availability of every feature must be verified; a column's presence in the dataset does not prove it was available the previous evening.

## ⚙️ Feature Engineering

### Categorical features

- `carrier`
- `origin`
- `dest_grouped`
- `flight`

### Numerical and scheduled-time features

- `distance`
- `sched_minute`
- `sched_arr_minute`
- `scheduled_origin_hour_count`
- `hour_sin`, `hour_cos`
- `arr_hour_sin`, `arr_hour_cos`
- `weekday_sin`, `weekday_cos`
- `year_day_sin`, `year_day_cos`

Cyclical encodings represent recurring time patterns. The `scheduled_origin_hour_count` feature represents flights scheduled from the same origin during the same scheduled hour. It is a schedule-based count, not a direct measurement of actual airport congestion.

The categorical `flight` feature should be checked for sparse or unseen identifiers and potential memorization.

## 📅 Chronological Data Split

| Split | Period | Rows | Purpose |
|---|---|---:|---|
| Training | January–August 2013 | 217,727 | Fit preprocessing and model parameters |
| Validation | September–October 2013 | 55,628 | Compare models and select the threshold |
| Testing | November–December 2013 | 53,991 | Final evaluation on later observations |

### Observed delay prevalence

| Split | Delayed-flight prevalence |
|---|---:|
| Training | 26.37% |
| Validation | 15.70% |
| Testing | 25.83% |

The validation period has a substantially lower delay prevalence than the training and test periods. This temporal variation should be considered when interpreting model performance and threshold-specific results.

## 🤖 Models and Preprocessing

### Preprocessing

The documented preprocessing pipeline uses:

- **Numerical features:** Median imputation followed by standardization.
- **Categorical features:** Most-frequent imputation followed by one-hot encoding.
- **Unknown categories:** Configured to be ignored by the encoder.
- **Infrequent categories:** Handled according to the encoder's configured minimum-frequency setting.

Preprocessing should be fitted on the training partition and then applied consistently to validation and test data.

### Models compared

| Model | Purpose |
|---|---|
| Dummy Classifier (`prior`) | Establishes a simple baseline using the training class distribution |
| Logistic Regression | Provides a linear classification baseline |
| XGBoost Classifier | Captures nonlinear relationships and feature interactions using boosted decision trees |

The documented XGBoost configuration uses 400 estimators, maximum depth 5, learning rate 0.05, row subsampling 0.8, and column subsampling 0.8. Verify these settings against the final notebook.

### Model selection and threshold

Average Precision (AP) on the validation dataset is the primary model-selection metric. The classification threshold is selected from validation predictions by maximizing the F1-score.

The selected threshold is then applied unchanged to test predictions. The test set must not be used for model selection or threshold tuning.

## 📏 Evaluation Metrics

The project uses the following evaluation metrics:

- **Average Precision (AP):** Summarizes precision across recall levels and is useful when evaluating the less prevalent positive class.
- **ROC-AUC:** Measures how well the model ranks delayed flights above non-delayed flights.
- **Precision:** Proportion of flights flagged as delayed that are actually delayed.
- **Recall:** Proportion of delayed flights correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Accuracy:** Proportion of correct predictions.
- **Classification report:** Provides class-wise precision, recall, F1-score, and support.
- **Permutation importance:** Estimates how shuffling a feature affects validation Average Precision.

### Validation model comparison

| Model | Validation AP | Validation ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.2348 | 0.6446 |
| XGBoost | 0.2301 | 0.6368 |
| Dummy Classifier (`prior`) | Verify from notebook | Verify from notebook |

Logistic Regression performed slightly better than XGBoost on the reported validation metrics. The difference is small and does not establish statistical superiority.

### Recorded test results

The following results are from a previously recorded Model Application run. Regenerate them from the final notebook if the model, data, or configuration changes.

| Metric | Test result |
|---|---:|
| Average Precision (AP) | 0.3359 |
| ROC-AUC | 0.6070 |
| Precision | 0.3415 |
| Recall | 0.4456 |
| F1-score | 0.3867 |
| Accuracy | 0.6349 |
| Selected threshold | 0.2611 |

These results are specific to the recorded execution and are not guaranteed for every run. Verify that the final notebook uses the same split, selected model, and threshold before submission.

The Dummy Classifier's actual validation scores should be recorded before making quantified claims about improvement over the baseline.

### Average Precision lift

Average Precision lift compares model AP with the positive-class prevalence on the same evaluation split:

```text
AP lift = Average Precision / Positive-class prevalence
```

Both values must be calculated on the same split for a meaningful comparison.

## 🔬 Main Findings and Limitations

- Approximately 24.47% of labelled flights in the reported EDA are delayed by at least 15 minutes.
- Observed delay rates vary by scheduled hour, weekday, and calendar period.
- Validation delay prevalence differs substantially from training and test prevalence.
- Weather observations are not included.
- Cancellations and diversions are not modelled as separate outcomes.
- Feature availability at the stated prediction cutoff requires verification.
- Results from one year and three airports should not automatically be generalized to other years or airports.
- The offline rescheduling experiment is an analytical exercise, not a validated operational recommendation.

These findings describe observed associations and limitations; they do not establish causation.

## 📦 Generated Outputs

Depending on the notebook's configured output directory, the Model Application stage may generate:

- `flight_delay_model.joblib` — Serialized selected model pipeline.
- `flight_delay_test_predictions.csv` — Test predictions and associated evaluation fields.
- `flight_delay_test_metrics.json` — Final test evaluation metrics.
- `validation_model_comparison.csv` — Validation comparison of candidate models.
- Permutation feature-importance results.
- Precision–Recall curves and other evaluation plots.

Check the actual output directory and filenames before relying on these artifacts. Generated files should only be committed when appropriate for the project's repository and submission requirements.

## ▶️ How to Run

### 1. Obtain the dataset

Download the NYC 2013 flights dataset from the source used by the project and place the CSV file at:

```text
raw_data/flights.csv
```

If using Google Colab, upload the CSV or mount the storage location expected by the notebook.

Do not commit raw data unless permitted by its licence and your course requirements.

### 2. Install dependencies

The notebooks use the following Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

The exact dependencies may vary with notebook changes. Record the Python and library versions used for the final run to improve reproducibility.

### 3. Run the notebooks in order

1. Open `notebooks/01_eda.ipynb` and run the EDA cells.
2. Open `notebooks/Preprocessing.ipynb` and run the preprocessing and chronological-split cells.
3. Confirm that the training, validation, and test files are saved at the paths expected by the next notebook.
4. Open `notebooks/02_Model_Application.ipynb` and execute the cells from top to bottom.
5. Review validation model comparison, threshold selection, test metrics, plots, and saved artifacts.
6. Ensure the final README and project report match the results from the final notebook run.

The paths and filenames above are expected project paths; update them if the repository uses different names.

## 📚 References and Resources

- **Dataset:** Kaggle — Flights Dataset. Add the exact dataset URL and version used by the project.
- **Official delay reporting:** [U.S. Bureau of Transportation Statistics — Airline On-Time Statistics](https://www.bts.gov/topics/airlines-and-airports/number-14-time-reporting).
- **XGBoost paper:** Chen, T. and Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* [DOI: 10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).
- **Additional systematic review:** Gharibzadeh et al., *Trends and gaps in machine learning based flight delay prediction in spatiotemporal and data-driven approaches: A systematic review.* [DOI: 10.1016/j.asoc.2026.115761](https://doi.org/10.1016/j.asoc.2026.115761). Verify publication metadata before academic submission.
- **Additional benchmarking study:** [Sustainability, 2025, article 9887](https://www.mdpi.com/2071-1050/17/21/9887). Verify the title, authors, and relevance before citing it in academic work.

---

**Project:** Flight Delay Prediction — NYC Flights (2013)  
**Team:** T029 · **Section:** E
