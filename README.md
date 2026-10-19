# ✈️ Flight Delay Prediction — NYC Flights (2013)

**Team T029 · Section E · Machine Learning Project**

Predict whether a flight arrives 15 minutes or more late using information intended to be available before departure. The project includes exploratory data analysis, preprocessing, machine learning model comparison, evaluation, and an offline rescheduling experiment.

## 👥 Team Members

| # | Name | Enrollment Number |
|---|---|---|
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

- `raw_data/flights.csv` — Original dataset; do not commit the raw file.
- `notebooks/01_eda.ipynb` — Exploratory data analysis.
- `notebooks/Preprocessing.ipynb` — Data cleaning, feature engineering, and chronological dataset preparation.
- `notebooks/02_Model_Application.ipynb` — Model training, validation-based comparison, threshold selection, test evaluation, and artifact generation.
- `figures/` — Project charts, where exported.
- `processed_flights/` — Preprocessed training, validation, and test datasets.
- `outputs/` — Generated model and evaluation artifacts, depending on the configured output directory.

Adjust filenames and output paths to match the files actually committed to the repository.

## ▶️ How to Run

1. Download the 2013 NYC flights dataset from Kaggle.
2. Place `flights.csv` in `raw_data/`, or upload it when prompted by Google Colab.
3. Open and run `notebooks/01_eda.ipynb`.
4. Open `notebooks/Preprocessing.ipynb` and execute its cells in order.
5. Open `notebooks/02_Model_Application.ipynb`.
6. Ensure the processed training, validation, and test CSV files are available at the paths expected by the notebook.
7. Execute the Model Application notebook from top to bottom.
8. Review validation comparisons, selected threshold, test metrics, plots, predictions, and saved artifacts.

### Requirements

The project uses the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- joblib

Install any missing dependencies before running the notebooks. The exact requirements depend on the cells being executed.

## 📊 Dataset Overview

The project uses the 2013 New York flights dataset covering flights associated with three airports:

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

These values come from the project's EDA output and should be recalculated if the dataset or cleaning procedure changes.

### Target Definition

A flight is labelled as delayed when its arrival delay is at least 15 minutes:

```python
y = (arr_delay >= 15).astype(int)
```

- `1` — Arrival delay of 15 minutes or more.
- `0` — Arrival delay of less than 15 minutes.

Rows with missing `arr_delay` are excluded because their target is unknown. Consequently, this classification task does not predict cancellations or diversions.

## 🧭 Prediction Rules and Leakage Prevention

The intended prediction time is **6:00 PM on the day before departure**. Model inputs should contain only information available by that cutoff.

The project follows these rules:

- Split observations chronologically rather than randomly.
- Exclude actual-operation outcomes from the predictor matrix.
- Use training data to fit models and preprocessing transformations.
- Use validation data for model comparison and threshold selection.
- Keep the test period separate for final evaluation.
- Check that schedule-derived features are available at the intended prediction time.

Forbidden predictor columns include `dep_time`, `dep_delay`, `arr_time`, `air_time`, `arr_delay`, `id`, `name`, `tailnum`, `y`, `date`, and `split`. The `arr_delay` column defines the target but must not be included in the model's input features.

## ⚙️ Feature Engineering

The Model Application notebook uses the following feature groups.

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

The cyclical encodings represent periodic time information. The `scheduled_origin_hour_count` feature represents the number of flights scheduled from the same origin during the same scheduled hour. It is a schedule-based count, not a direct measurement of actual airport congestion.

The contribution of `flight` should be interpreted carefully because flight identifiers may provide predictive information while also encouraging memorization.

## 🤖 Model Application

The Model Application stage trains and compares classification models using the preprocessed datasets.

### 1. Preprocessing pipeline

The notebook combines feature preprocessing and classification into pipelines.

- **Numerical features:** Missing values are imputed using the median, followed by standardization.
- **Categorical features:** Missing values are imputed using the most frequent value, followed by one-hot encoding.
- **Unseen categories:** The categorical encoder is configured to ignore previously unseen categories.
- **Infrequent categories:** The encoder uses the configured minimum frequency to handle infrequent categories.

Fitting preprocessing as part of the model pipeline helps apply the same learned transformations during training and prediction.

### 2. Models compared

| Model | Purpose |
|---|---|
| Dummy Classifier (`prior`) | Establishes a simple baseline using the training class distribution. |
| Logistic Regression | Provides a linear classification baseline using the engineered features. |
| XGBoost Classifier | Captures nonlinear relationships and feature interactions through boosted decision trees. |

The implemented XGBoost configuration uses 400 estimators, maximum depth 5, learning rate 0.05, subsampling 0.8, and column subsampling 0.8.

### 3. Validation-based model selection

The notebook compares the trained models using **Average Precision (PR-AUC)** on the validation dataset. The best-performing trained model is selected according to this validation criterion.

The classification threshold is then selected using validation predictions by maximizing the F1-score. This allows the final decision threshold to be determined without using test labels.

### 4. Final test evaluation

The selected model and validation-derived threshold are evaluated on the held-out November–December 2013 test period.

The test set is used for final performance reporting, not for selecting the model or optimizing its threshold.

## 📅 Chronological Data Split

| Split | Period | Rows | Purpose |
|---|---|---:|---|
| Training | January–August 2013 | 217,727 | Fit preprocessing and model parameters |
| Validation | September–October 2013 | 55,628 | Compare models and select the threshold |
| Testing | November–December 2013 | 53,991 | Final evaluation on later observations |

The chronological ordering was checked to ensure that the training period precedes validation and validation precedes testing.

### Observed delay prevalence

| Split | Delayed flights |
|---|---:|
| Training | 26.37% |
| Validation | 15.70% |
| Testing | 25.83% |

The validation period has a substantially lower delayed-flight prevalence than the training and test periods. This temporal difference should be considered when interpreting model comparisons and threshold-specific performance.

## 📏 Evaluation Metrics

The notebook evaluates model performance using the metrics implemented in its evaluation cells.

- **Average Precision (PR-AUC):** Summarizes precision–recall performance and is useful when evaluating the less prevalent positive class.
- **ROC-AUC:** Measures how well the model ranks delayed flights above non-delayed flights.
- **Precision:** Measures the proportion of flights flagged as delayed that are actually delayed.
- **Recall:** Measures the proportion of delayed flights correctly identified by the model.
- **F1-score:** Combines precision and recall into a single threshold-specific measure.
- **Accuracy:** Measures the proportion of correct predictions.
- **Classification report:** Provides class-wise precision, recall, F1-score, and support.
- **Permutation importance:** Estimates how shuffling an input feature affects validation Average Precision.

### Latest recorded Model Application results

The following results were reported by the latest Model Application notebook run. They should be regenerated if the model configuration, data, or execution environment changes.

| Metric | Test result |
|---|---:|
| Average Precision (PR-AUC) | 0.3359 |
| ROC-AUC | 0.6070 |
| Precision | 0.3415 |
| Recall | 0.4456 |
| F1-score | 0.3867 |
| Accuracy | 0.6349 |
| Selected classification threshold | 0.2611 |

The selected model in that run was Logistic Regression. Its validation Average Precision was 0.2348, compared with 0.2301 for XGBoost. The corresponding validation ROC-AUC values were 0.6446 and 0.6368, respectively.

These results are specific to the recorded notebook execution and should not be treated as guaranteed results for every run.

### PR-AUC lift

PR-AUC lift compares Average Precision with the positive-class prevalence baseline:

\[
\text{PR-AUC Lift} =
\frac{\text{Average Precision}}{\text{Positive-Class Prevalence}}
\]

Average Precision and prevalence must be calculated on the same evaluation split for a meaningful comparison.

## 🔬 Main Findings and Limitations

- Approximately 24.5% of labelled flights in the original EDA output are delayed by at least 15 minutes.
- Delay rates vary across the day and across calendar periods.
- The validation period has a substantially lower delayed-flight prevalence than the training and test periods.
- The dataset does not contain direct weather observations.
- The model does not predict cancellations or diversions.
- The model's usefulness depends on the availability and quality of the features at the stated prediction cutoff.
- Results from one year and three airports should not automatically be generalized to other years or airports.
- The offline rescheduling experiment is an analytical exercise, not a validated operational recommendation.

These are observed associations and modelling limitations, not proof of causation.

## 📦 Model Application Outputs

The Model Application notebook generates model and evaluation artifacts. Depending on the notebook's configured output directory, these include:

- `flight_delay_model.joblib` — Serialized selected model pipeline.
- `flight_delay_test_predictions.csv` — Test-set predictions and associated evaluation fields.
- `flight_delay_test_metrics.json` — Final test evaluation metrics.
- `validation_model_comparison.csv` — Validation comparison of candidate models.
- Permutation feature importance results.
- Precision–recall curve and other displayed evaluation results.

Check the actual output directory and filenames before relying on these artifacts. Generated files should only be committed if appropriate for the project's repository and submission requirements.

## 📚 Resources and References

- **Dataset:** Kaggle — Flights Dataset.
- **Official delay reporting:** U.S. Bureau of Transportation Statistics — On-Time Reporting.
- **Systematic review:** Gharibzadeh et al., *Trends and gaps in machine learning based flight delay prediction in spatiotemporal and data-driven approaches: A systematic review*, Applied Soft Computing, 2026. DOI: `10.1016/j.asoc.2026.115761`.
- **Benchmarking study:** *Machine Learning Methods Benchmarking for Predicting Flight Delays: An Efficiency Meta-Analysis*, Sustainability, 2025.

Verify publication metadata and reference details before final academic submission.

---

**Project:** Flight Delay Prediction (NYC 2013)  
**Team:** T029 · **Section:** E
