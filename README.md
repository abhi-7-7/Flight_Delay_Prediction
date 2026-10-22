# Flight Delay Prediction — NYC Flights (2013)

**Team T029 · Section E · Machine Learning Project**

A machine-learning project that predicts whether a flight departing from New York City's JFK, LaGuardia (LGA), or Newark Liberty (EWR) airports will arrive at least **15 minutes late**. The project covers exploratory data analysis (EDA), feature engineering, preprocessing, chronological model evaluation, and an offline rescheduling experiment.

**Repository:** [Flight Delay Prediction](https://github.com/abhi-7-7/Flight_Delay_Prediction)

> **Prediction cutoff:** 6:00 PM on the day before scheduled departure. Every predictor must be available by this cutoff for the model to be suitable for the intended prediction task.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Results](#key-results)
- [Team Members](#team-members)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Features and Preprocessing](#features-and-preprocessing)
- [Evaluation Metrics](#evaluation-metrics)
- [Key Findings and Limitations](#key-findings-and-limitations)
- [Project Structure](#project-structure)
- [Installation and Usage](#installation-and-usage)
- [References](#references)

## Project Overview

Flight delays affect passengers, airline schedules, airport operations, and connecting journeys. This project investigates whether scheduled-flight and calendar information can help identify flights likely to arrive at least 15 minutes late.

The workflow includes:

1. Exploratory data analysis and visualization.
2. Target-label cleaning and feature engineering.
3. Leakage-aware preprocessing.
4. Chronological training, validation, and test splits.
5. Comparison of a baseline classifier, Logistic Regression, and XGBoost.
6. Model selection using validation Average Precision.
7. Decision-threshold selection using validation F1-score.
8. Evaluation on a held-out November–December test period.

The project is an experimental machine-learning study, not a validated operational flight-dispatch or rescheduling system.

## Key Results

The following results come from the project's recorded model-application run. Re-run the final notebook before submission to confirm that the values remain unchanged.

| Item | Result |
|---|---:|
| Original dataset records | 336,776 |
| Labelled flights retained | 327,346 |
| Overall delayed-flight prevalence | 24.47% |
| Selected model | Logistic Regression |
| Validation Average Precision | 0.2348 |
| Test Average Precision | 0.3359 |
| Test ROC-AUC | 0.6070 |
| Test precision | 0.3415 |
| Test recall | 0.4456 |
| Test F1-score | 0.3867 |
| Selected classification threshold | 0.2611 |

Logistic Regression was selected because its validation Average Precision was slightly higher than XGBoost's. The test results indicate **limited but measurable ranking ability**; they do not demonstrate high-reliability prediction.

## Team Members

| No. | Name | Enrollment Number |
|---:|---|---|
| 1 | Aarsh Bhatnagar | 2401010007 |
| 2 | Anurag Kumar Tiwari | 2401020009 |
| 3 | Nav Prabhat Singh | 2401010293 |
| 4 | Ashish Kumar Yadav | 2401020015 |
| 5 | Shivansh Tiwari | 2401020108 |

## Dataset

The project uses the NYC flights dataset for 2013, covering departures from three airports:

- **JFK:** John F. Kennedy International Airport
- **LGA:** LaGuardia Airport
- **EWR:** Newark Liberty International Airport

### Dataset statistics

| Dataset detail | Reported value |
|---|---:|
| Original rows | 336,776 |
| Original columns | 21 |
| Rows with missing `arr_delay` | 9,430 |
| Labelled rows retained | 327,346 |
| Delayed-flight prevalence | 24.47% |

Flights without an observed `arr_delay` value are excluded because their target label cannot be determined. Cancellations and diversions are not modelled as separate outcomes.

### Target definition

A flight is labelled delayed if its arrival delay is at least 15 minutes:

```python
y = (arr_delay >= 15).astype(int)
```

- `1`: Arrival delay of 15 minutes or more.
- `0`: Arrival delay of less than 15 minutes.

## Methodology

### 1. Prediction-time constraints and leakage prevention

The intended forecast is made at 6:00 PM on the day before scheduled departure. Features must therefore reflect information genuinely available by that time.

The workflow is designed to:

- Use chronological rather than randomly shuffled partitions.
- Exclude post-departure outcomes from model predictors.
- Fit preprocessing transformations on training data only.
- Use validation data for model comparison and threshold selection.
- Keep the test period separate from model-selection decisions.
- Verify the availability and construction of schedule-derived features.

The recorded feature-exclusion list includes `dep_time`, `dep_delay`, `arr_time`, `air_time`, `arr_delay`, `id`, `name`, `tailnum`, `y`, `date`, and `split`. The exact feature allow-list should be checked against the final notebook.

### 2. Feature engineering

**Categorical features**

- `carrier`
- `origin`
- `dest_grouped`
- `flight`

**Numerical and schedule-derived features**

- `distance`
- `sched_minute`
- `sched_arr_minute`
- `scheduled_origin_hour_count`
- `hour_sin`, `hour_cos`
- `arr_hour_sin`, `arr_hour_cos`
- `weekday_sin`, `weekday_cos`
- `year_day_sin`, `year_day_cos`

Cyclical encodings represent repeating patterns in hours, weekdays, and days of the year. The `scheduled_origin_hour_count` feature represents a schedule-based count, not a direct measurement of real-time airport congestion.

The `flight` feature may capture recurring service patterns, but its contribution to generalization should be assessed through additional testing.

### 3. Chronological data split

| Split | Period | Rows | Delayed flights | Delay rate |
|---|---|---:|---:|---:|
| Training | January–August 2013 | 217,727 | 57,423 | 26.37% |
| Validation | September–October 2013 | 55,628 | 8,731 | 15.70% |
| Testing | November–December 2013 | 53,991 | 13,946 | 25.83% |

The validation period has a substantially lower delay prevalence than the training and test periods. This temporal variation can affect model selection, threshold selection, and comparisons of Average Precision.

### 4. Preprocessing

The documented preprocessing pipeline uses:

- **Numerical features:** Median imputation and standardization.
- **Categorical features:** Most-frequent imputation and one-hot encoding.
- **Unknown categories:** Ignored by the encoder.
- **Infrequent categories:** Grouped according to the encoder configuration.

Preprocessing is fitted on training data and then applied consistently to validation and test data.

### 5. Models compared

| Model | Purpose |
|---|---|
| Dummy Classifier (`prior`) | Establishes a class-prior baseline |
| Logistic Regression | Provides a regularized linear classification benchmark |
| XGBoost Classifier | Models nonlinear relationships and feature interactions using boosted trees |

The recorded XGBoost configuration uses 400 estimators, maximum depth 5, learning rate 0.05, row subsampling 0.8, and column subsampling 0.8. Confirm these settings against the final notebook.

### 6. Model selection and threshold

Average Precision (AP) on the validation set is the primary model-selection metric. The decision threshold is chosen by maximizing validation F1-score, then applied unchanged to the test predictions.

| Model | Validation AP | Validation ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.2348 | 0.6446 |
| XGBoost | 0.2301 | 0.6368 |
| Dummy Classifier (`prior`) | 0.1570 | 0.5000 |

Logistic Regression leads XGBoost by a small margin. This does not establish that it is universally superior or that the difference is statistically significant.

## Evaluation Metrics

The project reports:

- **Average Precision (AP):** Summarizes precision across recall increments.
- **ROC-AUC:** Measures ranking discrimination across classification thresholds.
- **Precision:** Fraction of flights flagged as delayed that are actually delayed.
- **Recall:** Fraction of actual delayed flights identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Accuracy:** Fraction of correctly classified observations.
- **Permutation importance:** Measures how shuffling a feature affects a validation metric.

### Held-out test performance

The selected Logistic Regression model was evaluated on the November–December test set at the validation-selected threshold of `0.2611`.

| Metric | Test result |
|---|---:|
| Average Precision | 0.3359 |
| AP lift over test prevalence | 1.30× |
| ROC-AUC | 0.6070 |
| Precision | 0.3415 |
| Recall | 0.4456 |
| F1-score | 0.3867 |
| Accuracy | 0.6349 |
| Test prevalence | 0.2583 |

AP lift is calculated as:

```text
AP lift = Test Average Precision / Test positive-class prevalence
```

For this run, the model's AP is approximately 1.30 times the test prevalence.

**Important interpretation:** Test accuracy is 0.6349, below the 0.7417 accuracy of an always-negative classifier. The selected threshold prioritizes a precision–recall trade-off through F1-score rather than maximizing accuracy. The appropriate threshold for real operations would depend on the relative costs of missed delays and false alerts.

## Key Findings and Limitations

### Findings

- Approximately 24.47% of labelled flights in the reported analysis are delayed by at least 15 minutes.
- Observed delay rates vary by scheduled departure hour, weekday, and carrier.
- Validation delay prevalence is substantially lower than training and test prevalence.
- Logistic Regression slightly outperforms XGBoost on the recorded validation Average Precision.

### Limitations

- **No weather observations:** The model cannot use measured weather conditions.
- **Missing target labels:** Cancellations and diversions are not separately predicted.
- **Temporal drift:** The evaluation covers one year and one chronological split.
- **Feature availability:** Schedule-derived inputs must be verified against the 6:00 PM cutoff.
- **Potential feature memorization:** The `flight` feature requires further generalization testing.
- **Threshold choice:** F1-score is an experimental selection rule, not a proven operational cost optimum.
- **Limited scope:** Results from three New York airports in 2013 should not automatically be generalized to other locations or years.

The offline rescheduling experiment is an analytical exercise. It should not be interpreted as a validated operational recommendation.

## Project Structure

The following is the intended logical structure. Confirm directory names and filenames against the current repository before relying on these paths.

```text
Flight_Delay_Prediction/
├── Datasets/
│   ├── flight_delay_project_outputs.zip
│   └── flights.zip
├── Notebooks/
│   ├── 01_eda.ipynb
│   ├── Preprocessing.ipynb
│   └── 02_Model_Application.ipynb
├── Reports/
│   ├── Flight_Delay_Prediction_Report.pdf
│   └── Literature_Review_Flight_Delay.pdf
├── figures/
├── .gitignore
└── README.md
```

| Component | Description |
|---|---|
| EDA notebook | Dataset inspection, target distribution, and exploratory visualizations |
| Preprocessing notebook | Data cleaning, feature engineering, and chronological splitting |
| Model-application notebook | Model comparison, threshold selection, and test evaluation |
| `figures/` | Charts exported by the analysis, if committed |
| `Reports/` | Project report and literature review, if committed |

## Installation and Usage

### 1. Obtain the dataset

Obtain the same version of the NYC 2013 flights dataset used by the project. Place the source file at the path expected by the notebooks.

In Google Colab, upload the dataset or mount the storage location expected by the notebook.

Do not commit raw data unless its licence and your course requirements permit it.

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

The exact package versions should be recorded for a reproducible final run.

### 3. Run the notebooks

Run the notebooks in this order:

1. `01_eda.ipynb` — inspect the dataset and generate exploratory analysis.
2. `Preprocessing.ipynb` — clean the data, engineer features, and create chronological partitions.
3. `02_Model_Application.ipynb` — compare models, select the threshold, evaluate the test set, and generate outputs.

Restart the runtime when appropriate and execute the notebooks from top to bottom. Verify that data paths, output paths, split sizes, feature exclusions, model selection, threshold, and metrics match this README and the final report.

**Notebook links:** Add the two verified Google Colab URLs here, and make sure the sharing permissions allow your evaluator to open them.

- EDA / preprocessing notebook: `ADD_VERIFIED_COLAB_LINK`
- Model-application notebook: `ADD_VERIFIED_COLAB_LINK`

### Generated outputs

Depending on notebook configuration, the model-application stage may generate a serialized model, test predictions, metric summaries, model-comparison results, permutation-importance results, and evaluation plots.

Check the actual output directory and filenames before documenting or using these artifacts.

## References

- **Dataset:** [NYC flights dataset on Kaggle](https://www.kaggle.com/datasets/matinsajadi/flights) — confirm that this is the exact source and version used in the project.
- **NYC flights reference package:** [tidyverse/nycflights13](https://github.com/tidyverse/nycflights13).
- **Airline on-time statistics:** [U.S. Bureau of Transportation Statistics](https://www.bts.gov/topics/airlines-and-airports/number-14-time-reporting).
- **XGBoost:** Chen, T. and Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* [https://doi.org/10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).
- **Flight-delay prediction review:** Gharibzadeh et al., *Trends and gaps in machine learning based flight delay prediction in spatiotemporal and data-driven approaches: A systematic review.* [https://doi.org/10.1016/j.asoc.2026.115761](https://doi.org/10.1016/j.asoc.2026.115761). Verify the final publication metadata before academic submission.
- **Benchmarking study:** [Sustainability, 2025, article 9887](https://www.mdpi.com/2071-1050/17/21/9887). Verify the full title, authors, and relevance before citing it academically.

---

**Project:** Flight Delay Prediction — NYC Flights (2013)  
**Team:** T029 · **Section:** E
