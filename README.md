# Flight Delay Prediction — NYC Flights (2013)

**Team T029 | Section E | Machine Learning Project**

This project predicts whether a flight will arrive at least 15 minutes late using the NYC flights dataset from 2013. It covers exploratory data analysis, feature engineering, data preprocessing, machine learning model comparison, evaluation, and an offline rescheduling experiment.

**GitHub Repository:** [Flight_Delay_Prediction](https://github.com/abhi-7-7/Flight_Delay_Prediction)

**Prediction cutoff:** 6:00 PM on the day before departure. Feature availability at this cutoff must be verified before operational use.

## Team Members

| No. | Name | Enrollment Number |
|---:|---|---|
| 1 | Aarsh Bhatnagar | 2401010007 |
| 2 | Anurag Kumar Tiwari | 2401020009 |
| 3 | Nav Prabhat Singh | 2401010293 |
| 4 | Ashish Kumar Yadav | 2401020015 |
| 5 | Shivansh Tiwari | 2401020108 |

## Project Structure

The project is organized around exploratory analysis, preprocessing, model application, and generated artifacts. Verify the filenames against the current repository.

```text
Flight_Delay_Prediction/
├── Datasets/
│   ├── flight_delay_project_outputs.zip
│   └── flights.zip
├── Notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_Model_Application.ipynb
│   └── Preprocessing.ipynb
├── Reports/
│   ├── Flight_Delay_Prediction_Report.pdf
│   └── Literature_Review_Flight_Delay.pdf
├── figures/
│   ├── 01_cancellations.png
│   ├── 02_time_patterns.png
│   ├── 03_daily_timeline.png
│   ├── 04_carrier_origin.png
│   └── 05_feature_signal.png
├── .gitignore
└── README.md
```


| File or directory | Description |
|---|---|
| `notebooks/01_eda.ipynb` | Exploratory data analysis and visualizations |
| `notebooks/Preprocessing.ipynb` | Data cleaning, feature engineering, and chronological splitting |
| `notebooks/02_Model_Application.ipynb` | Model training, validation, threshold selection, and evaluation |
| `figures/` | Exported charts, if included in the repository |
| `processed_flights/` | Processed datasets, if generated |
| `outputs/` | Model files, predictions, metrics, and other generated artifacts, if saved |
| `README.md` | Project documentation |

## Dataset Overview

The project uses the 2013 NYC flights dataset, covering flights associated with three airports:

- **JFK:** John F. Kennedy International Airport
- **LGA:** LaGuardia Airport
- **EWR:** Newark Liberty International Airport

### Dataset Statistics

| Dataset detail | Reported value |
|---|---:|
| Original rows | 336,776 |
| Original columns | 21 |
| Rows with missing `arr_delay` | 9,430 |
| Labelled rows retained | 327,346 |
| Delayed-flight prevalence | 24.47% |

These values are based on the project's reported exploratory analysis and should be recalculated if the dataset or preprocessing changes.

### Target Definition

A flight is classified as delayed if its arrival delay is at least 15 minutes.

```python
y = (arr_delay >= 15).astype(int)
```

- `1`: Arrival delay of 15 minutes or more.
- `0`: Arrival delay of less than 15 minutes.

Rows with missing arrival-delay labels are excluded because their target cannot be determined. Cancellations and diversions are not modelled as separate outcomes.

## Prediction Cutoff and Data Leakage Prevention

The intended prediction time is 6:00 PM on the day before departure. Model inputs must contain only information available by that cutoff.

The project follows these principles:

- Use chronological rather than random data splitting.
- Exclude actual operational outcomes from predictor features.
- Fit preprocessing transformations on training data only.
- Use validation data for model comparison and threshold selection.
- Reserve the test period for final evaluation.
- Verify that schedule-derived features are available at prediction time.

The following columns are excluded from predictors:

`dep_time`, `dep_delay`, `arr_time`, `air_time`, `arr_delay`, `id`, `name`, `tailnum`, `y`, `date`, and `split`.

The `arr_delay` column defines the target and must not be included in the model inputs. The construction and availability of every feature must be checked independently.

## Feature Engineering

### Categorical Features

- `carrier`
- `origin`
- `dest_grouped`
- `flight`

### Numerical and Scheduled-Time Features

- `distance`
- `sched_minute`
- `sched_arr_minute`
- `scheduled_origin_hour_count`
- `hour_sin`, `hour_cos`
- `arr_hour_sin`, `arr_hour_cos`
- `weekday_sin`, `weekday_cos`
- `year_day_sin`, `year_day_cos`

Cyclical encodings represent recurring time patterns. The `scheduled_origin_hour_count` feature represents flights scheduled from the same origin during the same scheduled hour. It is a schedule-based count, not a direct measurement of actual congestion.

The categorical `flight` feature should be evaluated carefully because sparse or unseen identifiers may limit generalization.

## Chronological Data Split

The labelled dataset is divided chronologically to simulate prediction on later observations.

| Split | Period | Rows | Purpose |
|---|---|---:|---|
| Training | January–August 2013 | 217,727 | Fit preprocessing and model parameters |
| Validation | September–October 2013 | 55,628 | Compare models and select the threshold |
| Testing | November–December 2013 | 53,991 | Final evaluation |

### Delay Prevalence by Split

| Split | Delayed-flight prevalence |
|---|---:|
| Training | 26.37% |
| Validation | 15.70% |
| Testing | 25.83% |

The validation period has a substantially lower delay prevalence than the training and test periods. This temporal variation is important when interpreting model performance.

## Models and Preprocessing

### Preprocessing

The documented preprocessing pipeline uses:

- **Numerical features:** Median imputation followed by standardization.
- **Categorical features:** Most-frequent imputation followed by one-hot encoding.
- **Unknown categories:** Configured to be ignored by the encoder.
- **Infrequent categories:** Handled according to the encoder's configured minimum-frequency setting.

Preprocessing should be fitted on the training partition and applied consistently to validation and test data.

### Models Compared

| Model | Purpose |
|---|---|
| Dummy Classifier (`prior`) | Establishes a baseline using the training class distribution |
| Logistic Regression | Provides a linear classification model |
| XGBoost Classifier | Models nonlinear relationships and feature interactions using boosted decision trees |

The documented XGBoost configuration uses 400 estimators, maximum depth 5, learning rate 0.05, row subsampling 0.8, and column subsampling 0.8. These settings should be verified against the final notebook.

### Model Selection and Threshold

Average Precision (AP) on the validation dataset is the primary model-selection metric. The classification threshold is selected using validation predictions by maximizing the F1-score.

The selected threshold is applied unchanged to the test predictions. The test set must not be used for model selection or threshold optimization.

## Evaluation Metrics

The project uses the following metrics:

- **Average Precision (AP):** Summarizes precision across recall levels.
- **ROC-AUC:** Measures ranking discrimination between delayed and non-delayed flights.
- **Precision:** Proportion of flights flagged as delayed that are actually delayed.
- **Recall:** Proportion of actual delayed flights correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Accuracy:** Proportion of correct predictions.
- **Classification report:** Provides class-wise precision, recall, F1-score, and support.
- **Permutation feature importance:** Estimates how shuffling a feature affects validation Average Precision.

### Validation Model Comparison

| Model | Validation AP | Validation ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.2348 | 0.6446 |
| XGBoost | 0.2301 | 0.6368 |
| Dummy Classifier (`prior`) | Verify from notebook | Verify from notebook |

Logistic Regression performed slightly better than XGBoost on the recorded validation metrics. This small difference does not establish statistical superiority.

### Recorded Test Results

The following results are from a previously recorded model run. Verify them against the final notebook before submission.

| Metric | Test result |
|---|---:|
| Average Precision | 0.3359 |
| ROC-AUC | 0.6070 |
| Precision | 0.3415 |
| Recall | 0.4456 |
| F1-score | 0.3867 |
| Accuracy | 0.6349 |
| Selected threshold | 0.2611 |

These values are specific to the recorded execution. The final report and README should use the same selected model, threshold, and evaluation results as the final notebook.

### Average Precision Lift

Average Precision lift compares the model's AP with the positive-class prevalence on the same evaluation split.

```text
AP lift = Average Precision / Positive-class prevalence
```

Both quantities must be calculated on the same split for a meaningful comparison.

## Main Findings and Limitations

- Approximately 24.47% of labelled flights in the reported analysis are delayed by at least 15 minutes.
- Delay rates vary by scheduled hour, weekday, and calendar period.
- Validation delay prevalence differs substantially from training and test prevalence.
- Weather observations are not included.
- Cancellations and diversions are not separately modelled.
- Feature availability at the intended prediction cutoff requires verification.
- Results from one year and three airports should not automatically be generalized to other years or airports.
- The offline rescheduling experiment is an analytical exercise, not a validated operational recommendation.

These findings describe observed associations and limitations; they do not establish causation.

## Generated Outputs

Depending on the notebook configuration, the model application stage may generate:

- `flight_delay_model.joblib` — Serialized selected model pipeline.
- `flight_delay_test_predictions.csv` — Test predictions and associated evaluation fields.
- `flight_delay_test_metrics.json` — Final test metrics.
- `validation_model_comparison.csv` — Validation comparison.
- Permutation feature-importance results.
- Precision–Recall curves and other evaluation plots.

Check the actual output directory and filenames before relying on these artifacts.

## How to Run the Project

### 1. Obtain the Dataset

Download the NYC 2013 flights dataset from the source used by the project. Place the CSV at the location expected by the preprocessing notebook.

If using Google Colab, upload the CSV or mount the storage location expected by the notebook.

Do not commit raw data unless permitted by its licence and your course requirements.

### 2. Install Dependencies

The project uses the following Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

The exact dependencies may vary with notebook changes. Record the Python and library versions used for the final run to improve reproducibility.

### 3. Run the Notebooks

1. Open `notebooks/01_eda.ipynb` and run the exploratory analysis.
2. Open `notebooks/Preprocessing.ipynb` and execute the cleaning, feature-engineering, and chronological-split cells.
3. Confirm that the processed datasets are saved at the locations expected by the model application notebook.
4. Open `notebooks/02_Model_Application.ipynb` and run the cells from top to bottom.
5. Review the validation comparison, selected threshold, test metrics, plots, and generated artifacts.
6. Ensure the final README and project report match the results of the final notebook run.

Update the notebook names and paths above if the repository uses different ones.

## References and Resources

- **Dataset:** Add the exact NYC flights dataset source URL and version used by the project.
- **Official delay reporting:** [U.S. Bureau of Transportation Statistics — Airline On-Time Statistics](https://www.bts.gov/topics/airlines-and-airports/number-14-time-reporting).
- **XGBoost paper:** Chen, T. and Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* [DOI: 10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).
- **Additional systematic review:** Gharibzadeh et al., *Trends and gaps in machine learning based flight delay prediction in spatiotemporal and data-driven approaches: A systematic review.* [DOI: 10.1016/j.asoc.2026.115761](https://doi.org/10.1016/j.asoc.2026.115761). Verify publication metadata before academic submission.
- **Additional benchmarking study:** [Sustainability, 2025, article 9887](https://www.mdpi.com/2071-1050/17/21/9887). Verify the title, authors, and relevance before citing it academically.

---

**Project:** Flight Delay Prediction — NYC Flights (2013)  
**Team:** T029 | **Section:** E
