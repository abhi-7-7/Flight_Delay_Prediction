# Flight Delay Prediction — NYC Flights (2013)

**Team T029 · Section E · Machine Learning Project**

Can scheduled flight information help predict whether a flight will arrive late? This project investigates that question using the 2013 New York City flights dataset, applying exploratory data analysis, feature engineering, preprocessing, and machine-learning classification.

The project focuses on flights departing from John F. Kennedy International Airport (JFK), LaGuardia Airport (LGA), and Newark Liberty International Airport (EWR). A flight is considered delayed when it arrives at least **15 minutes after its scheduled arrival time**.

**Repository:** [Flight Delay Prediction on GitHub](https://github.com/abhi-7-7/Flight_Delay_Prediction)

> **Prediction-time assumption:** Predictions are intended to be made at 6:00 PM on the day before scheduled departure. Features must be available by this cutoff; schedule-derived features require particular care before any operational use.

## Project Overview

Flight delays affect passengers, airline schedules, airport resources, and connecting journeys. Predicting delay risk in advance could help planning, but a model is useful only if it learns from information genuinely available when the prediction is made.

This project therefore addresses two connected questions:

1. **Prediction:** Do scheduled-flight and calendar attributes contain useful information for identifying flights likely to arrive late?
2. **Evaluation:** Can that information improve predictions on a later time period without relying on future information or post-departure outcomes?

To answer these questions, the project first examines the dataset, then constructs a leakage-conscious modelling pipeline, compares multiple classifiers, and evaluates the selected model on a held-out test period. An offline rescheduling experiment is also part of the broader project, but it should be interpreted as an analytical exercise rather than a validated operational recommendation.

## Dataset and Target

The analysis uses the 2013 NYC flights dataset, which records flight and operational information for departures from JFK, LGA, and EWR.

### Dataset profile

| Dataset detail | Reported value |
|---|---:|
| Original records | 336,776 |
| Original columns | 21 |
| Missing `arr_delay` labels | 9,430 |
| Labelled records retained | 327,346 |
| Delayed-flight prevalence | 24.47% |

The target cannot be determined for records without an observed arrival delay, so those records are excluded from this classification experiment. Consequently, cancellations and diversions are not predicted as separate outcomes, and the retained sample may not represent every scheduled flight.

### Defining the prediction target

Let `arr_delay` represent the difference between actual and scheduled arrival times, in minutes. The binary target is defined as:

```python
y = (arr_delay >= 15).astype(int)
```

- `1` — arrival delay of at least 15 minutes.
- `0` — arrival delay of less than 15 minutes.

Approximately one-quarter of the labelled flights belong to the delayed class. Since the classes are imbalanced, accuracy alone may give a misleading impression of model quality. This motivates the use of Average Precision, recall, and other complementary metrics later in the evaluation.

## Exploratory Data Analysis

After defining the target, exploratory data analysis (EDA) is used to understand the distribution of delays, identify patterns that may be useful for prediction, and expose potential data-leakage risks.

The recorded analysis highlights several patterns:

- **Time of day:** Delay rates generally increase later in the day.
- **Day of the week:** Saturday has the lowest observed delay rate, while Thursday has the highest in the reported analysis.
- **Carrier:** Observed delay rates differ between airlines.
- **Class distribution:** Non-delayed flights are more common than delayed flights.

These findings motivate the use of scheduled-time and calendar features. However, they describe associations in the 2013 dataset, not causal effects; other factors, including seasonality and operational conditions, may contribute.

EDA also demonstrates why predictor availability matters. Actual departure delay is strongly associated with arrival delay, but it becomes known only after departure. It is therefore unsuitable for the intended previous-evening prediction, regardless of its predictive power.

## Methodology

The modelling workflow follows the same principle established during EDA: features must be valid for the intended prediction time, and performance must be measured on observations that were not used to develop the model.

The main stages are:

1. Clean the target labels and inspect the dataset.
2. Construct schedule-based and calendar features.
3. Divide the labelled observations chronologically.
4. Fit preprocessing and candidate models on training data.
5. Compare models and select a classification threshold using validation data.
6. Evaluate the selected model on the later, held-out test period.

### Preventing data leakage

The intended prediction cutoff is 6:00 PM on the day before departure. The recorded exclusion list includes `dep_time`, `dep_delay`, `arr_time`, `air_time`, `arr_delay`, `id`, `name`, `tailnum`, `y`, `date`, and `split`.

The exact predictor allow-list must be checked against the final notebook. In particular, target-related columns and outcomes observed after departure must not enter the model inputs. Schedule-derived variables must also be constructed using information available by the cutoff.

Chronological evaluation and training-only preprocessing are essential parts of this design. They help ensure that the reported performance reflects prediction into a later period rather than information inadvertently shared across development and evaluation.

## Feature Engineering and Preprocessing

With the prediction constraints established, feature engineering represents the available schedule and calendar information in a form suitable for classification.

### Predictor groups

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

Cyclical encoding represents recurring time patterns using sine and cosine transformations, allowing times near the end and beginning of a cycle to remain close in feature space.

The `scheduled_origin_hour_count` feature describes the number of flights scheduled for a particular origin and hour. It is a schedule-based proxy, not a direct measurement of weather, runway congestion, or real-time airport conditions. Its construction and availability at the prediction cutoff must be verified.

The categorical `flight` feature may capture recurring service patterns, but it could also encourage memorization. Testing the model without this feature would help determine whether it improves generalization.

### Preprocessing pipeline

The recorded workflow uses the following transformations:

| Feature type | Preprocessing |
|---|---|
| Numerical | Median imputation and standardization |
| Categorical | Most-frequent imputation and one-hot encoding |
| Unknown categories | Ignored by the encoder |
| Infrequent categories | Grouped according to the encoder configuration |

These transformations should be fitted on the training partition only, then applied unchanged to validation and test data. This keeps information from later observations out of the preprocessing fit.

## Experimental Design

Once the features and preprocessing pipeline are defined, the next step is to compare models under a consistent evaluation protocol.

### Chronological partitioning

The labelled data are split by calendar month rather than randomly shuffled, reflecting the goal of predicting future flights.

| Split | Period | Rows | Delayed flights | Delay rate |
|---|---|---:|---:|---:|
| Training | January–August 2013 | 217,727 | 57,423 | 26.37% |
| Validation | September–October 2013 | 55,628 | 8,731 | 15.70% |
| Testing | November–December 2013 | 53,991 | 13,946 | 25.83% |

The validation period has a substantially lower delay prevalence than the training and test periods. This variation is important because model selection and Average Precision can be affected by changes in the positive-class rate.

### Models compared

| Model | Role in the experiment |
|---|---|
| Dummy Classifier (`prior`) | Provides a simple class-prior baseline |
| Logistic Regression | Provides a regularized linear classification benchmark |
| XGBoost Classifier | Models nonlinear relationships and feature interactions using boosted trees |

The recorded XGBoost configuration uses 400 estimators, maximum depth 5, learning rate 0.05, and row and column subsampling of 0.8. These settings should be verified against the final model-application notebook.

### Model selection and threshold

Average Precision (AP) is the primary model-selection metric because the delayed class is less prevalent. ROC-AUC provides a complementary measure of ranking discrimination.

The decision threshold is selected by maximizing F1-score on validation predictions. After selection, that threshold is frozen and applied to the test predictions. The test set is not used to choose the model or tune the threshold.

## Results

The recorded validation comparison selects Logistic Regression, although its advantage over XGBoost is small.

### Validation comparison

| Model | Validation AP | Validation ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.2348 | 0.6446 |
| XGBoost | 0.2301 | 0.6368 |
| Dummy Classifier (`prior`) | 0.1570 | 0.5000 |

Logistic Regression has the highest validation Average Precision in the recorded run. This supports its selection under the stated criterion, but does not establish universal or statistically significant superiority over XGBoost.

### Held-out test performance

The selected Logistic Regression model is evaluated on the November–December test period using the validation-selected threshold of `0.2611`.

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
| Decision threshold | 0.2611 |

AP lift compares the model's Average Precision with the positive-class prevalence on the same split:

```text
AP lift = Average Precision / Positive-class prevalence
```

The recorded test AP is approximately 1.30 times the test prevalence, indicating improvement over a constant-score baseline, but only modest ranking performance.

The selected threshold identifies fewer than half of delayed flights, while many flights flagged as delayed are not delayed by the target definition. Test accuracy is also lower than the 0.7417 accuracy of an always-negative classifier. This is consistent with selecting the threshold to balance precision and recall through F1-score rather than to maximize accuracy.

These results must be interpreted together: no single metric fully describes whether the model would be useful in an operational setting.

## Interpretation and Limitations

The experiment shows that scheduled-flight and calendar information contains some predictive signal, but the measured performance is limited.

The principal limitations are:

- **No weather observations:** The model does not use direct weather measurements.
- **Missing target labels:** Cancellations and diversions are not modelled separately.
- **Temporal variation:** The validation period has a lower delay prevalence than training and testing.
- **Limited scope:** The experiment covers one year and three New York airports.
- **Feature availability:** Schedule-derived predictors must be verified against the stated cutoff.
- **Potential memorization:** The `flight` feature requires additional generalization testing.
- **Decision threshold:** F1-score is an experimental selection rule, not a demonstrated operational cost optimum.

Future work should include rolling-origin validation across multiple periods, evaluation on later years and additional airports, carefully timestamped weather information, probability calibration, and threshold selection based on explicit operational costs.

The offline rescheduling experiment should remain separate from claims about predictive performance. Without additional validation, its outputs should not be treated as operational recommendations.

## Project Structure

The project is organized around exploratory analysis, preprocessing, and model application. The structure below represents the intended layout; verify filenames and capitalization against the actual repository.

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

| Component | Purpose |
|---|---|
| EDA notebook | Dataset inspection, delay distributions, and exploratory visualizations |
| Preprocessing notebook | Data cleaning, feature engineering, and chronological splitting |
| Model-application notebook | Model comparison, threshold selection, and test evaluation |
| `figures/` | Exported analysis charts, if committed |
| `Reports/` | Project report and literature review, if committed |

## Installation and Usage

### 1. Obtain the dataset

Obtain the same version of the NYC 2013 flights dataset used in the project. Place the source file at the location expected by the notebooks, or upload it to Google Colab or mount the required storage location.

Do not commit raw data unless its licence and course requirements permit it.

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

For reproducibility, record the Python and library versions used in the final run.

### 3. Execute the notebooks

Run the notebooks in the following order:

1. `01_eda.ipynb` — inspect the dataset and explore delay patterns.
2. `Preprocessing.ipynb` — clean the data, construct features, and create chronological partitions.
3. `02_Model_Application.ipynb` — compare models, select the threshold, evaluate the test period, and generate outputs.

Execute each notebook from top to bottom. Confirm that dataset paths, output paths, split sizes, feature exclusions, selected model, threshold, and metrics agree with this README and the final report.

**Notebook links**

- EDA / preprocessing: `ADD_VERIFIED_COLAB_LINK`
- Model application: `ADD_VERIFIED_COLAB_LINK`

Replace these placeholders with the actual notebook URLs and ensure the sharing permissions allow your evaluator to open them.

### Generated outputs

Depending on notebook configuration, the modelling stage may generate a serialized model, test predictions, evaluation metrics, model-comparison records, permutation-importance results, and plots. Verify the actual filenames and output locations before relying on these artifacts.

## References

- **NYC flights dataset:** [Kaggle — Flights](https://www.kaggle.com/datasets/matinsajadi/flights). Confirm that this is the exact source and version used.
- **NYC flights reference package:** [tidyverse/nycflights13](https://github.com/tidyverse/nycflights13).
- **Airline on-time statistics:** [U.S. Bureau of Transportation Statistics](https://www.bts.gov/topics/airlines-and-airports/number-14-time-reporting).
- **XGBoost:** Chen, T. and Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* [https://doi.org/10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).
- **Flight-delay prediction review:** Gharibzadeh et al., *Trends and gaps in machine learning based flight delay prediction in spatiotemporal and data-driven approaches: A systematic review.* [https://doi.org/10.1016/j.asoc.2026.115761](https://doi.org/10.1016/j.asoc.2026.115761). Verify publication metadata before academic submission.
- **Benchmarking study:** [Sustainability, 2025, article 9887](https://www.mdpi.com/2071-1050/17/21/9887). Verify the full title, authors, and relevance before citing it academically.

## Team Members

| No. | Name | Enrollment Number |
|---:|---|---|
| 1 | Aarsh Bhatnagar | 2401010007 |
| 2 | Anurag Kumar Tiwari | 2401020009 |
| 3 | Nav Prabhat Singh | 2401010293 |
| 4 | Ashish Kumar Yadav | 2401020015 |
| 5 | Shivansh Tiwari | 2401020108 |

---

**Project:** Flight Delay Prediction — NYC Flights (2013)  
**Team:** T029 · **Section:** E
