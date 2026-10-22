✈️ Flight Delay Prediction — NYC Flights (2013)
Team T029 · Section E · Machine Learning Project

Predict whether a flight arrives 15 minutes or more late using scheduled-flight information intended to be available before departure. This project covers exploratory data analysis (EDA), feature engineering, chronological data splitting, model comparison, evaluation, and an offline rescheduling experiment.
Important: The stated prediction cutoff is 6:00 PM on the day before departure. Each feature must be verified as available by that time. Results are specific to the dataset and notebook execution documented here; they are not a guarantee of future performance.

Team Members
#	Name	Enrollment Number
1	Aarsh Bhatnagar	2401010007
2	Anurag Kumar Tiwari	2401020009
3	Nav Prabhat Singh	2401010293
4	Ashish Kumar Yadav	2401020015
5	Shivansh Tiwari	2401020108


Project Structure
Flight_Delay_Prediction/
├── raw_data/
│   └── flights.csv                 # Local input; do not commit raw data
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── Preprocessing.ipynb         # Cleaning, feature engineering, time split
│   └── 02_Model_Application.ipynb  # Training, evaluation, outputs
├── figures/                        # Exported charts (if committed)
├── processed_flights/              # Processed split files (if generated)
├── outputs/                        # Models, metrics, predictions (if generated)
└── README.md
Filenames and output paths can differ by repository version. Check the actual repository and notebook configuration before relying on a listed file.
Dataset
The project uses the 2013 NYC flights dataset for flights associated with:
- JFK — John F. Kennedy International Airport
- LGA — LaGuardia Airport
- EWR — Newark Liberty International Airport
The project EDA reports the following counts:
Dataset detail	Reported value
Original rows	336,776
Original columns	21
Rows with missing arr_delay	9,430
Rows with an observed arrival-delay label	327,346
Delayed-flight prevalence among labelled rows	24.47%


These figures depend on the exact dataset version and cleaning steps. Recalculate them if the source data or preprocessing changes.
Target
A flight is labelled as delayed if its arrival delay is at least 15 minutes:
y = (arr_delay >= 15).astype(int)
- 1: arrival delay is 15 minutes or more.
- 0: arrival delay is less than 15 minutes.
Rows with missing arr_delay are excluded because their target cannot be determined. This classification task does not separately predict cancellations or diversions.
Prediction Cutoff and Leakage Prevention
The intended prediction time is 6:00 PM on the day before departure. Features should use only information available by that cutoff.

The project design follows these rules:
- Use chronological training, validation, and test periods.
- Exclude realized operational outcomes from the predictor matrix.
- Fit preprocessing transformations on training data only.
- Use validation data for model comparison and threshold selection.
- Reserve the test period for final evaluation.
- Verify the timing and provenance of every schedule-derived feature.
The following columns are excluded from predictors:

dep_time, dep_delay, arr_time, air_time, arr_delay, id, name, tailnum, y, date, and split.

arr_delay defines the target but must not be used as an input feature. The cutoff-time assumption must be checked against how each feature was actually constructed; a column's presence in the dataset does not prove it was available the previous evening.
Feature Engineering
The Model Application notebook documents these feature groups.
Categorical features
- carrier
- origin
- dest_grouped
- flight
Numeric and scheduled-time features
- distance
- sched_minute
- sched_arr_minute
- scheduled_origin_hour_count
- hour_sin, hour_cos
- arr_hour_sin, arr_hour_cos
- weekday_sin, weekday_cos
- year_day_sin, year_day_cos
Cyclical encodings represent periodic time patterns. scheduled_origin_hour_count represents flights scheduled from the same origin during the same scheduled hour; it is a schedule-based count, not a direct measurement of actual congestion. The categorical flight feature should be checked for sparse or unseen identifiers and possible memorization.
Chronological Data Split
Split	Period	Rows	Purpose
Training	January–August 2013	217,727	Fit preprocessing and model parameters
Validation	September–October 2013	55,628	Compare models and select the threshold
Testing	November–December 2013	53,991	Final evaluation on later observations


The reported partitions are chronological. Verify row counts and ordering after running preprocessing.
Observed delay prevalence
Split	Delayed-flight prevalence
Training	26.37%
Validation	15.70%
Testing	25.83%


The validation period has a substantially lower delay prevalence than the training and test periods. This temporal variation matters when interpreting model metrics and the selected threshold.
Models and Preprocessing
Preprocessing pipeline
The documented pipeline uses:
- Numeric features: median imputation followed by standardization.
- Categorical features: most-frequent imputation followed by one-hot encoding.
- Unknown categories: configured to be ignored by the encoder.
- Infrequent categories: handled according to the encoder's configured minimum-frequency setting.
Preprocessing should be fitted on the training partition and then applied unchanged to validation and test data.
Models compared
Model	Role
Dummy Classifier (prior)	Simple baseline using the training class distribution
Logistic Regression	Linear classification model
XGBoost Classifier	Gradient-boosted decision trees for nonlinear effects and interactions


The documented XGBoost configuration uses 400 estimators, maximum depth 5, learning rate 0.05, row subsampling 0.8, and column subsampling 0.8. Confirm these settings against the notebook if it has changed.
Model selection and threshold
Average Precision (AP) on the validation set is the primary model-selection metric. The classification threshold is selected from validation predictions by maximizing F1-score. The chosen threshold is then applied unchanged to test predictions. Test labels must not be used for model selection or threshold tuning.
Evaluation Metrics
The project reports or considers:
- Average Precision (AP): summarizes precision across recall levels and is useful for the less prevalent positive class.
- ROC-AUC: evaluates ranking discrimination between delayed and non-delayed flights.
- Precision: proportion of flights predicted delayed that are actually delayed.
- Recall: proportion of delayed flights correctly identified.
- F1-score: harmonic mean of precision and recall.
- Accuracy: proportion of correct predictions.
- Classification report: class-wise precision, recall, F1-score, and support.
- Permutation importance: estimates how shuffling a feature changes validation AP.
Recorded model results
The following metrics are the results recorded in the project report and a prior Model Application run. Re-run the notebook and update this table if the final experiment differs.

Validation comparison
Model	Validation AP	Validation ROC-AUC
Logistic Regression	0.2348	0.6446
XGBoost	0.2301	0.6368
Dummy Classifier (prior)	Verify from notebook	Verify from notebook


Selected Logistic Regression — test set
Metric	Test result
Average Precision (AP)	0.3359
ROC-AUC	0.6070
Precision	0.3415
Recall	0.4456
F1-score	0.3867
Accuracy	0.6349
Selected threshold	0.2611


These values are tied to a specific notebook execution and should not be treated as guaranteed results for every run. Confirm that the final submitted notebook uses the same data split, selected model, and threshold. Do not claim a measured improvement over the baseline until its actual score is recorded.
Average Precision lift
For a given evaluation split:
AP lift = Average Precision / positive-class prevalence
Calculate AP and prevalence on the same split before interpreting this ratio.
Main Findings and Limitations
- The reported labelled dataset has a delayed-flight prevalence of approximately 24.47%.
- Observed delay rates vary by scheduled hour, weekday, and time period.
- Validation prevalence differs substantially from training and test prevalence.
- Weather observations are not included.
- Cancellations and diversions are not modelled as separate outcomes.
- Feature availability at the stated prediction cutoff requires verification.
- Results from one year and three airports should not automatically be generalized to other years or airports.
- The offline rescheduling experiment is an analytical exercise, not a validated operational recommendation.
These findings describe associations in the data; they do not establish causation.
Running the Project
1. Get the data
Download the NYC 2013 flights dataset from the source used by your project and place the CSV at:
raw_data/flights.csv
If you use Google Colab, upload the CSV or mount the storage location expected by the notebook. The repository should not include raw data unless its licence and your course rules permit it.
2. Install dependencies
The notebooks use the following Python libraries:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
The exact dependencies may vary with notebook changes. For reproducible results, record the Python and library versions used for the final run.
3. Run notebooks in order
1. Open notebooks/01_eda.ipynb and run the EDA cells.
2. Open notebooks/Preprocessing.ipynb and run the preprocessing and chronological split cells.
3. Confirm that the generated train, validation, and test files are saved to the paths expected by the next notebook.
4. Open notebooks/02_Model_Application.ipynb and execute the cells from top to bottom.
5. Review validation model comparison, threshold selection, test metrics, plots, and saved artifacts.
6. Check that the final report and README match the results from this exact run.
The notebook paths listed above are expected project paths; update this section if the repository uses different filenames.
Generated Outputs
Depending on the configured output directory, the Model Application notebook may generate:
- flight_delay_model.joblib — selected fitted model pipeline.
- flight_delay_test_predictions.csv — test predictions and evaluation fields.
- flight_delay_test_metrics.json — final test metrics.
- validation_model_comparison.csv — validation comparison.
- Permutation feature-importance results.
- Precision–Recall curves and other evaluation plots.
Check the actual output directory and filenames before relying on these artifacts. Generated files need not be committed unless required for reproducibility or submission.
References and Resources
- Dataset: Kaggle — Flights Dataset. Use the exact dataset URL/version used by the project.
- Official delay reporting: U.S. Bureau of Transportation Statistics — Airline On-Time Statistics.
- XGBoost paper: Chen, T. and Guestrin, C. (2016), XGBoost: A Scalable Tree Boosting System. DOI: 10.1145/2939672.2939785.
- Additional paper: Gharibzadeh et al., Trends and gaps in machine learning based flight delay prediction in spatiotemporal and data-driven approaches: A systematic review. DOI: 10.1016/j.asoc.2026.115761. Verify the publication metadata before academic submission.
- Additional benchmarking study: Sustainability, 2025, article 9887. Confirm the title, authors, and relevance before citing it in academic work.
Project: Flight Delay Prediction — NYC Flights (2013)
Team: T029 · Section: E
