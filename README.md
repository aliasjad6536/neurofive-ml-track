# Neurofive ML Track — Titanic, Housing & Churn Projects

This repo tracks my progress through the Neurofive Solutions Machine Learning internship.

## Week 1 — Exploratory Data Analysis (Titanic)
Dataset: `https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`
- Loaded with `pandas.read_csv()`, inspected with `.info()`, `.describe()`, `.head()`.
- 891 rows, 12 columns. Missing values in `Age`, `Cabin`, `Embarked`. Split into numerical vs. categorical columns.

## Week 2 — Data Cleaning & Visualization (Titanic)
- Missing values handled with justification: `Age` → median per `Pclass`; `Embarked` → mode; `Cabin` → dropped, replaced with a `has_cabin` flag.
- Outliers in `Fare` detected via boxplot (kept — real high-fare bookings, not errors).
- 4 visualizations: histogram, boxplot, bar chart, correlation heatmap.
- Conclusion: `Sex` is the strongest single driver of survival, followed by `Pclass`/`Fare`.

## Week 3 — Predict Survival: Classification Model (Titanic)
- Features: all columns except `PassengerId`, `Name`, `Ticket`; `Sex`/`Embarked` one-hot encoded with `pd.get_dummies(drop_first=True)`.
- 80/20 stratified `train_test_split`, `LogisticRegression` (scikit-learn).
- **Baseline accuracy: 81.01%**
- Confusion matrix: `[[96, 14], [20, 49]]`.

## Model Evaluation & Tuning: Beyond Accuracy (Titanic)
- Explained why accuracy alone is misleading on imbalanced data (~38% survived vs. ~62% did not).
- `GridSearchCV` (5-fold CV, F1 scoring) tuned `C` and `penalty` for `LogisticRegression`.
- **Best hyperparameters:** `C=100`, `penalty='l1'`, `solver='liblinear'` — best CV F1: 0.7323.

| Metric | Baseline (C=1.0, l2) | Tuned (GridSearchCV) | Change |
|---|---|---|---|
| Accuracy | 0.8101 | 0.8101 | 0.0000 |
| Precision (Survived) | 0.7778 | 0.7692 | -0.0085 |
| Recall (Survived) | 0.7101 | 0.7246 | +0.0145 |
| F1-score (Survived) | 0.7424 | 0.7463 | +0.0038 |

## Week 4 — House Price Prediction with Linear Regression (California Housing)
Dataset: `https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv`
- 5 features: `median_income`, `total_rooms`, `housing_median_age`, `total_bedrooms`, `population`.
- **RMSE: $79,537.35** | **R² score: 0.5172**

## Customer Churn Prediction — Working with a Business Problem (Telco Churn)
Dataset: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv` (7,043 customers)
- EDA: churn is highest for month-to-month contracts, short tenure, and higher monthly charges.
- Class imbalance flagged (~73% no-churn vs. ~27% churn), noted rather than resampled.
- Compared `DecisionTreeClassifier` (79.42% accuracy) vs. `LogisticRegression` (80.62% accuracy).
- **Top 3 churn drivers (Decision Tree feature importances):** `tenure`, `InternetService_Fiber optic`, `TotalCharges`.
- Business summary written for a non-technical manager: highest-risk customers are new sign-ups on flexible, no-commitment plans — a clear target for retention offers.

## Build a Proper ML Pipeline with Feature Engineering (Titanic)
- Built a single `Pipeline` combining a `ColumnTransformer` (`StandardScaler` on numerical columns, `OneHotEncoder` on categorical columns) chained into `LogisticRegression`.
- Added 2 engineered features: `family_size` (`SibSp` + `Parch` + 1) and `is_alone` (binary flag for solo travelers).
- Manual approach vs. pipeline: both landed at **81.01% accuracy** — the win is leakage-proof, reusable preprocessing, not a raw accuracy jump.
- Saved the fitted pipeline with `joblib.dump()` as `titanic_pipeline.joblib`; confirmed a reloaded copy reproduces identical predictions.

## Ensemble Learning: Random Forest vs. XGBoost (Titanic)
Used the same engineered feature set (`family_size`, `is_alone`, `has_cabin`) as the pipeline task.

**Comparison table (model, metric, score):**

| Model | Metric | Score |
|---|---|---|
| Logistic Regression (single model) | Accuracy | 0.7989 |
| Random Forest (ensemble) | Accuracy | 0.7933 |
| XGBoost (ensemble) | Accuracy | 0.7821 |

- **Feature importances:** Random Forest top 3 → `Sex_male`, `Fare`, `Age`. XGBoost top 3 → `Sex_male`, `Pclass`, `has_cabin`. Both agree `Sex_male` is the strongest signal; they diverge further down the ranking.
- **Bagging vs. boosting:** Random Forest builds many trees independently in parallel on bootstrapped samples and averages their votes (bagging, reduces variance). XGBoost builds trees sequentially, each one correcting the previous ensemble's errors (boosting, reduces bias). On this dataset, the simple Logistic Regression baseline actually edged out both ensembles slightly — a reminder that ensembles aren't automatically better on small, low-complexity datasets like this one.

## Files
- `titanic_eda.ipynb` — Weeks 1–3 (EDA, cleaning/visualization, classification model)
- `model_tuning.ipynb` — Model evaluation & hyperparameter tuning
- `house_price_regression.ipynb` — House price prediction with Linear Regression
- `churn_prediction.ipynb` — Customer churn prediction (Decision Tree vs. Logistic Regression)
- `pipeline_feature_engineering.ipynb` — ML Pipeline with ColumnTransformer + feature engineering
- `titanic_pipeline.joblib` — the saved, reloadable pipeline
- `ensemble_learning.ipynb` — Random Forest vs. XGBoost ensemble comparison
- `titanic.csv`, `housing.csv`, `Telco-Customer-Churn.csv` — datasets used
