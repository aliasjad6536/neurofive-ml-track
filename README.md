# Neurofive ML Track — Titanic, Housing & Churn Projects

This repo tracks my full progress through the Neurofive Solutions Machine Learning internship.

**Live demo apps:**
- Titanic Survival Predictor: https://aliasjad6536-neurofive-ml-track-app-sah9z0.streamlit.app/
- Employee Attrition Predictor (Capstone): https://aliasjad6536-neurofive-ml-track-attrition-app-py6ffo.streamlit.app/

---

## 1. Exploratory Data Analysis (Titanic)
Dataset: `https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`
- Loaded with `pandas.read_csv()`, inspected with `.info()`, `.describe()`, `.head()`.
- 891 rows, 12 columns. Missing values in `Age`, `Cabin`, `Embarked`. Split into numerical vs. categorical columns.

## 2. Data Cleaning & Visualization (Titanic)
- Missing values handled with justification: `Age` → median per `Pclass`; `Embarked` → mode; `Cabin` → dropped, replaced with a `has_cabin` flag.
- Outliers in `Fare` detected via boxplot (kept — real high-fare bookings, not errors).
- 4 visualizations: histogram, boxplot, bar chart, correlation heatmap.
- Conclusion: `Sex` is the strongest single driver of survival, followed by `Pclass`/`Fare`.

## 3. Predict Survival: Classification Model (Titanic)
- Features: all columns except `PassengerId`, `Name`, `Ticket`; `Sex`/`Embarked` one-hot encoded with `pd.get_dummies(drop_first=True)`.
- 80/20 stratified `train_test_split`, `LogisticRegression` (scikit-learn).
- **Accuracy: 81.01%** | Confusion matrix: `[[96, 14], [20, 49]]`.

## 4. Model Evaluation & Tuning: Beyond Accuracy (Titanic)
- Explained why accuracy alone is misleading on imbalanced data (~38% survived vs. ~62% did not).
- `GridSearchCV` (5-fold CV, F1 scoring) tuned `C` and `penalty` for `LogisticRegression`.
- **Best hyperparameters:** `C=100`, `penalty='l1'`, `solver='liblinear'` — best CV F1: 0.7323.

| Metric | Baseline (C=1.0, l2) | Tuned (GridSearchCV) | Change |
|---|---|---|---|
| Accuracy | 0.8101 | 0.8101 | 0.0000 |
| Precision (Survived) | 0.7778 | 0.7692 | -0.0085 |
| Recall (Survived) | 0.7101 | 0.7246 | +0.0145 |
| F1-score (Survived) | 0.7424 | 0.7463 | +0.0038 |

## 5. House Price Prediction with Linear Regression (California Housing)
Dataset: `https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv`
- 5 features: `median_income`, `total_rooms`, `housing_median_age`, `total_bedrooms`, `population`.
- **RMSE: $79,537.35** | **R² score: 0.5172**
- R² in plain English: roughly half the differences between cheap and expensive homes are explained by these 5 features; the rest comes from factors the model doesn't see (exact location, home condition, etc.).

## 6. Customer Churn Prediction — Working with a Business Problem (Telco Churn)
Dataset: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv` (7,043 customers)
- EDA: churn is highest for month-to-month contracts, short tenure, and higher monthly charges.
- Class imbalance flagged (~73% no-churn vs. ~27% churn).
- Compared `DecisionTreeClassifier` (79.42% accuracy) vs. `LogisticRegression` (80.62% accuracy).
- **Top 3 churn drivers:** `tenure`, `InternetService_Fiber optic`, `TotalCharges`.
- Business summary: highest-risk customers are new sign-ups on flexible, no-commitment plans — a clear target for retention offers.

## 7. Build a Proper ML Pipeline with Feature Engineering (Titanic)
- Single `Pipeline` combining a `ColumnTransformer` (`StandardScaler` on numerical columns, `OneHotEncoder` on categorical columns) chained into `LogisticRegression`.
- 2 engineered features: `family_size` (`SibSp` + `Parch` + 1) and `is_alone` (binary flag for solo travelers).
- Manual approach vs. pipeline: both landed at **81.01% accuracy** — the real win is leakage-proof, reusable preprocessing.
- Saved with `joblib.dump()` as `titanic_pipeline.joblib`; reloaded copy reproduces identical predictions. **This is the model powering the live app above.**

## 8. Ensemble Learning: Random Forest vs. XGBoost (Titanic)
Used the same engineered feature set (`family_size`, `is_alone`, `has_cabin`).

| Model | Metric | Score |
|---|---|---|
| Logistic Regression (single model) | Accuracy | 0.7989 |
| Random Forest (ensemble) | Accuracy | 0.7933 |
| XGBoost (ensemble) | Accuracy | 0.7821 |

- Feature importances: Random Forest top 3 → `Sex_male`, `Fare`, `Age`. XGBoost top 3 → `Sex_male`, `Pclass`, `has_cabin`.
- Bagging (Random Forest, parallel independent trees, reduces variance) vs. boosting (XGBoost, sequential error-correcting trees, reduces bias). On this dataset the simple Logistic Regression baseline slightly edged out both ensembles — a reminder ensembles aren't automatically better on small, low-complexity datasets.

## 9. Handling Imbalanced & Messy Real-World Data (Telco Churn)
- Confirmed and visualized the ~73%/27% class imbalance with a bar chart.
- Explained why accuracy alone is misleading: a model predicting "No Churn" for everyone would score ~73% accuracy while catching zero churners.
- Applied **SMOTE** (`imbalanced-learn`) to the training set only (test set left untouched/imbalanced for a realistic evaluation).

| Metric | Before (imbalanced) | After (SMOTE) |
|---|---|---|
| Accuracy | 0.8062 | 0.7601 |
| Precision (Churn) | 0.66 | 0.54 |
| Recall (Churn) | 0.56 | 0.63 |
| F1-score (Churn) | 0.60 | 0.58 |

- Takeaway: accuracy dropped after balancing, but recall on the class that actually matters (churners) went up — a deliberate, worthwhile trade for this business problem.

## 10. Deploy Your Model as a Live Web App
- Built a **Streamlit** app (`app.py`) that loads the saved `titanic_pipeline.joblib` and predicts survival from user-entered passenger details (class, sex, age, fare, embarkation port, family size, cabin).
- Deployed for free on **Streamlit Community Cloud**.
- **Live app:** https://aliasjad6536-neurofive-ml-track-app-sah9z0.streamlit.app/

## 11. Capstone: End-to-End Employee Attrition Prediction
A self-chosen project outside the guided track datasets — see full details in [`CAPSTONE_README.md`](./CAPSTONE_README.md) and [`CASE_STUDY.md`](./CASE_STUDY.md).
- **Problem:** predict employee attrition risk (IBM HR Analytics dataset, 1,470 employees).
- Compared Logistic Regression vs. Random Forest; Logistic Regression selected for better recall on leavers (62% vs. 32%) despite lower raw accuracy.
- **Final pipeline accuracy: 85.71%**, saved as `attrition_pipeline.joblib`, deployed as a live Streamlit app.
- **Live app:** https://aliasjad6536-neurofive-ml-track-attrition-app-py6ffo.streamlit.app/

## Files
- `titanic_eda.ipynb` — EDA, cleaning/visualization, classification model
- `model_tuning.ipynb` — Model evaluation & hyperparameter tuning
- `house_price_regression.ipynb` — House price prediction with Linear Regression
- `churn_prediction.ipynb` — Customer churn prediction (Decision Tree vs. Logistic Regression)
- `pipeline_feature_engineering.ipynb` — ML Pipeline with ColumnTransformer + feature engineering
- `titanic_pipeline.joblib` — the saved, reloadable pipeline (used by the live app)
- `ensemble_learning.ipynb` — Random Forest vs. XGBoost ensemble comparison
- `imbalanced_data.ipynb` — SMOTE-based handling of imbalanced churn data
- `app.py`, `requirements.txt` — the Streamlit web app
- `titanic.csv`, `housing.csv`, `Telco-Customer-Churn.csv` — datasets used
