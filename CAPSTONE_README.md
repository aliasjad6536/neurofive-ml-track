# Capstone: Predicting Employee Attrition

**Neurofive ML Track — Capstone: End-to-End Machine Learning Project**

## Problem Statement

Employee turnover costs companies roughly 6-9 months of an employee's salary once recruiting, onboarding, and lost productivity are counted. This project predicts which employees are at elevated risk of leaving, using job, compensation, and satisfaction data — a self-chosen problem (not the Titanic/housing tasks from earlier in the track) because it's a real, high-value HR/business use case I wanted to explore end-to-end, from raw data to a deployed app.

**Dataset:** IBM HR Analytics Employee Attrition dataset (1,470 employees, 35 columns), loaded directly from a public GitHub CSV mirror:
`https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv`

Full case study on the business value: see [`CASE_STUDY.md`](./CASE_STUDY.md).

## Approach

1. **Clean:** dropped zero-variance columns (`EmployeeCount`, `Over18`, `StandardHours`) and the non-predictive `EmployeeNumber` ID field. No missing values in the raw data.
2. **EDA:** overtime status, income, and tenure showed the clearest visible relationship with attrition — overworked, lower-paid, newer employees churn more.
3. **Feature engineering:** added `income_per_year_worked` (pay relative to career experience) and `promotion_stagnation` (years since last promotion relative to tenure) — both designed around known attrition risk patterns rather than raw columns.
4. **Pipeline:** built a `ColumnTransformer` (`StandardScaler` on numeric columns, `OneHotEncoder` on categorical columns) chained into a model.
5. **Imbalance handling:** the target is imbalanced (~84% stayed / ~16% left); applied **SMOTE** to the training data only, for fair model comparison.
6. **Model comparison:** trained and compared Logistic Regression and Random Forest, evaluated by accuracy and ROC-AUC (not accuracy alone, given the imbalance).
7. **Deployment:** saved the winning pipeline with `joblib` and built a Streamlit app around it.

## Results

| Model | Accuracy | Recall (Attrition) | ROC-AUC-selected? |
|---|---|---|---|
| Logistic Regression | 0.77 | 0.62 | **Yes — best** |
| Random Forest | 0.85 | 0.32 | No |

Random Forest scores higher on raw accuracy, but Logistic Regression catches nearly twice as many actual leavers (62% recall vs. 32%) — for an attrition-prediction tool, missing a departing employee is the costlier mistake, so Logistic Regression was selected as the final model despite its lower headline accuracy.

**Final deployed pipeline accuracy (refit on original training data): 85.71%**

**Top attrition signals found:** working overtime, lower income relative to experience, and longer promotion stagnation.

## Live App

Built with Streamlit, loading the saved `attrition_pipeline.joblib`.

**Live app:** [PASTE YOUR STREAMLIT URL HERE]

## How to Run This Project Locally

```bash
# 1. Clone the repo
git clone https://github.com/aliasjad6536/neurofive-ml-track.git
cd neurofive-ml-track

# 2. Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib streamlit

# 3. Open and run the notebook (regenerates attrition_pipeline.joblib)
jupyter notebook capstone_attrition.ipynb
# (or open it in VS Code and click "Run All")

# 4. Run the Streamlit app locally
streamlit run attrition_app.py
```

## Files

- `capstone_attrition.ipynb` — full notebook: problem → clean → EDA → feature engineering → multiple models → evaluation → best model → save
- `attrition_pipeline.joblib` — the saved, deployable pipeline (used by the app)
- `attrition_app.py` — the Streamlit app
- `CASE_STUDY.md` — half-page business case study
- `hr_attrition.csv` — the dataset used
