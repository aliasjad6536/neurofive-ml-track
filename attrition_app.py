import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="💼", layout="centered")

st.title("Employee Attrition Predictor")
st.write(
    "This app predicts the risk that an employee will leave the company, based on their "
    "job, compensation, and satisfaction data. Built as the capstone project for the "
    "Neurofive ML Track, using the IBM HR Analytics Employee Attrition dataset."
)

@st.cache_resource
def load_pipeline():
    return joblib.load("attrition_pipeline.joblib")

pipeline = load_pipeline()

st.header("Enter Employee Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=65, value=35)
    monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=5000, step=500)
    total_working_years = st.number_input("Total Working Years", min_value=0, max_value=45, value=8)
    years_at_company = st.number_input("Years at This Company", min_value=0, max_value=40, value=5)
    years_since_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=2)
    overtime = st.selectbox("Works Overtime?", options=["Yes", "No"])

with col2:
    job_satisfaction = st.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3)
    work_life_balance = st.slider("Work-Life Balance (1=Bad, 4=Best)", 1, 4, 3)
    department = st.selectbox("Department", options=["Sales", "Research & Development", "Human Resources"])
    job_role = st.selectbox("Job Role", options=[
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    ])
    marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"])
    business_travel = st.selectbox("Business Travel", options=["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

# Derived engineered features, matching training exactly
income_per_year_worked = monthly_income / (total_working_years + 1)
promotion_stagnation = years_since_promotion / (years_at_company + 1)

st.caption(
    f"Derived features → income_per_year_worked: {income_per_year_worked:.1f}, "
    f"promotion_stagnation: {promotion_stagnation:.2f}"
)

if st.button("Predict Attrition Risk", type="primary"):
    # Fields the user doesn't enter directly are filled with dataset medians/modes,
    # matching the defaults used during training-data exploration.
    input_row = {
        "Age": age,
        "DailyRate": 802,
        "DistanceFromHome": 7,
        "Education": 3,
        "EnvironmentSatisfaction": 3,
        "HourlyRate": 66,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobSatisfaction": job_satisfaction,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": 14235,
        "NumCompaniesWorked": 2,
        "PercentSalaryHike": 14,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": 3,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": 3,
        "YearsSinceLastPromotion": years_since_promotion,
        "YearsWithCurrManager": 3,
        "income_per_year_worked": income_per_year_worked,
        "promotion_stagnation": promotion_stagnation,
        "BusinessTravel": business_travel,
        "Department": department,
        "EducationField": "Life Sciences",
        "Gender": "Male",
        "JobRole": job_role,
        "MaritalStatus": marital_status,
        "OverTime": overtime,
    }

    input_df = pd.DataFrame([input_row])
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0]

    st.divider()
    if prediction == 1:
        st.error(f"**High Attrition Risk** (probability: {probability[1]*100:.1f}%)")
        st.write("Consider a retention conversation, workload review, or compensation check-in.")
    else:
        st.success(f"**Low Attrition Risk** (probability of staying: {probability[0]*100:.1f}%)")

    st.write("Prediction probabilities:")
    st.bar_chart(pd.DataFrame({
        "Outcome": ["Stays", "Leaves"],
        "Probability": [probability[0], probability[1]]
    }).set_index("Outcome"))

st.divider()
st.caption("Neurofive ML Track — Capstone: End-to-End Machine Learning Project")
