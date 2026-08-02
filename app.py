import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("Titanic Survival Predictor")
st.write(
    "This app uses a Logistic Regression pipeline (trained on the Titanic dataset, "
    "with engineered `family_size` and `is_alone` features) to predict whether a "
    "passenger would have survived. Built as part of the Neurofive ML Track."
)

# Load the saved pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load("titanic_pipeline.joblib")

pipeline = load_pipeline()

st.header("Enter Passenger Details")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", options=[1, 2, 3], index=2,
                           help="1 = First class, 2 = Second class, 3 = Third class")
    sex = st.selectbox("Sex", options=["male", "female"])
    age = st.slider("Age", min_value=0, max_value=100, value=30)
    embarked = st.selectbox("Port of Embarkation", options=["S", "C", "Q"],
                             help="S = Southampton, C = Cherbourg, Q = Queenstown")

with col2:
    fare = st.number_input("Fare Paid ($)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
    siblings_spouses = st.number_input("Siblings / Spouses Aboard", min_value=0, max_value=10, value=0)
    parents_children = st.number_input("Parents / Children Aboard", min_value=0, max_value=10, value=0)
    has_cabin = st.selectbox("Had a Recorded Cabin?", options=["Yes", "No"], index=1)

# Derive engineered features exactly as done in training
family_size = siblings_spouses + parents_children + 1
is_alone = 1 if family_size == 1 else 0
has_cabin_val = 1 if has_cabin == "Yes" else 0

st.caption(f"Derived features → family_size: {family_size}, is_alone: {is_alone}")

if st.button("Predict", type="primary"):
    input_df = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "Fare": fare,
        "Embarked": embarked,
        "family_size": family_size,
        "is_alone": is_alone,
        "has_cabin": has_cabin_val,
    }])

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0]

    st.divider()
    if prediction == 1:
        st.success(f"**Prediction: Survived** (confidence: {probability[1]*100:.1f}%)")
    else:
        st.error(f"**Prediction: Did Not Survive** (confidence: {probability[0]*100:.1f}%)")

    st.write("Prediction probabilities:")
    st.bar_chart(pd.DataFrame({
        "Outcome": ["Did Not Survive", "Survived"],
        "Probability": [probability[0], probability[1]]
    }).set_index("Outcome"))

st.divider()
st.caption("Neurofive ML Track — Deploy Your Model as a Live Web App")
