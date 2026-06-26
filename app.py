import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load('churn_model.pkl')


st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")

st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details below to predict if they are likely to churn.")

# --- Collect User Input ---
# NOTE: These should match the features used during model training
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=800.0)

# Example categorical inputs
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
senior_citizen = st.radio("Senior Citizen", ["No", "Yes"])

# --- Preprocess User Input ---
# This must match the preprocessing done during training
input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0,
    "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
    "InternetService_No": 1 if internet_service == "No" else 0,
    "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
    "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,
    "PaymentMethod_Bank transfer": 1 if payment_method == "Bank transfer" else 0,
    "PaymentMethod_Credit card": 1 if payment_method == "Credit card" else 0,
}

feature_cols = joblib.load('feature_columns.pkl')
# Create a dictionary initialized to 0 for all features to avoid fragmentation
encoded_dict = {col: 0 for col in feature_cols}
for col, val in input_dict.items():
    if col in encoded_dict:
        encoded_dict[col] = val

input_df = pd.DataFrame([encoded_dict])
input_df = input_df[feature_cols]


# --- Prediction ---
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This customer is **likely to churn** (Probability: {probability:.2f})")
    else:
        st.success(f"✅ This customer is **not likely to churn** (Probability: {probability:.2f})")
