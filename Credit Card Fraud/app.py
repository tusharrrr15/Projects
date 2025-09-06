import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model/fraud_model.pkl")

st.title("💳 Credit Card Fraud Detection")
st.write("Upload a CSV file with transaction data to check for fraud.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", data.head())

    predictions = model.predict(data)
    data['Prediction'] = predictions
    
    st.write("Prediction Results:", data)

    fraud_cases = data[data['Prediction'] == 1]
    st.write(f"⚠️ Detected {len(fraud_cases)} Fraudulent Transactions")
