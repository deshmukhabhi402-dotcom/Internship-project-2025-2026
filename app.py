import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Health Risk Prediction System")

st.subheader("Enter Patient Details")

# Example input fields (adjust to match your model)
age = st.number_input("Age", min_value=0, max_value=120)
bmi = st.number_input("BMI", min_value=0.0)
bp = st.number_input("Blood Pressure", min_value=0.0)
glucose = st.number_input("Glucose Level", min_value=0.0)

# Predict button
if st.button("Predict Health Risk"):
    input_data = np.array([[age, bmi, bp, glucose]])
    prediction = model.predict(input_data)

    # Only target value shown on frontend
    st.success(f"Health Risk Level: {prediction[0]}")
    
def predict_health_risk(input_data):
    return model.predict(input_data)[0]
risk = predict_health_risk(input_data)
st.success(f"Health Risk Level: {risk}")
