import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Page Title
# -----------------------------
st.title("Health Risk Analysis Dashboard")
st.write("This app displays dataset and health risk outputs based on life expectancy.")

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("UnifiedDataset.csv")

# -----------------------------
# Dataset Preview
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(data.head(20))

# -----------------------------
# Dataset Information
# -----------------------------
st.subheader("Dataset Information")
st.write("Number of rows:", data.shape[0])
st.write("Number of columns:", data.shape[1])
st.write("Column names:")
st.write(list(data.columns))

# -----------------------------
# Risk Mapping Function (INVERSE)
# -----------------------------
def risk_from_le(le):
    if le < 60:
        return "High"
    elif le < 70:
        return "Medium"
    else:
        return "Low"

# -----------------------------
# Create Health Risk Column
# -----------------------------
data["Health Risk Level"] = data["Life_expectancy"].apply(risk_from_le)

# -----------------------------
# Show Health Risk Output
# -----------------------------
st.subheader("Health Risk Level Output")
st.dataframe(data[["Life_expectancy", "Health Risk Level"]].head(20))

# -----------------------------
# Risk Level Distribution Plot
# -----------------------------
st.subheader("Distribution of Health Risk Levels")

risk_counts = data["Health Risk Level"].value_counts()

fig, ax = plt.subplots()
ax.bar(risk_counts.index, risk_counts.values)
ax.set_xlabel("Health Risk Level")
ax.set_ylabel("Number of Countries")
ax.set_title("Distribution of Health Risk Levels")

st.pyplot(fig)
