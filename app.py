import streamlit as st
import pandas as pd

st.title("Health Risk Analysis Dashboard")
st.write("This app displays dataset and model outputs.")
# Load dataset
data = pd.read_csv("UnifiedDataset.csv")

st.subheader("Dataset Preview")
st.dataframe(data.head(20))   # shows first 20 rows
st.subheader("Dataset Information")
st.write("Number of rows:", data.shape[0])
st.write("Number of columns:", data.shape[1])
st.write("Column names:")
st.write(list(data.columns))
# Load predictions
y_test = pd.read_csv("y_test.csv")["Life_expectancy"]
y_pred = pd.read_csv("y_pred.csv")["Predicted_LE"]

output_df = pd.DataFrame({
    "Actual Life Expectancy": y_test,
    "Predicted Life Expectancy": y_pred
})

st.subheader("Model Output: Life Expectancy Prediction")
st.dataframe(output_df.head(20))
def risk_from_le(le):
    if le < 60:
        return "High"
    elif le < 70:
        return "Medium"
    else:
        return "Low"

output_df["Health Risk Level"] = output_df["Actual Life Expectancy"].apply(risk_from_le)

st.subheader("Health Risk Level Output")
st.dataframe(output_df[["Actual Life Expectancy", "Health Risk Level"]].head(20))
import matplotlib.pyplot as plt

risk_counts = output_df["Health Risk Level"].value_counts()

fig, ax = plt.subplots()
ax.bar(risk_counts.index, risk_counts.values)
ax.set_xlabel("Health Risk Level")
ax.set_ylabel("Count")
ax.set_title("Distribution of Health Risk Levels")

st.pyplot(fig)
