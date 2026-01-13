# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

st.set_page_config(page_title="Health Risk Level Analysis", layout="wide")
st.title("📊 Health Risk Level Analysis")

st.markdown("""
### Risk Definition (from notebook)
- **High Risk**: Life Expectancy < 60  
- **Medium Risk**: 60 ≤ Life Expectancy < 70  
- **Low Risk**: Life Expectancy ≥ 70  
""")

# -----------------------------
# Risk mapping (same as notebook)
# -----------------------------
def risk_level(le):
    if pd.isna(le):
        return np.nan
    if le < 60:
        return "High"
    elif le < 70:
        return "Medium"
    else:
        return "Low"

# -----------------------------
# Upload data
# -----------------------------
uploaded_file = st.file_uploader(
    "📁 Upload CSV (must contain Life Expectancy + predictions)",
    type=["csv"]
)

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
st.success(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

with st.expander("Preview Dataset"):
    st.dataframe(df.head(), use_container_width=True)

# -----------------------------
# Detect Life Expectancy column
# -----------------------------
life_col = None
for col in df.columns:
    c = col.lower().replace(" ", "")
    if "life" in c and "expect" in c:
        life_col = col
        break

if life_col is None:
    st.error("Life Expectancy column not found.")
    st.write(df.columns.tolist())
    st.stop()

st.success(f"Detected Life Expectancy column → {life_col}")

# -----------------------------
# Detect prediction column
# -----------------------------
pred_col = None
for col in df.columns:
    c = col.lower()
    if "pred" in c:
        pred_col = col
        break

if pred_col is None:
    st.error("Prediction column not found (expected name with 'pred').")
    st.stop()

st.success(f"Detected Prediction column → {pred_col}")

# -----------------------------
# Create risk labels
# -----------------------------
df["Actual_Risk"] = df[life_col].apply(risk_level)
df["Predicted_Risk"] = df[pred_col].apply(risk_level)

mask = df["Actual_Risk"].notna() & df["Predicted_Risk"].notna()
df_eval = df.loc[mask]

# -----------------------------
# Risk distribution bar chart
# -----------------------------
st.header("📊 Risk Level Distribution")

def bar_plot(series, title):
    counts = series.value_counts().reindex(["Low", "Medium", "High"], fill_value=0)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)

    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold")

    st.pyplot(fig)

col1, col2 = st.columns(2)
with col1:
    bar_plot(df_eval["Actual_Risk"], "Actual Risk Distribution")
with col2:
    bar_plot(df_eval["Predicted_Risk"], "Predicted Risk Distribution")

# -----------------------------
# Actual vs Predicted comparison
# -----------------------------
st.header("📈 Actual vs Predicted Risk Comparison")

comparison = pd.crosstab(
    df_eval["Actual_Risk"],
    df_eval["Predicted_Risk"]
).reindex(index=["Low", "Medium", "High"], columns=["Low", "Medium", "High"], fill_value=0)

st.dataframe(comparison, use_container_width=True)

# -----------------------------
# Classification Report (TABLE)
# -----------------------------
st.header("📋 Classification Report (Tabular)")

report_dict = classification_report(
    df_eval["Actual_Risk"],
    df_eval["Predicted_Risk"],
    labels=["Low", "Medium", "High"],
    output_dict=True
)

report_df = (
    pd.DataFrame(report_dict)
    .transpose()
    .reset_index()
    .rename(columns={"index": "Class"})
)

st.dataframe(
    report_df.style.format({
        "precision": "{:.3f}",
        "recall": "{:.3f}",
        "f1-score": "{:.3f}",
        "accuracy": "{:.3f}"
    }),
    use_container_width=True
)

# -----------------------------
# Key metrics
# -----------------------------
st.subheader("📌 Key Metrics")

accuracy = report_dict["accuracy"]
macro_f1 = report_dict["macro avg"]["f1-score"]

m1, m2 = st.columns(2)
m1.metric("Accuracy", f"{accuracy:.2%}")
m2.metric("Macro Avg F1-score", f"{macro_f1:.3f}")

# -----------------------------
# Sample output
# -----------------------------
st.header("🔍 Sample Data (Actual vs Predicted)")

st.dataframe(
    df_eval[[life_col, pred_col, "Actual_Risk", "Predicted_Risk"]].head(15),
    use_container_width=True
)
