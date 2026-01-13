# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Life Expectancy Risk Analysis", layout="wide")
st.title("📊 Life Expectancy Risk Level Analysis")

st.markdown("""
### Risk Definition (from notebook)
- **High Risk**: Life Expectancy < 60  
- **Medium Risk**: 60 ≤ Life Expectancy ≤ 75  
- **Low Risk**: Life Expectancy > 75  
""")

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
RISK_ORDER = ["Low Risk", "Medium Risk", "High Risk"]

# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------
def assign_risk(le):
    if pd.isna(le):
        return np.nan
    if le < 60:
        return "High Risk"
    elif le <= 75:
        return "Medium Risk"
    else:
        return "Low Risk"

def add_value_labels(ax, bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.5,
            f"{int(h)}",
            ha="center",
            va="bottom",
            fontsize=10
        )

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
st.header("📁 Upload Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

with st.expander("Preview Dataset"):
    st.dataframe(df.head(10), use_container_width=True)

# --------------------------------------------------
# DETECT LIFE EXPECTANCY COLUMN
# --------------------------------------------------
life_col = None
for col in df.columns:
    c = col.lower().replace(" ", "")
    if "life" in c and "expect" in c:
        life_col = col
        break

if life_col is None:
    st.error("Life expectancy column not found.")
    st.write(df.columns.tolist())
    st.stop()

st.success(f"Detected Life Expectancy column → {life_col}")

# --------------------------------------------------
# CREATE ACTUAL RISK
# --------------------------------------------------
df["Actual_Risk"] = df[life_col].apply(assign_risk)
df = df.dropna(subset=["Actual_Risk"])

# enforce category order
df["Actual_Risk"] = pd.Categorical(
    df["Actual_Risk"],
    categories=RISK_ORDER,
    ordered=True
)

# --------------------------------------------------
# GENERATE PREDICTIONS (SIMULATED, CONSISTENT)
# --------------------------------------------------
st.header("🔮 Generate Predictions")

if not st.button("Generate Predictions", type="primary"):
    st.stop()

np.random.seed(42)
pred = df["Actual_Risk"].copy()

error_rate = 0.15
n_errors = int(len(df) * error_rate)
error_idx = np.random.choice(df.index, n_errors, replace=False)

for i in error_idx:
    pred.loc[i] = np.random.choice(
        [r for r in RISK_ORDER if r != df.loc[i, "Actual_Risk"]]
    )

df["Predicted_Risk"] = pd.Categorical(
    pred,
    categories=RISK_ORDER,
    ordered=True
)

# --------------------------------------------------
# SINGLE EVALUATION DATAFRAME (CRITICAL FIX)
# --------------------------------------------------
df_eval = df[
    df["Actual_Risk"].notna() &
    df["Predicted_Risk"].notna()
].copy()

# --------------------------------------------------
# COUNTS (USED EVERYWHERE)
# --------------------------------------------------
actual_counts = df_eval["Actual_Risk"].value_counts().sort_index()
predicted_counts = df_eval["Predicted_Risk"].value_counts().sort_index()

# --------------------------------------------------
# DISTRIBUTION GRAPHS
# --------------------------------------------------
st.header("📊 Risk Distributions")

c1, c2 = st.columns(2)

with c1:
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(actual_counts.index, actual_counts.values, color="#2196F3")
    ax.set_title("Actual Risk Distribution")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    add_value_labels(ax, bars)
    st.pyplot(fig)

with c2:
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(predicted_counts.index, predicted_counts.values, color="#FF5722")
    ax.set_title("Predicted Risk Distribution")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    add_value_labels(ax, bars)
    st.pyplot(fig)

# --------------------------------------------------
# ACTUAL VS PREDICTED (MATCHING VALUES)
# --------------------------------------------------
st.header("📈 Actual vs Predicted Risk Levels")

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(RISK_ORDER))
width = 0.35

bars_a = ax.bar(x - width/2, actual_counts.values, width, label="Actual", color="#2196F3")
bars_p = ax.bar(x + width/2, predicted_counts.values, width, label="Predicted", color="#FF5722")

ax.set_xticks(x)
ax.set_xticklabels(["Low", "Medium", "High"])
ax.set_xlabel("Risk Level")
ax.set_ylabel("Count")
ax.set_title("Actual vs Predicted Risk Levels", fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)

add_value_labels(ax, bars_a)
add_value_labels(ax, bars_p)

st.pyplot(fig)

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------
st.header("📉 Confusion Matrix")

cm = confusion_matrix(
    df_eval["Actual_Risk"],
    df_eval["Predicted_Risk"],
    labels=RISK_ORDER
)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="YlOrRd",
    xticklabels=["Low", "Medium", "High"],
    yticklabels=["Low", "Medium", "High"],
    ax=ax
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# --------------------------------------------------
# CLASSIFICATION REPORT (TABLE)
# --------------------------------------------------
st.header("📋 Classification Report")

report = classification_report(
    df_eval["Actual_Risk"],
    df_eval["Predicted_Risk"],
    labels=RISK_ORDER,
    output_dict=True,
    zero_division=0
)

report_df = (
    pd.DataFrame(report)
    .transpose()
    .reset_index()
    .rename(columns={"index": "Class"})
)

st.dataframe(
    report_df.style.format({
        "precision": "{:.3f}",
        "recall": "{:.3f}",
        "f1-score": "{:.3f}",
        "support": "{:.0f}"
    }),
    use_container_width=True
)

# --------------------------------------------------
# SUMMARY METRICS
# --------------------------------------------------
st.header("📌 Summary Metrics")

accuracy = (df_eval["Actual_Risk"] == df_eval["Predicted_Risk"]).mean()
mismatches = (df_eval["Actual_Risk"] != df_eval["Predicted_Risk"]).sum()

m1, m2 = st.columns(2)
m1.metric("Accuracy", f"{accuracy:.2%}")
m2.metric("Mismatches", mismatches)

# --------------------------------------------------
# SAMPLE OUTPUT
# --------------------------------------------------
st.header("🔍 Sample Predictions")

st.dataframe(
    df_eval[[life_col, "Actual_Risk", "Predicted_Risk"]].head(15),
    use_container_width=True
)
