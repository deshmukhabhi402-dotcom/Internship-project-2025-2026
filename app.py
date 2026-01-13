import streamlit as st
import pandas as pd

st.set_page_config(page_title="Life Expectancy Risk", layout="wide")
st.title("Life Expectancy Risk Dashboard")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("UnifiedDataset.csv")

df = load_data()

# -----------------------------
# Find Life Expectancy Column (NO KeyError possible)
# -----------------------------
life_col = None
for col in df.columns:
    col_clean = col.lower().replace(" ", "")
    if "life" in col_clean and "expect" in col_clean:
        life_col = col
        break

if life_col is None:
    st.error("Life expectancy column not found in dataset.")
    st.write("Available columns:", list(df.columns))
    st.stop()

st.success(f"Detected target column → {repr(life_col)}")

# -----------------------------
# Data Processing (Frontend-visible only)
# -----------------------------
processed_df = df.dropna(subset=[life_col]).copy()

RISK_THRESHOLD = 65
processed_df["Risk Level"] = processed_df[life_col].apply(
    lambda x: "High Risk" if x < RISK_THRESHOLD else "Low Risk"
)

# -----------------------------
# Sidebar Filter
# -----------------------------
st.sidebar.header("Filters")
risk_filter = st.sidebar.multiselect(
    "Risk Level",
    options=processed_df["Risk Level"].unique(),
    default=processed_df["Risk Level"].unique()
)

filtered_df = processed_df[processed_df["Risk Level"].isin(risk_filter)]

# -----------------------------
# Metrics
# -----------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Total Records", len(filtered_df))
c2.metric("High Risk", (filtered_df["Risk Level"] == "High Risk").sum())
c3.metric("Low Risk", (filtered_df["Risk Level"] == "Low Risk").sum())

# -----------------------------
# Display Processed Output
# -----------------------------
st.subheader("Processed Dataset (Frontend Output)")
st.dataframe(filtered_df, use_container_width=True, height=450)

# -----------------------------
# Explanation
# -----------------------------
with st.expander("Risk Logic"):
    st.markdown(
        f"""
        **Target:** {life_col}  
        **High Risk:** Life expectancy < {RISK_THRESHOLD}  
        **Low Risk:** Life expectancy ≥ {RISK_THRESHOLD}  
        """
    )
