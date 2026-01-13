import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="ML Intern Project",
    layout="wide"
)

st.title("📊 Machine Learning Project Dashboard")

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("UnifiedDataset.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.write("Shape:", df.shape)

# ----------------------------
# Target selection
# ----------------------------
st.subheader("Target Variable Selection")

target_col = st.selectbox(
    "Select the target column",
    options=df.columns
)

X = df.drop(columns=[target_col])
y = df[target_col]

# ----------------------------
# Preprocessing
# ----------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ----------------------------
# Model
# ----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# ----------------------------
# Train Model
# ----------------------------
if st.button("🚀 Train Model"):
    with st.spinner("Training model..."):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

    st.success("Model trained successfully!")

    st.metric("Accuracy", f"{acc:.4f}")

    # ----------------------------
    # Classification Report
    # ----------------------------
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # ----------------------------
    # Confusion Matrix
    # ----------------------------
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
