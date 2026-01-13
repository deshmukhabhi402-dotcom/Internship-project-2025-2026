import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_backend():
    # Load data (same as notebook)
    df = pd.read_csv("UnifiedDataset.csv")

    # ⚠️ COPY THIS PART DIRECTLY FROM YOUR NOTEBOOK
    # (example shown — replace with YOUR exact logic)
    df["Health Risk Level"] = df["Health Risk Level"]

    X = df.drop(columns=["Health Risk Level"])
    y = df["Health Risk Level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "model": model
    }
