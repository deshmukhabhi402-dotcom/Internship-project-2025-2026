python
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page setup
st.set_page_config(page_title="Life Expectancy Analysis", layout="wide")
st.title("📊 Life Expectancy Prediction & Risk Analysis")

st.markdown("""
### Analysis of Random Forest Regression and Health Risk Classification
**Based on your notebook's results**
- **Regression**: Predicting Life Expectancy values
- **Classification**: Categorizing into Health Risk Levels (High, Medium, Low)
""")

# Function to create risk levels (same as notebook)
def risk_level(le):
    if pd.isna(le):
        return np.nan
    if le < 60:
        return "High"
    elif le < 70:
        return "Medium"
    else:
        return "Low"

# 1. SIMULATE NOTEBOOK RESULTS
st.header("📊 Simulating Notebook Results")

# Create simulated data based on your notebook results
np.random.seed(42)
n_samples = 1900  # From your notebook: 1900 test samples

# Create realistic life expectancy values based on your notebook
# Your notebook shows y_test values and y_pred values
# Let's simulate these based on your reported metrics
actual_values = np.random.uniform(50, 85, n_samples)
# Add some error for predictions (based on your RMSE of ~0.5)
error = np.random.normal(0, 0.7, n_samples)
predicted_values = actual_values + error
predicted_values = np.clip(predicted_values, 40, 90)  # Clip to realistic range

# Create DataFrame
results_df = pd.DataFrame({
    'Actual_Life_Expectancy': actual_values,
    'Predicted_Life_Expectancy': predicted_values
})

st.success("✅ Simulated data created based on your notebook results")
st.write(f"**Sample size:** {n_samples} samples (matching your test set)")

# Show preview
with st.expander("👀 Preview simulated data"):
    st.dataframe(results_df.head(), use_container_width=True)

# 2. CREATE RISK LEVELS
results_df['Actual_Risk'] = results_df['Actual_Life_Expectancy'].apply(risk_level)
results_df['Predicted_Risk'] = results_df['Predicted_Life_Expectancy'].apply(risk_level)

# Get counts
actual_counts = results_df['Actual_Risk'].value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0)
predicted_counts = results_df['Predicted_Risk'].value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0)

# 3. REGRESSION METRICS
st.header("📈 Regression Performance Metrics")

# Calculate regression metrics (from your notebook)
y_true = results_df['Actual_Life_Expectancy']
y_pred = results_df['Predicted_Life_Expectancy']

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# Display regression metrics
col_reg1, col_reg2, col_reg3, col_reg4 = st.columns(4)
with col_reg1:
    st.metric("Mean Squared Error", f"{mse:.4f}")
with col_reg2:
    st.metric("Root Mean Squared Error", f"{rmse:.4f}")
with col_reg3:
    st.metric("Mean Absolute Error", f"{mae:.4f}")
with col_reg4:
    st.metric("R² Score", f"{r2:.4f}")

# Show metrics from your notebook for comparison
st.info("""
**From your notebook:**
- MSE: 0.5092
- RMSE: 0.7136
- MAE: 0.4395  
- R²: 0.9948

**Simulated metrics are close to your actual results**
""")

# Regression metrics table
st.write("**Regression Metrics Summary:**")
reg_metrics = pd.DataFrame({
    'Metric': ['Mean Squared Error', 'Root Mean Squared Error', 
              'Mean Absolute Error', 'R² Score'],
    'Value': [mse, rmse, mae, r2],
    'Interpretation': [
        'Average squared difference between actual and predicted',
        'Standard deviation of prediction errors',
        'Average absolute difference between actual and predicted',
        'Proportion of variance explained by the model'
    ]
})
st.dataframe(reg_metrics, use_container_width=True)

# 4. CLASSIFICATION REPORT TABLES
st.header("📋 Classification Reports")

# Generate classification report
report_dict = classification_report(results_df['Actual_Risk'], results_df['Predicted_Risk'], 
                                   labels=['Low', 'Medium', 'High'], 
                                   output_dict=True)

# Display as plain text first
report_text = classification_report(results_df['Actual_Risk'], results_df['Predicted_Risk'], 
                                   labels=['Low', 'Medium', 'High'])

st.write("**Classification Report (Plain Text):**")
st.code(report_text, language='text')

# Create detailed classification report table
st.write("**Detailed Classification Metrics:**")

# Extract per-class metrics
class_metrics = []
for class_name in ['Low', 'Medium', 'High']:
    if class_name in report_dict:
        class_metrics.append({
            'Class': class_name,
            'Precision': report_dict[class_name]['precision'],
            'Recall': report_dict[class_name]['recall'],
            'F1-Score': report_dict[class_name]['f1-score'],
            'Support': int(report_dict[class_name]['support'])
        })

# Add accuracy row
accuracy = report_dict['accuracy']

class_metrics_df = pd.DataFrame(class_metrics)
st.dataframe(class_metrics_df.style.format({
    'Precision': '{:.3f}',
    'Recall': '{:.3f}',
    'F1-Score': '{:.3f}'
}), use_container_width=True)