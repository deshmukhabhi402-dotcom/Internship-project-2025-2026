# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                            r2_score, classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Health Risk Prediction",
    layout="wide"
)

st.title("🏥 Health Risk Prediction from Notebook")
st.write("Loading data and displaying output from Intern_project.ipynb")

# Load and process data
@st.cache_data
def load_and_process_data():
    try:
        # Load data
        df = pd.read_csv("UnifiedDataset.csv")
        
        # Handle missing values
        num_cols = df.select_dtypes(include=np.number).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        
        cat_cols = df.select_dtypes(exclude=np.number).columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        # Encode categorical variables
        label_encoder = LabelEncoder()
        for col in cat_cols:
            df[col] = label_encoder.fit_transform(df[col])
        
        # Create Health Risk Level
        def risk_level(le):
            if le < 60:
                return "High"
            elif le < 70:
                return "Medium"
            else:
                return "Low"
        
        df["Health_Risk_Level"] = df["Life Expectancy"].apply(risk_level)
        
        # Prepare features and target
        X = df.drop(["Life Expectancy", "Health_Risk_Level"], axis=1)
        y = df["Life Expectancy"]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train model
        rf_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Convert to risk levels
        y_test_risk = y_test.apply(risk_level)
        y_pred_risk = pd.Series(y_pred).apply(risk_level)
        
        # Classification report
        class_report = classification_report(y_test_risk, y_pred_risk, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test_risk, y_pred_risk)
        
        return {
            'df': df,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_test_risk': y_test_risk,
            'y_pred_risk': y_pred_risk,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'class_report': class_report,
            'conf_matrix': conf_matrix,
            'model': rf_model,
            'feature_names': X.columns
        }
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Main app
if st.button("Load Data and Run Analysis", type="primary"):
    with st.spinner("Loading data and running analysis..."):
        results = load_and_process_data()
        
        if results is not None:
            st.success("✅ Data loaded and analysis completed!")
            
            # Show dataset info
            st.header("📊 Dataset Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", results['df'].shape[0])
            with col2:
                st.metric("Total Columns", results['df'].shape[1])
            with col3:
                st.metric("Original Columns", 150)
            
            # Show first 5 rows
            with st.expander("View First 5 Rows of Data"):
                st.dataframe(results['df'].head())
            
            # Show regression metrics
            st.header("📈 Regression Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Squared Error", f"{results['mse']:.4f}")
            with col2:
                st.metric("Mean Absolute Error", f"{results['mae']:.4f}")
            with col3:
                st.metric("R² Score", f"{results['r2']:.4f}")
            
            # Show classification report
            st.header("🔍 Classification Report (Risk Levels)")
            class_report_df = pd.DataFrame(results['class_report']).transpose()
            st.dataframe(class_report_df)
            
            # Show confusion matrix
            st.header("🤖 Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            classes = ['High', 'Low', 'Medium']
            sns.heatmap(results['conf_matrix'], annot=True, fmt='d', cmap='Blues',
                       xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            
            # Show actual vs predicted
            st.header("📊 Actual vs Predicted Life Expectancy")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(results['y_test'], results['y_pred'], alpha=0.6, color='blue')
            min_val = min(results['y_test'].min(), results['y_pred'].min())
            max_val = max(results['y_test'].max(), results['y_pred'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            ax.set_xlabel('Actual Life Expectancy')
            ax.set_ylabel('Predicted Life Expectancy')
            ax.set_title('Actual vs Predicted Values')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Show feature importance
            st.header("🎯 Top 15 Feature Importances")
            feature_importance = pd.DataFrame({
                'feature': results['feature_names'],
                'importance': results['model'].feature_importances_
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False).head(15)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            y_pos = np.arange(len(feature_importance))
            bars = ax.barh(y_pos, feature_importance['importance'].values)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_importance['feature'].values)
            ax.invert_yaxis()
            ax.set_xlabel('Importance')
            ax.set_title('Top 15 Feature Importances')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show risk distribution
            st.header("📋 Health Risk Level Distribution")
            risk_counts = results['df']['Health_Risk_Level'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                # Pie chart
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = plt.cm.Set3(np.linspace(0, 1, len(risk_counts)))
                ax.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                      colors=colors, startangle=90)
                ax.set_title('Risk Level Distribution')
                ax.axis('equal')
                st.pyplot(fig)
            
            with col2:
                # Bar chart
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = plt.cm.Set3(np.linspace(0, 1, len(risk_counts)))
                bars = ax.bar(risk_counts.index, risk_counts.values, color=colors)
                ax.set_xlabel('Risk Level')
                ax.set_ylabel('Count')
                ax.set_title('Risk Level Counts')
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
                st.pyplot(fig)
            
            st.success("🎉 All analysis completed! All graphs and classification report are displayed above.")
            
else:
    st.info("👆 Click the button above to load the data and run the analysis")
    st.write("This will:")
    st.write("1. Load UnifiedDataset.csv")
    st.write("2. Preprocess the data (handle missing values, encode categorical variables)")
    st.write("3. Train a Random Forest model")
    st.write("4. Display all metrics and visualizations")