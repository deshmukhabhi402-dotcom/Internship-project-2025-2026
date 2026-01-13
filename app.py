# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page setup
st.set_page_config(page_title="Risk Analysis", layout="wide")
st.title("📊 Life Expectancy Risk Analysis")

# 1. LOAD DATASET
st.header("1. Load Dataset")
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Show first 5 rows
    with st.expander("Preview data"):
        st.dataframe(df.head())
    
    # Show all column names
    st.write("**Available columns in your dataset:**")
    st.write(", ".join(df.columns.tolist()))
    
    # 2. SELECT LIFE EXPECTANCY COLUMN
    st.header("2. Select Columns")
    
    # Let user select which column contains life expectancy
    all_columns = df.columns.tolist()
    
    # Try to find life expectancy column automatically
    life_exp_columns = [col for col in all_columns if any(keyword in col.lower() 
                       for keyword in ['life', 'expectancy', 'lifexp', 'life_exp'])]
    
    if life_exp_columns:
        default_col = life_exp_columns[0]
    else:
        default_col = all_columns[0]
    
    life_exp_col = st.selectbox(
        "Select the Life Expectancy column:",
        all_columns,
        index=all_columns.index(default_col) if default_col in all_columns else 0
    )
    
    st.info(f"Using '{life_exp_col}' as life expectancy column")
    
    # 3. CREATE RISK CATEGORIES
    st.header("3. Risk Categories")
    
    # Check if column is numeric
    if pd.api.types.is_numeric_dtype(df[life_exp_col]):
        # Define risk levels based on selected column
        df['Actual_Risk'] = pd.cut(
            df[life_exp_col],
            bins=[0, 60, 75, 100],
            labels=['High Risk', 'Medium Risk', 'Low Risk']
        )
        
        st.success("✓ Risk categories created successfully!")
        
        # Show distribution
        risk_counts = df['Actual_Risk'].value_counts()
        st.write("**Initial Risk Distribution:**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("High Risk", risk_counts.get('High Risk', 0))
        with col2:
            st.metric("Medium Risk", risk_counts.get('Medium Risk', 0))
        with col3:
            st.metric("Low Risk", risk_counts.get('Low Risk', 0))
        
        # 4. GENERATE SAMPLE PREDICTIONS
        st.header("4. Generate Sample Predictions")
        
        if st.button("Generate Predictions", type="primary"):
            # Generate sample predictions with 85% accuracy
            np.random.seed(42)
            n = len(df)
            
            # Start with actual values
            predictions = df['Actual_Risk'].copy()
            
            # Make 15% wrong predictions
            wrong_count = int(0.15 * n)
            wrong_indices = np.random.choice(n, wrong_count, replace=False)
            
            for idx in wrong_indices:
                actual = df.loc[idx, 'Actual_Risk']
                # Predict a different risk level
                if actual == 'High Risk':
                    predictions[idx] = np.random.choice(['Medium Risk', 'Low Risk'])
                elif actual == 'Medium Risk':
                    predictions[idx] = np.random.choice(['High Risk', 'Low Risk'])
                else:
                    predictions[idx] = np.random.choice(['High Risk', 'Medium Risk'])
            
            df['Predicted_Risk'] = predictions
            
            # Calculate accuracy
            actual_accuracy = (df['Actual_Risk'] == df['Predicted_Risk']).mean()
            
            st.success("✓ Predictions generated!")
            
            # Show sample predictions
            with st.expander("View predictions (first 10 rows)"):
                display_cols = [life_exp_col, 'Actual_Risk', 'Predicted_Risk']
                if 'Country' in df.columns:
                    display_cols = ['Country'] + display_cols
                st.dataframe(df[display_cols].head(10))
            
            # 5. VISUALIZATIONS
            st.header("5. Visualizations")
            
            # Count values
            categories = ['High Risk', 'Medium Risk', 'Low Risk']
            actual_counts = df['Actual_Risk'].value_counts().reindex(categories, fill_value=0)
            predicted_counts = df['Predicted_Risk'].value_counts().reindex(categories, fill_value=0)
            
            # Create two columns layout
            col1_viz, col2_viz = st.columns(2)
            
            with col1_viz:
                # BAR GRAPH - Predicted vs Actual
                st.subheader("Predicted vs Actual (Bar Graph)")
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(categories))
                width = 0.35
                
                ax.bar(x - width/2, actual_counts.values, width, label='Actual', color='blue', alpha=0.7)
                ax.bar(x + width/2, predicted_counts.values, width, label='Predicted', color='orange', alpha=0.7)
                
                ax.set_xlabel('Risk Level')
                ax.set_ylabel('Count')
                ax.set_title('Predicted vs Actual Risk Levels')
                ax.set_xticks(x)
                ax.set_xticklabels(['High', 'Medium', 'Low'])
                ax.legend()
                
                # Add numbers on bars
                for i, (act, pred) in enumerate(zip(actual_counts.values, predicted_counts.values)):
                    ax.text(i - width/2, act + 0.5, str(int(act)), ha='center')
                    ax.text(i + width/2, pred + 0.5, str(int(pred)), ha='center')
                
                st.pyplot(fig)
            
            with col2_viz:
                # RISK DISTRIBUTION - Bar Graph
                st.subheader("Risk Distribution (Bar Graph)")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                colors = ['#FF6B6B', '#FFD93D', '#6BCF7F']  # Red, Yellow, Green
                
                # Plot actual distribution
                x = np.arange(len(categories))
                width = 0.35
                
                ax.bar(x - width/2, actual_counts.values, width, 
                       label='Actual', color=colors, alpha=0.8)
                ax.bar(x + width/2, predicted_counts.values, width, 
                       label='Predicted', color=colors, alpha=0.5)
                
                ax.set_xlabel('Risk Level')
                ax.set_ylabel('Count')
                ax.set_title('Risk Distribution: Actual vs Predicted')
                ax.set_xticks(x)
                ax.set_xticklabels(['High', 'Medium', 'Low'])
                ax.legend()
                
                # Add value labels
                for i, (act, pred) in enumerate(zip(actual_counts.values, predicted_counts.values)):
                    ax.text(i - width/2, act + 0.5, f'{int(act)}', ha='center', va='bottom')
                    ax.text(i + width/2, pred + 0.5, f'{int(pred)}', ha='center', va='bottom')
                
                st.pyplot(fig)
            
            # CLASSIFICATION REPORT
            st.subheader("Classification Report")
            
            # Calculate classification metrics
            from sklearn.metrics import classification_report, confusion_matrix
            
            # Create confusion matrix
            cm = confusion_matrix(df['Actual_Risk'], df['Predicted_Risk'], 
                                  labels=categories)
            
            # Create two columns for classification report
            col1_report, col2_report = st.columns(2)
            
            with col1_report:
                # Display confusion matrix
                st.write("**Confusion Matrix**")
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['High', 'Medium', 'Low'],
                           yticklabels=['High', 'Medium', 'Low'],
                           ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
            
            with col2_report:
                # Calculate classification report
                report_dict = classification_report(df['Actual_Risk'], df['Predicted_Risk'], 
                                                   output_dict=True, zero_division=0)
                
                # Display classification metrics table
                st.write("**Classification Metrics**")
                
                # Create a clean dataframe for display
                metrics_df = pd.DataFrame()
                
                for class_name in ['High Risk', 'Medium Risk', 'Low Risk', 'weighted avg', 'accuracy']:
                    if class_name in report_dict:
                        if class_name == 'accuracy':
                            # Handle accuracy separately
                            metrics_df = pd.concat([
                                metrics_df,
                                pd.DataFrame({
                                    'Class': ['Overall Accuracy'],
                                    'Precision': [report_dict[class_name]],
                                    'Recall': [report_dict[class_name]],
                                    'F1-Score': [report_dict[class_name]],
                                    'Support': [len(df)]
                                })
                            ])
                        else:
                            # Handle class metrics
                            metrics_df = pd.concat([
                                metrics_df,
                                pd.DataFrame({
                                    'Class': [class_name],
                                    'Precision': [report_dict[class_name].get('precision', 0)],
                                    'Recall': [report_dict[class_name].get('recall', 0)],
                                    'F1-Score': [report_dict[class_name].get('f1-score', 0)],
                                    'Support': [report_dict[class_name].get('support', 0)]
                                })
                            ])
                
                # Display the dataframe
                st.dataframe(
                    metrics_df.style.format({
                        'Precision': '{:.3f}',
                        'Recall': '{:.3f}',
                        'F1-Score': '{:.3f}',
                        'Support': '{:.0f}'
                    }).background_gradient(subset=['Precision', 'Recall', 'F1-Score'], 
                                          cmap='YlOrRd', low=0.3, high=0.7),
                    use_container_width=True
                )
            
            # Visualize classification metrics
            st.write("**Metrics Visualization**")
            
            # Create metrics visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            metrics_to_show = ['precision', 'recall', 'f1-score']
            metric_titles = ['Precision', 'Recall', 'F1-Score']
            metric_colors = ['#FF9800', '#2196F3', '#4CAF50']
            
            for i, (metric, title, color) in enumerate(zip(metrics_to_show, metric_titles, metric_colors)):
                ax = axes[i]
                values = []
                for class_name in ['High Risk', 'Medium Risk', 'Low Risk']:
                    if class_name in report_dict:
                        values.append(report_dict[class_name][metric])
                    else:
                        values.append(0)
                
                bars = ax.bar(['High', 'Medium', 'Low'], values, color=color, alpha=0.8)
                ax.set_ylim([0, 1])
                ax.set_title(title)
                ax.set_ylabel('Score')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                           f'{val:.3f}', ha='center', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
        
    else:
        st.error(f"❌ Column '{life_exp_col}' is not numeric. Please select a numeric column.")

else:
    st.info("👆 Please upload a CSV file to begin analysis")
    
    # Show example of what the data should look like
    st.write("**Your data should look something like this:**")
    example_data = pd.DataFrame({
        'Country': ['USA', 'UK', 'Japan', 'India', 'Brazil'],
        'Life_Expectancy': [78.5, 81.2, 84.6, 69.7, 75.9],
        'GDP_per_capita': [65000, 45000, 42000, 2100, 8900],
        'Healthcare_Spending': [11000, 4500, 4800, 75, 950]
    })
    st.dataframe(example_data)
    
    st.write("**Or with different column names:**")
    st.write("- 'LifeExpectancy', 'life_expectancy', 'LifeExp', 'longevity', etc.")
    st.write("Any numeric column can be used as life expectancy!")