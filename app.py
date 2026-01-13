# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page setup
st.set_page_config(page_title="Life Expectancy Analysis", layout="wide")
st.title("📊 Life Expectancy Prediction Analysis")

st.markdown("""
### This app visualizes your notebook's regression results
**Your notebook shows:** Actual vs Predicted Life Expectancy values (not just risk categories)
""")

# 1. UPLOAD YOUR REGRESSION RESULTS
st.header("📁 Upload Your Regression Results CSV")
uploaded_file = st.file_uploader("Upload CSV with actual and predicted life expectancy", type=['csv'])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Show columns
        st.write("**Columns in your file:**")
        st.write(", ".join(df.columns.tolist()))
        
        # Show preview
        with st.expander("👀 Preview your data"):
            st.dataframe(df.head(), use_container_width=True)
        
        # 2. SELECT COLUMNS
        st.header("🔍 Select Your Columns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Find actual life expectancy column
            actual_cols = [col for col in df.columns if any(word in col.lower() 
                          for word in ['actual', 'true', 'test', 'y_test', 'life'])]
            if actual_cols:
                actual_col = st.selectbox("Actual Life Expectancy:", 
                                         df.columns, 
                                         index=df.columns.get_loc(actual_cols[0]))
            else:
                actual_col = st.selectbox("Actual Life Expectancy:", df.columns)
        
        with col2:
            # Find predicted life expectancy column
            pred_cols = [col for col in df.columns if any(word in col.lower() 
                         for word in ['predicted', 'pred', 'y_pred', 'predict'])]
            if pred_cols:
                predicted_col = st.selectbox("Predicted Life Expectancy:", 
                                            df.columns, 
                                            index=df.columns.get_loc(pred_cols[0]))
            else:
                predicted_col = st.selectbox("Predicted Life Expectancy:", df.columns)
        
        # Check if columns are numeric
        actual_is_numeric = pd.api.types.is_numeric_dtype(df[actual_col])
        predicted_is_numeric = pd.api.types.is_numeric_dtype(df[predicted_col])
        
        if not (actual_is_numeric and predicted_is_numeric):
            st.error("❌ Please select numeric columns for life expectancy values")
        else:
            # 3. PROCESS DATA
            if st.button("📊 Generate Visualizations", type="primary"):
                # Create results dataframe
                results_df = pd.DataFrame({
                    'Actual': df[actual_col],
                    'Predicted': df[predicted_col]
                })
                
                # Add Risk Categories based on actual life expectancy
                results_df['Actual_Risk'] = pd.cut(
                    results_df['Actual'],
                    bins=[0, 60, 75, 100],
                    labels=['High Risk', 'Medium Risk', 'Low Risk']
                )
                
                # Add Risk Categories based on predicted life expectancy
                results_df['Predicted_Risk'] = pd.cut(
                    results_df['Predicted'],
                    bins=[0, 60, 75, 100],
                    labels=['High Risk', 'Medium Risk', 'Low Risk']
                )
                
                # Calculate regression metrics
                mse = mean_squared_error(results_df['Actual'], results_df['Predicted'])
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(results_df['Actual'], results_df['Predicted'])
                r2 = r2_score(results_df['Actual'], results_df['Predicted'])
                
                # Calculate classification accuracy for risk categories
                risk_accuracy = (results_df['Actual_Risk'] == results_df['Predicted_Risk']).mean()
                
                # Display metrics
                st.success("✅ Analysis complete!")
                
                col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                with col_met1:
                    st.metric("R² Score", f"{r2:.3f}")
                with col_met2:
                    st.metric("RMSE", f"{rmse:.2f}")
                with col_met3:
                    st.metric("MAE", f"{mae:.2f}")
                with col_met4:
                    st.metric("Risk Accuracy", f"{risk_accuracy:.1%}")
                
                # 4. VISUALIZATIONS
                st.header("📈 Visualizations")
                
                # Create tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Actual vs Predicted", 
                    "Risk Distribution", 
                    "Regression Plot",
                    "Error Analysis"
                ])
                
                with tab1:
                    # BAR GRAPH - Actual vs Predicted Life Expectancy (Binned)
                    st.subheader("Actual vs Predicted Life Expectancy Distribution")
                    
                    # Bin the life expectancy values for visualization
                    bins = np.arange(40, 91, 5)  # 40 to 90 in steps of 5
                    actual_binned = pd.cut(results_df['Actual'], bins=bins, include_lowest=True)
                    predicted_binned = pd.cut(results_df['Predicted'], bins=bins, include_lowest=True)
                    
                    # Get counts
                    actual_counts = actual_binned.value_counts().sort_index()
                    predicted_counts = predicted_binned.value_counts().sort_index()
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(12, 6))
                    x = np.arange(len(actual_counts))
                    width = 0.35
                    
                    ax.bar(x - width/2, actual_counts.values, width, 
                          label='Actual', color='blue', alpha=0.7)
                    ax.bar(x + width/2, predicted_counts.values, width, 
                          label='Predicted', color='orange', alpha=0.7)
                    
                    ax.set_xlabel('Life Expectancy Range (years)', fontsize=12)
                    ax.set_ylabel('Number of Samples', fontsize=12)
                    ax.set_title('Distribution of Actual vs Predicted Life Expectancy', fontsize=14, fontweight='bold')
                    ax.set_xticks(x)
                    
                    # Format x-tick labels
                    labels = [f"{int(interval.left)}-{int(interval.right)}" for interval in actual_counts.index]
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                    
                    ax.legend()
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Add value labels on top bars
                    for i, (act, pred) in enumerate(zip(actual_counts.values, predicted_counts.values)):
                        if act > 0:
                            ax.text(i - width/2, act + 0.5, str(int(act)), ha='center', fontsize=8)
                        if pred > 0:
                            ax.text(i + width/2, pred + 0.5, str(int(pred)), ha='center', fontsize=8)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.info("**Interpretation:** Shows how actual and predicted life expectancy values are distributed across different ranges.")
                
                with tab2:
                    # BAR GRAPH - Risk Level Distribution
                    st.subheader("Health Risk Level Distribution")
                    
                    # Get counts for risk categories
                    actual_risk_counts = results_df['Actual_Risk'].value_counts()
                    predicted_risk_counts = results_df['Predicted_Risk'].value_counts()
                    
                    # Ensure all categories are present
                    categories = ['High Risk', 'Medium Risk', 'Low Risk']
                    actual_risk_counts = actual_risk_counts.reindex(categories, fill_value=0)
                    predicted_risk_counts = predicted_risk_counts.reindex(categories, fill_value=0)
                    
                    # Create two visualizations
                    col_risk1, col_risk2 = st.columns(2)
                    
                    with col_risk1:
                        # Bar chart comparing actual vs predicted risk
                        fig, ax = plt.subplots(figsize=(8, 6))
                        x = np.arange(len(categories))
                        width = 0.35
                        
                        ax.bar(x - width/2, actual_risk_counts.values, width, 
                              label='Actual Risk', color='blue', alpha=0.7)
                        ax.bar(x + width/2, predicted_risk_counts.values, width, 
                              label='Predicted Risk', color='orange', alpha=0.7)
                        
                        ax.set_xlabel('Health Risk Level', fontsize=12)
                        ax.set_ylabel('Number of Samples', fontsize=12)
                        ax.set_title('Actual vs Predicted Health Risk Levels', fontsize=14, fontweight='bold')
                        ax.set_xticks(x)
                        ax.set_xticklabels(['Low', 'Medium', 'High'])
                        ax.legend()
                        ax.grid(True, alpha=0.3, axis='y')
                        
                        # Add value labels
                        for i, (act, pred) in enumerate(zip(actual_risk_counts.values, predicted_risk_counts.values)):
                            ax.text(i - width/2, act + 0.5, str(int(act)), ha='center', fontsize=10)
                            ax.text(i + width/2, pred + 0.5, str(int(pred)), ha='center', fontsize=10)
                        
                        st.pyplot(fig)
                    
                    with col_risk2:
                        # Actual risk distribution only
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        colors = ['#FF6B6B', '#FFD93D', '#6BCF7F']  # Red, Yellow, Green
                        bars = ax.bar(categories, actual_risk_counts.values, 
                                     color=colors, alpha=0.8, edgecolor='black')
                        
                        ax.set_xlabel('Health Risk Level', fontsize=12)
                        ax.set_ylabel('Number of Samples', fontsize=12)
                        ax.set_title('Distribution of Health Risk Levels', fontsize=14, fontweight='bold')
                        ax.set_xticklabels(['Low', 'Medium', 'High'])
                        ax.grid(True, alpha=0.3, axis='y')
                        
                        # Add value labels
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                                   f'{int(height)}', ha='center', fontweight='bold')
                        
                        st.pyplot(fig)
                    
                    # Show risk classification summary
                    st.write("**Risk Classification Summary:**")
                    risk_summary = pd.DataFrame({
                        'Risk Level': categories,
                        'Actual Count': actual_risk_counts.values,
                        'Predicted Count': predicted_risk_counts.values,
                        'Correct Predictions': [(results_df[(results_df['Actual_Risk'] == cat) & 
                                                          (results_df['Predicted_Risk'] == cat)]).shape[0] 
                                              for cat in categories]
                    })
                    
                    risk_summary['Accuracy'] = risk_summary['Correct Predictions'] / risk_summary['Actual Count']
                    risk_summary['Accuracy'] = risk_summary['Accuracy'].fillna(0)
                    
                    st.dataframe(risk_summary.style.format({
                        'Accuracy': '{:.1%}'
                    }), use_container_width=True)
                
                with tab3:
                    # SCATTER PLOT - Actual vs Predicted
                    st.subheader("Regression: Actual vs Predicted Life Expectancy")
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Create scatter plot
                    scatter = ax.scatter(results_df['Actual'], results_df['Predicted'], 
                                        alpha=0.6, s=50, c='steelblue', edgecolors='white', linewidth=0.5)
                    
                    # Add perfect prediction line
                    min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
                    max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
                    ax.plot([min_val, max_val], [min_val, max_val], 
                           'r--', label='Perfect Prediction', linewidth=2)
                    
                    # Add regression line
                    z = np.polyfit(results_df['Actual'], results_df['Predicted'], 1)
                    p = np.poly1d(z)
                    ax.plot(results_df['Actual'], p(results_df['Actual']), 
                           'g-', label=f'Regression Line (R²={r2:.3f})', alpha=0.8)
                    
                    ax.set_xlabel('Actual Life Expectancy (years)', fontsize=12)
                    ax.set_ylabel('Predicted Life Expectancy (years)', fontsize=12)
                    ax.set_title('Actual vs Predicted Life Expectancy', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Set equal aspect ratio
                    ax.set_aspect('equal', adjustable='box')
                    
                    # Add text with metrics
                    textstr = f'R² = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}'
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                           verticalalignment='top', bbox=props)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Residual plot
                    st.write("**Residual Plot:**")
                    results_df['Residuals'] = results_df['Predicted'] - results_df['Actual']
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Residuals vs Actual
                    ax1.scatter(results_df['Actual'], results_df['Residuals'], alpha=0.6, s=40)
                    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                    ax1.set_xlabel('Actual Life Expectancy')
                    ax1.set_ylabel('Residuals (Predicted - Actual)')
                    ax1.set_title('Residuals vs Actual')
                    ax1.grid(True, alpha=0.3)
                    
                    # Residuals histogram
                    ax2.hist(results_df['Residuals'], bins=30, edgecolor='black', alpha=0.7)
                    ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5)
                    ax2.set_xlabel('Residuals')
                    ax2.set_ylabel('Frequency')
                    ax2.set_title('Distribution of Residuals')
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with tab4:
                    # ERROR ANALYSIS
                    st.subheader("Error Analysis")
                    
                    # Calculate absolute errors
                    results_df['Absolute_Error'] = np.abs(results_df['Predicted'] - results_df['Actual'])
                    
                    # Create error analysis visualizations
                    col_err1, col_err2 = st.columns(2)
                    
                    with col_err1:
                        # Error by actual life expectancy
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        # Bin actual values and calculate mean error per bin
                        results_df['Actual_Bin'] = pd.cut(results_df['Actual'], bins=10)
                        error_by_bin = results_df.groupby('Actual_Bin')['Absolute_Error'].mean()
                        
                        # Plot
                        x_pos = np.arange(len(error_by_bin))
                        bars = ax.bar(x_pos, error_by_bin.values, alpha=0.7)
                        
                        ax.set_xlabel('Actual Life Expectancy Range', fontsize=11)
                        ax.set_ylabel('Mean Absolute Error', fontsize=11)
                        ax.set_title('Prediction Error by Life Expectancy Range', fontsize=13)
                        ax.set_xticks(x_pos)
                        
                        # Format x-tick labels
                        labels = []
                        for interval in error_by_bin.index:
                            left = int(interval.left)
                            right = int(interval.right)
                            labels.append(f"{left}-{right}")
                        ax.set_xticklabels(labels, rotation=45, ha='right')
                        
                        # Add value labels
                        for bar, val in zip(bars, error_by_bin.values):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                                   f'{height:.1f}', ha='center', fontsize=9)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with col_err2:
                        # Error distribution
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        # Plot error distribution
                        ax.hist(results_df['Absolute_Error'], bins=30, 
                               edgecolor='black', alpha=0.7, color='steelblue')
                        
                        # Add mean line
                        mean_error = results_df['Absolute_Error'].mean()
                        ax.axvline(x=mean_error, color='red', linestyle='--', 
                                  label=f'Mean Error: {mean_error:.2f}')
                        
                        ax.set_xlabel('Absolute Prediction Error (years)', fontsize=11)
                        ax.set_ylabel('Frequency', fontsize=11)
                        ax.set_title('Distribution of Prediction Errors', fontsize=13)
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Summary statistics
                    st.write("**Error Statistics:**")
                    error_stats = pd.DataFrame({
                        'Statistic': ['Mean Absolute Error', 'Root Mean Square Error', 
                                     'Max Absolute Error', 'Min Absolute Error',
                                     'Error Std Dev', 'Median Absolute Error'],
                        'Value': [
                            results_df['Absolute_Error'].mean(),
                            rmse,
                            results_df['Absolute_Error'].max(),
                            results_df['Absolute_Error'].min(),
                            results_df['Absolute_Error'].std(),
                            results_df['Absolute_Error'].median()
                        ]
                    })
                    
                    error_stats['Value'] = error_stats['Value'].round(3)
                    st.dataframe(error_stats, use_container_width=True)
                    
                    # Show samples with largest errors
                    st.write("**Samples with Largest Prediction Errors:**")
                    largest_errors = results_df.nlargest(10, 'Absolute_Error')[['Actual', 'Predicted', 'Absolute_Error']]
                    largest_errors = largest_errors.round(2)
                    st.dataframe(largest_errors, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Make sure you've selected numeric columns for life expectancy values.")

else:
    # Show instructions
    st.info("👆 Upload your regression results CSV file")
    
    st.write("""
    ### How to prepare your data:
    
    **In your notebook, save your regression results:**
    ```python
    # After training your regression model
    results = pd.DataFrame({
        'Actual_Life_Expectancy': y_test,      # Actual values from test set
        'Predicted_Life_Expectancy': y_pred    # Model predictions
    })
    
    # Save to CSV
    results.to_csv('regression_results.csv', index=False)
    ```
    
    **Expected format:**
    - Two numeric columns: Actual and Predicted life expectancy values
    - Example:
    """)
    
    example = pd.DataFrame({
        'Actual_Life_Expectancy': [58.3, 72.1, 81.5, 65.8, 79.2],
        'Predicted_Life_Expectancy': [56.8, 71.5, 80.9, 67.2, 78.5]
    })
    st.dataframe(example, use_container_width=True)
    
    st.write("""
    **What this app shows:**
    1. **Distribution plots** of actual vs predicted values
    2. **Risk level analysis** based on life expectancy thresholds
    3. **Regression plots** with performance metrics
    4. **Error analysis** to understand prediction accuracy
    """)