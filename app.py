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

# 1. UPLOAD RESULTS DATA
uploaded_file = st.file_uploader("📁 Upload your model results CSV", type=['csv'])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Show preview
        with st.expander("👀 Preview your data"):
            st.dataframe(df.head(), use_container_width=True)
        
        # 2. IDENTIFY COLUMNS
        st.header("🔍 Column Identification")
        
        # Find columns automatically
        actual_cols = []
        predicted_cols = []
        
        for col in df.columns:
            col_lower = str(col).lower()
            # Look for actual/test columns
            if any(word in col_lower for word in ['y_test', 'actual', 'true', 'test', 'real']):
                actual_cols.append(col)
            # Look for predicted columns
            elif any(word in col_lower for word in ['y_pred', 'predicted', 'pred', 'predict']):
                predicted_cols.append(col)
        
        if not actual_cols or not predicted_cols:
            st.error("❌ Could not find both actual and predicted columns automatically")
            st.write("Please ensure your CSV has columns like 'y_test' and 'y_pred'")
            st.stop()
        
        actual_col = actual_cols[0]
        predicted_col = predicted_cols[0]
        
        st.info(f"✅ Using: '{actual_col}' as Actual, '{predicted_col}' as Predicted")
        
        # 3. CREATE RISK LEVELS
        df['Actual_Risk'] = df[actual_col].apply(risk_level)
        df['Predicted_Risk'] = df[predicted_col].apply(risk_level)
        
        # Clean data
        df_clean = df.dropna(subset=['Actual_Risk', 'Predicted_Risk'])
        
        # Get counts
        actual_counts = df_clean['Actual_Risk'].value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0)
        predicted_counts = df_clean['Predicted_Risk'].value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0)
        
        # 4. REGRESSION METRICS
        st.header("📈 Regression Performance Metrics")
        
        # Calculate regression metrics
        y_true = df_clean[actual_col]
        y_pred = df_clean[predicted_col]
        
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
        
        # 5. CLASSIFICATION REPORT TABLES
        st.header("📋 Classification Reports")
        
        # Generate classification report
        report_dict = classification_report(df_clean['Actual_Risk'], df_clean['Predicted_Risk'], 
                                           labels=['Low', 'Medium', 'High'], 
                                           output_dict=True)
        
        # Display as plain text first
        report_text = classification_report(df_clean['Actual_Risk'], df_clean['Predicted_Risk'], 
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
        
        # Display accuracy metrics
        col_acc1, col_acc2, col_acc3, col_acc4 = st.columns(4)
        with col_acc1:
            st.metric("Accuracy", f"{accuracy:.2%}")
        with col_acc2:
            macro_precision = report_dict['macro avg']['precision']
            st.metric("Macro Avg Precision", f"{macro_precision:.3f}")
        with col_acc3:
            macro_recall = report_dict['macro avg']['recall']
            st.metric("Macro Avg Recall", f"{macro_recall:.3f}")
        with col_acc4:
            macro_f1 = report_dict['macro avg']['f1-score']
            st.metric("Macro Avg F1-Score", f"{macro_f1:.3f}")
        
        # 6. RISK LEVEL DISTRIBUTION BAR GRAPH
        st.header("📊 Risk Level Distribution")
        
        # Create figure with subplots
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        colors = {'High': '#FF6B6B', 'Medium': '#FFD93D', 'Low': '#6BCF7F'}
        
        # Left: Actual Risk Distribution
        bars1 = ax1.bar(actual_counts.index, actual_counts.values,
                       color=[colors[risk] for risk in actual_counts.index],
                       alpha=0.8, edgecolor='black')
        ax1.set_title('Actual Health Risk Level Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Risk Level', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add percentage labels
        total_actual = actual_counts.sum()
        for i, (risk, count) in enumerate(zip(actual_counts.index, actual_counts.values)):
            percentage = (count / total_actual * 100) if total_actual > 0 else 0
            ax1.text(i, count/2, f'{percentage:.1f}%', ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white')
        
        # Right: Predicted Risk Distribution
        bars2 = ax2.bar(predicted_counts.index, predicted_counts.values,
                       color=[colors[risk] for risk in predicted_counts.index],
                       alpha=0.8, edgecolor='black', hatch='//')
        ax2.set_title('Predicted Health Risk Level Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Risk Level', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add percentage labels
        total_predicted = predicted_counts.sum()
        for i, (risk, count) in enumerate(zip(predicted_counts.index, predicted_counts.values)):
            percentage = (count / total_predicted * 100) if total_predicted > 0 else 0
            ax2.text(i, count/2, f'{percentage:.1f}%', ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white')
        
        plt.tight_layout()
        st.pyplot(fig1)
        
        # 7. ACTUAL VS PREDICTED RISK LEVELS BAR GRAPHS
        st.header("📈 Actual vs Predicted Risk Levels Comparison")
        
        # Create comparison figure
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top: Side-by-side comparison
        x = np.arange(len(actual_counts))
        width = 0.35
        
        bars_actual = ax1.bar(x - width/2, actual_counts.values, width,
                             label='Actual', color=[colors[risk] for risk in actual_counts.index],
                             alpha=0.8, edgecolor='black')
        
        bars_pred = ax1.bar(x + width/2, predicted_counts.values, width,
                           label='Predicted', color=[colors[risk] for risk in predicted_counts.index],
                           alpha=0.6, edgecolor='black', hatch='//')
        
        ax1.set_xlabel('Risk Level', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Side-by-Side Comparison: Actual vs Predicted', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Low', 'Medium', 'High'])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars_actual, bars_pred]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                        f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        # Bottom: Grouped comparison with error bars
        risk_levels = ['Low', 'Medium', 'High']
        x_pos = np.arange(len(risk_levels))
        
        # Calculate accuracy for each risk level
        accuracies = []
        for risk in risk_levels:
            mask_actual = df_clean['Actual_Risk'] == risk
            mask_pred = df_clean['Predicted_Risk'] == risk
            correct = sum(mask_actual & mask_pred)
            total = sum(mask_actual)
            accuracy = correct / total if total > 0 else 0
            accuracies.append(accuracy)
        
        bars3 = ax2.bar(x_pos, accuracies, 
                       color=[colors[risk] for risk in risk_levels],
                       alpha=0.7, edgecolor='black')
        
        ax2.set_xlabel('Risk Level', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Classification Accuracy by Risk Level', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(risk_levels)
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add accuracy labels
        for i, (bar, acc) in enumerate(zip(bars3, accuracies)):
            ax2.text(bar.get_x() + bar.get_width()/2, acc + 0.02,
                    f'{acc:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig2)
        
        # 8. CONFUSION MATRIX
        st.header("🎯 Confusion Matrix")
        
        # Create confusion matrix
        cm = confusion_matrix(df_clean['Actual_Risk'], df_clean['Predicted_Risk'], 
                             labels=['Low', 'Medium', 'High'])
        
        fig3, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Set labels
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['Low', 'Medium', 'High'], 
               yticklabels=['Low', 'Medium', 'High'],
               title='Confusion Matrix',
               ylabel='Actual Risk',
               xlabel='Predicted Risk')
        
        # Rotate tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig3)
        
        # 9. SAMPLE RESULTS
        st.header("🔍 Sample Results")
        
        # Show sample comparisons
        sample_df = pd.DataFrame({
            'Actual_Value': df_clean[actual_col].head(10),
            'Predicted_Value': df_clean[predicted_col].head(10),
            'Actual_Risk': df_clean['Actual_Risk'].head(10),
            'Predicted_Risk': df_clean['Predicted_Risk'].head(10),
            'Match': df_clean['Actual_Risk'].head(10) == df_clean['Predicted_Risk'].head(10)
        })
        
        st.dataframe(sample_df.style.apply(
            lambda x: ['background-color: #C8E6C9' if v else 'background-color: #FFCDD2' 
                      for v in x] if x.name == 'Match' else [''] * len(x),
            axis=0
        ), use_container_width=True)
        
        # 10. SUMMARY STATISTICS
        st.header("📊 Summary Statistics")
        
        # Calculate summary statistics
        total_samples = len(df_clean)
        correct_predictions = sum(df_clean['Actual_Risk'] == df_clean['Predicted_Risk'])
        overall_accuracy = correct_predictions / total_samples
        
        summary_data = []
        for risk in ['Low', 'Medium', 'High']:
            actual_count = actual_counts.get(risk, 0)
            predicted_count = predicted_counts.get(risk, 0)
            correct = sum((df_clean['Actual_Risk'] == risk) & (df_clean['Predicted_Risk'] == risk))
            accuracy = correct / actual_count if actual_count > 0 else 0
            
            summary_data.append({
                'Risk Level': risk,
                'Actual Count': actual_count,
                'Predicted Count': predicted_count,
                'Correct': correct,
                'Accuracy': accuracy
            })
        
        # Add overall row
        summary_data.append({
            'Risk Level': 'OVERALL',
            'Actual Count': total_samples,
            'Predicted Count': total_samples,
            'Correct': correct_predictions,
            'Accuracy': overall_accuracy
        })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df.style.format({
            'Accuracy': '{:.2%}'
        }), use_container_width=True)
        
        # 11. INTERPRETATION
        st.header("💡 Interpretation")
        
        st.info("""
        **Model Performance Summary:**
        
        1. **Regression Performance**:
           - R² Score of {:.3f} indicates {} of the variance in Life Expectancy is explained by the model
           - RMSE of {:.2f} years shows the average prediction error
           - MAE of {:.2f} years indicates the average absolute error
        
        2. **Classification Performance**:
           - Overall accuracy of {:.1%} for risk level prediction
           - {} risk level has the highest accuracy at {:.1%}
           - {} risk level has the lowest accuracy at {:.1%}
        
        3. **Risk Distribution**:
           - Actual data has {:.1%} Low risk, {:.1%} Medium risk, {:.1%} High risk
           - Model predicts {:.1%} Low risk, {:.1%} Medium risk, {:.1%} High risk
        
        **Recommendations**:
        - {} risk predictions show strong performance
        - Consider improving predictions for {} risk category
        - Model is {} at distinguishing between different risk levels
        """.format(
            r2, "good" if r2 > 0.7 else "moderate" if r2 > 0.5 else "weak",
            rmse, mae,
            accuracy,
            max(zip(['Low', 'Medium', 'High'], accuracies), key=lambda x: x[1])[0],
            max(accuracies),
            min(zip(['Low', 'Medium', 'High'], accuracies), key=lambda x: x[1])[0],
            min(accuracies),
            actual_counts['Low']/total_actual*100, actual_counts['Medium']/total_actual*100, actual_counts['High']/total_actual*100,
            predicted_counts['Low']/total_predicted*100, predicted_counts['Medium']/total_predicted*100, predicted_counts['High']/total_predicted*100,
            "High" if accuracies[2] > 0.9 else "Medium" if accuracies[1] > 0.9 else "Low",
            "Medium" if accuracies[1] < 0.8 else "High" if accuracies[2] < 0.8 else "Low",
            "excellent" if accuracy > 0.9 else "good" if accuracy > 0.8 else "moderate" if accuracy > 0.7 else "needs improvement"
        ))
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.code(f"Error details: {e}", language='python')

else:
    # Show instructions
    st.info("👆 Upload your model results CSV file")
    
    st.write("""
    ### Expected Data Format:
    
    **Your CSV should contain actual and predicted Life Expectancy values**, for example:
    
    | y_test | y_pred |
    |--------|--------|
    | 58.3   | 56.8   |
    | 72.1   | 71.5   |
    | 81.5   | 80.9   |
    | 65.8   | 67.2   |
    | 79.2   | 78.5   |
    
    **From your notebook, save results like this:**
    ```python
    # After training your model
    results = pd.DataFrame({
        'y_test': y_test.values,
        'y_pred': y_pred
    })
    results.to_csv('model_results.csv', index=False)
    ```
    
    **The app will show:**
    1. Regression performance metrics (MSE, RMSE, MAE, R²)
    2. Classification report tables
    3. Risk level distribution bar graphs
    4. Actual vs Predicted risk level comparisons
    5. Confusion matrix
    6. Sample results
    7. Summary statistics
    8. Interpretation of results
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Life Expectancy Prediction Analysis Dashboard</strong></p>
    <p>Analyzing Random Forest regression results and health risk level classifications</p>
</div>
""", unsafe_allow_html=True)