# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page setup
st.set_page_config(page_title="Health Risk Level Analysis", layout="wide")
st.title("📊 Health Risk Level Analysis")

st.markdown("""
### Comparing Actual vs Predicted Health Risk Levels
**Risk categories based on Life Expectancy:**
- **High Risk**: Life Expectancy < 60 years
- **Medium Risk**: Life Expectancy 60-70 years  
- **Low Risk**: Life Expectancy > 70 years
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

# 1. UPLOAD YOUR DATA
st.header("📁 Upload Your Results")
uploaded_file = st.file_uploader("Upload CSV with Actual and Predicted Life Expectancy", type=['csv'])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Show preview
        with st.expander("👀 Preview your data"):
            st.dataframe(df.head(), use_container_width=True)
        
        # 2. FIND COLUMNS
        st.header("🔍 Identify Your Columns")
        
        # Try to auto-detect columns
        actual_cols = []
        predicted_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(word in col_lower for word in ['actual', 'true', 'y_test', 'real', 'test']):
                actual_cols.append(col)
            elif any(word in col_lower for word in ['predicted', 'pred', 'y_pred', 'predict', 'estimate']):
                predicted_cols.append(col)
            elif 'life' in col_lower and 'expectancy' in col_lower:
                if not actual_cols:  # Use first life expectancy as actual
                    actual_cols.append(col)
        
        # If we couldn't auto-detect, ask user
        if not actual_cols or not predicted_cols:
            st.warning("Could not auto-detect all required columns. Please select manually:")
            
            col1, col2 = st.columns(2)
            with col1:
                actual_col = st.selectbox("Select Actual Life Expectancy:", df.columns)
            with col2:
                predicted_col = st.selectbox("Select Predicted Life Expectancy:", df.columns)
        else:
            actual_col = actual_cols[0]
            predicted_col = predicted_cols[0]
            st.info(f"✅ Auto-detected: '{actual_col}' as Actual, '{predicted_col}' as Predicted")
        
        # Check if columns are numeric
        if not pd.api.types.is_numeric_dtype(df[actual_col]):
            st.error(f"❌ Column '{actual_col}' must contain numeric values")
            st.stop()
        
        if not pd.api.types.is_numeric_dtype(df[predicted_col]):
            st.error(f"❌ Column '{predicted_col}' must contain numeric values")
            st.stop()
        
        # Create risk levels
        df['Actual_Risk'] = df[actual_col].apply(risk_level)
        df['Predicted_Risk'] = df[predicted_col].apply(risk_level)
        
        # Remove any rows with NaN
        df_clean = df.dropna(subset=['Actual_Risk', 'Predicted_Risk'])
        
        # Get counts
        actual_counts = df_clean['Actual_Risk'].value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0)
        predicted_counts = df_clean['Predicted_Risk'].value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0)
        
        # 3. DISPLAY COUNTS
        st.header("📊 Risk Level Counts")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Low Risk")
            st.metric("Actual", actual_counts.get('Low', 0))
            st.metric("Predicted", predicted_counts.get('Low', 0))
        
        with col2:
            st.subheader("Medium Risk")
            st.metric("Actual", actual_counts.get('Medium', 0))
            st.metric("Predicted", predicted_counts.get('Medium', 0))
        
        with col3:
            st.subheader("High Risk")
            st.metric("Actual", actual_counts.get('High', 0))
            st.metric("Predicted", predicted_counts.get('High', 0))
        
        # 4. VISUALIZATIONS - ACTUAL VS PREDICTED
        st.header("📈 Actual vs Predicted Health Risk Levels")
        
        # Create two columns for visualizations
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            # FIGURE 1: Side-by-side comparison
            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            
            # Colors for risk levels
            colors = {'High': '#FF6B6B', 'Medium': '#FFD93D', 'Low': '#6BCF7F'}
            
            # Top: Actual distribution
            bars1 = ax1.bar(actual_counts.index, actual_counts.values,
                           color=[colors[risk] for risk in actual_counts.index],
                           alpha=0.8, edgecolor='black')
            ax1.set_title('Actual Risk Level Distribution', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Count', fontsize=12)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                        f'{int(height)}', ha='center', fontsize=11, fontweight='bold')
            
            # Bottom: Predicted distribution
            bars2 = ax2.bar(predicted_counts.index, predicted_counts.values,
                           color=[colors[risk] for risk in predicted_counts.index],
                           alpha=0.8, edgecolor='black', hatch='//')
            ax2.set_title('Predicted Risk Level Distribution', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Count', fontsize=12)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                        f'{int(height)}', ha='center', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col_viz2:
            # FIGURE 2: Paired comparison
            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            
            # Top: Side-by-side bars
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
            ax1.set_title('Actual vs Predicted Comparison', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(['Low', 'Medium', 'High'])
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bars in [bars_actual, bars_pred]:
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                            f'{int(height)}', ha='center', fontsize=10)
            
            # Bottom: Confusion Matrix
            risk_levels = ['Low', 'Medium', 'High']
            cm = confusion_matrix(df_clean['Actual_Risk'], df_clean['Predicted_Risk'], 
                                 labels=risk_levels, normalize='true')
            
            im = ax2.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
            ax2.figure.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            
            # Set labels
            ax2.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=risk_levels, 
                   yticklabels=risk_levels,
                   title='Confusion Matrix (Normalized)',
                   ylabel='Actual Risk',
                   xlabel='Predicted Risk')
            
            # Rotate tick labels
            plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax2.text(j, i, format(cm[i, j], '.2f'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black",
                            fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig2)
        
        # 5. CLASSIFICATION REPORT
        st.header("📋 Classification Performance")
        
        # Generate classification report
        report = classification_report(df_clean['Actual_Risk'], df_clean['Predicted_Risk'], 
                                      labels=risk_levels, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Display metrics
        st.write("**Performance Metrics:**")
        
        # Create metrics display
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            accuracy = report_df.loc['accuracy', 'f1-score']
            st.metric("Accuracy", f"{accuracy:.2%}")
        
        with metrics_col2:
            avg_precision = report_df.loc['macro avg', 'precision']
            st.metric("Avg Precision", f"{avg_precision:.2f}")
        
        with metrics_col3:
            avg_recall = report_df.loc['macro avg', 'recall']
            st.metric("Avg Recall", f"{avg_recall:.2f}")
        
        with metrics_col4:
            avg_f1 = report_df.loc['macro avg', 'f1-score']
            st.metric("Avg F1-Score", f"{avg_f1:.2f}")
        
        # Display detailed report
        st.write("**Detailed Classification Report:**")
        st.dataframe(report_df.style.format({
            'precision': '{:.3f}',
            'recall': '{:.3f}',
            'f1-score': '{:.3f}',
            'support': '{:.0f}'
        }), use_container_width=True)
        
        # 6. SAMPLE RESULTS
        st.header("🔍 Sample Results")
        
        # Show some sample comparisons
        sample_size = st.slider("Number of samples to show:", 5, 20, 10)
        
        # Create display dataframe
        display_df = pd.DataFrame({
            'Actual_Life_Expectancy': df_clean[actual_col].head(sample_size),
            'Predicted_Life_Expectancy': df_clean[predicted_col].head(sample_size),
            'Actual_Risk': df_clean['Actual_Risk'].head(sample_size),
            'Predicted_Risk': df_clean['Predicted_Risk'].head(sample_size),
            'Match': df_clean['Actual_Risk'].head(sample_size) == df_clean['Predicted_Risk'].head(sample_size)
        })
        
        # Format the display
        st.dataframe(display_df.style.apply(
            lambda x: ['background-color: #C8E6C9' if v else 'background-color: #FFCDD2' 
                      for v in x] if x.name == 'Match' else [''] * len(x),
            axis=0
        ), use_container_width=True)
        
        # 7. DISTRIBUTION ANALYSIS
        st.header("📊 Distribution Patterns")
        
        # Create distribution comparison
        fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Pie charts comparison
        # Actual pie
        wedges1, texts1, autotexts1 = ax1.pie(actual_counts.values,
                                             labels=actual_counts.index,
                                             colors=[colors[risk] for risk in actual_counts.index],
                                             autopct='%1.1f%%',
                                             startangle=90)
        ax1.set_title('Actual Distribution', fontsize=13, fontweight='bold')
        
        # Predicted pie
        wedges2, texts2, autotexts2 = ax2.pie(predicted_counts.values,
                                             labels=predicted_counts.index,
                                             colors=[colors[risk] for risk in predicted_counts.index],
                                             autopct='%1.1f%%',
                                             startangle=90)
        ax2.set_title('Predicted Distribution', fontsize=13, fontweight='bold')
        
        # Make percentages bold
        for autotexts in [autotexts1, autotexts2]:
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        plt.tight_layout()
        st.pyplot(fig3)
        
        # 8. SUMMARY STATISTICS
        st.header("📈 Summary Statistics")
        
        # Calculate overall statistics
        total_samples = len(df_clean)
        correct_predictions = sum(df_clean['Actual_Risk'] == df_clean['Predicted_Risk'])
        overall_accuracy = correct_predictions / total_samples
        
        # Create summary dataframe
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
                'Correct Predictions': correct,
                'Accuracy': accuracy
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.loc[len(summary_df)] = {
            'Risk Level': 'OVERALL',
            'Actual Count': total_samples,
            'Predicted Count': total_samples,
            'Correct Predictions': correct_predictions,
            'Accuracy': overall_accuracy
        }
        
        # Display summary
        st.dataframe(summary_df.style.format({
            'Accuracy': '{:.2%}'
        }), use_container_width=True)
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.code(f"Error details: {e}", language='python')

else:
    # Show instructions
    st.info("👆 Upload your CSV file with Actual and Predicted Life Expectancy values")
    
    st.write("""
    ### Required Data Format:
    
    Your CSV should have **two numeric columns**:
    1. **Actual Life Expectancy** (from test set)
    2. **Predicted Life Expectancy** (from your model)
    
    **Example:**
    ```csv
    Actual_Life_Expectancy,Predicted_Life_Expectancy
    58.3,56.8
    72.1,71.5
    81.5,80.9
    65.8,67.2
    79.2,78.5
    ```
    
    **From your notebook, you can save results like this:**
    ```python
    # After training your model
    results = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred
    })
    results.to_csv('model_predictions.csv', index=False)
    ```
    
    **What will be shown:**
    1. Actual vs Predicted Risk Level counts
    2. Side-by-side visualizations
    3. Confusion matrix
    4. Classification performance report
    5. Sample comparisons
    6. Summary statistics
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Health Risk Level Analysis Dashboard</strong></p>
    <p>Comparing actual vs predicted risk classifications based on life expectancy predictions</p>
</div>
""", unsafe_allow_html=True)