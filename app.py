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
### This app analyzes Health Risk Level classifications from your notebook
**Risk categories based on Life Expectancy:**
- **High Risk**: Life Expectancy < 60 years
- **Medium Risk**: Life Expectancy 60-70 years  
- **Low Risk**: Life Expectancy > 70 years
""")

# 1. UPLOAD YOUR RESULTS DATA
st.header("📁 Upload Your Results Data")
uploaded_file = st.file_uploader("Upload CSV with Actual and Predicted values", type=['csv'])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Show preview
        with st.expander("👀 Preview your data"):
            st.dataframe(df.head(), use_container_width=True)
        
        # 2. SELECT OR CREATE COLUMNS
        st.header("🔍 Setup Risk Level Analysis")
        
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
        
        # Check what columns we have
        actual_cols = [col for col in df.columns if any(word in col.lower() 
                      for word in ['actual', 'true', 'y_test', 'life', 'test', 'real'])]
        pred_cols = [col for col in df.columns if any(word in col.lower() 
                     for word in ['predicted', 'pred', 'y_pred', 'predict', 'estimate'])]
        
        if actual_cols and pred_cols:
            # Use existing columns
            actual_col = actual_cols[0]
            predicted_col = pred_cols[0]
            
            # Check if columns are numeric
            if not (pd.api.types.is_numeric_dtype(df[actual_col]) and 
                   pd.api.types.is_numeric_dtype(df[predicted_col])):
                st.error("❌ Selected columns must contain numeric Life Expectancy values")
                st.stop()
            
            # Create risk levels
            df['Actual_Risk'] = df[actual_col].apply(risk_level)
            df['Predicted_Risk'] = df[predicted_col].apply(risk_level)
            
            st.info(f"Using columns: '{actual_col}' for Actual and '{predicted_col}' for Predicted")
        
        else:
            st.warning("Could not find clearly named Actual/Predicted columns.")
            st.write("**Please select your columns:**")
            
            col1, col2 = st.columns(2)
            with col1:
                actual_col = st.selectbox("Select Actual Life Expectancy column:", 
                                         df.columns)
            with col2:
                predicted_col = st.selectbox("Select Predicted Life Expectancy column:", 
                                            df.columns)
            
            if not (pd.api.types.is_numeric_dtype(df[actual_col]) and 
                   pd.api.types.is_numeric_dtype(df[predicted_col])):
                st.error("❌ Selected columns must contain numeric Life Expectancy values")
                st.stop()
            
            # Create risk levels
            df['Actual_Risk'] = df[actual_col].apply(risk_level)
            df['Predicted_Risk'] = df[predicted_col].apply(risk_level)
        
        # Remove any rows with NaN in risk columns
        df_clean = df.dropna(subset=['Actual_Risk', 'Predicted_Risk'])
        
        if len(df_clean) < len(df):
            st.warning(f"Removed {len(df) - len(df_clean)} rows with missing risk values")
        
        # 3. DISPLAY RISK LEVEL COUNTS
        st.header("📊 Risk Level Distribution")
        
        # Get counts
        actual_counts = df_clean['Actual_Risk'].value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0)
        predicted_counts = df_clean['Predicted_Risk'].value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0)
        
        # Create metrics
        col_high, col_medium, col_low = st.columns(3)
        with col_high:
            st.metric("Actual High Risk", actual_counts.get('High', 0))
        with col_medium:
            st.metric("Actual Medium Risk", actual_counts.get('Medium', 0))
        with col_low:
            st.metric("Actual Low Risk", actual_counts.get('Low', 0))
        
        # 4. VISUALIZATIONS
        st.header("📈 Visualizations")
        
        # Create tabs
        tab1, tab2 = st.tabs(["Risk Level Comparison", "Distribution Analysis"])
        
        with tab1:
            # ACTUAL VS PREDICTED RISK LEVELS (SIDE BY SIDE)
            st.subheader("Actual vs Predicted Health Risk Levels")
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Health Risk Level Analysis', fontsize=16, fontweight='bold')
            
            # Colors for risk levels
            colors = {'High': '#FF6B6B', 'Medium': '#FFD93D', 'Low': '#6BCF7F'}
            
            # 1. Actual Risk Distribution (Pie chart - Top Left)
            actual_pie_data = actual_counts
            actual_colors = [colors.get(risk, 'gray') for risk in actual_pie_data.index]
            
            wedges1, texts1, autotexts1 = ax1.pie(actual_pie_data.values, 
                                                 labels=actual_pie_data.index, 
                                                 colors=actual_colors,
                                                 autopct='%1.1f%%', 
                                                 startangle=90,
                                                 explode=[0.05, 0.05, 0.05])
            ax1.set_title('Actual Risk Distribution', fontsize=12, fontweight='bold')
            
            # Make percentages bold
            for autotext in autotexts1:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            # 2. Predicted Risk Distribution (Pie chart - Top Right)
            predicted_pie_data = predicted_counts
            predicted_colors = [colors.get(risk, 'gray') for risk in predicted_pie_data.index]
            
            wedges2, texts2, autotexts2 = ax2.pie(predicted_pie_data.values, 
                                                 labels=predicted_pie_data.index, 
                                                 colors=predicted_colors,
                                                 autopct='%1.1f%%', 
                                                 startangle=90,
                                                 explode=[0.05, 0.05, 0.05])
            ax2.set_title('Predicted Risk Distribution', fontsize=12, fontweight='bold')
            
            # Make percentages bold
            for autotext in autotexts2:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            # 3. Side-by-side Bar Chart (Bottom Left)
            x = np.arange(len(actual_counts))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, actual_counts.values, width, 
                           label='Actual', color=[colors.get(risk, 'gray') for risk in actual_counts.index],
                           alpha=0.8, edgecolor='black')
            
            bars2 = ax3.bar(x + width/2, predicted_counts.values, width, 
                           label='Predicted', color=[colors.get(risk, 'gray') for risk in predicted_counts.index],
                           alpha=0.6, edgecolor='black', hatch='//')
            
            ax3.set_xlabel('Risk Level', fontsize=11)
            ax3.set_ylabel('Count', fontsize=11)
            ax3.set_title('Actual vs Predicted Risk Counts', fontsize=12, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(['Low', 'Medium', 'High'])
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                            f'{int(height)}', ha='center', fontsize=9, fontweight='bold')
            
            # 4. Confusion Matrix Heatmap (Bottom Right)
            # Create confusion matrix
            risk_levels = ['Low', 'Medium', 'High']
            cm = confusion_matrix(df_clean['Actual_Risk'], df_clean['Predicted_Risk'], 
                                 labels=risk_levels, normalize='true')
            
            im = ax4.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
            ax4.figure.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
            
            # Set labels
            ax4.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=risk_levels, 
                   yticklabels=risk_levels,
                   title='Normalized Confusion Matrix',
                   ylabel='Actual Risk',
                   xlabel='Predicted Risk')
            
            # Rotate tick labels
            plt.setp(ax4.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax4.text(j, i, format(cm[i, j], '.2f'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black",
                            fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display classification report
            st.write("**Classification Report:**")
            report = classification_report(df_clean['Actual_Risk'], df_clean['Predicted_Risk'], 
                                         labels=risk_levels, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format({
                'precision': '{:.2f}',
                'recall': '{:.2f}',
                'f1-score': '{:.2f}',
                'support': '{:.0f}'
            }), use_container_width=True)
        
        with tab2:
            # DISTRIBUTION ANALYSIS (ONE ON TOP OF ANOTHER)
            st.subheader("Risk Level Distribution (Stacked View)")
            
            # Create figure for stacked/overlay visualizations
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # 1. Stacked Bar Chart (Top)
            x_labels = ['Low', 'Medium', 'High']
            x_pos = np.arange(len(x_labels))
            
            # Create stacked bars
            bottom_vals = np.zeros(len(x_labels))
            
            for risk in ['Low', 'Medium', 'High']:
                if risk in actual_counts.index:
                    count = actual_counts[risk]
                    ax1.bar(risk, count, color=colors[risk], alpha=0.7, 
                           edgecolor='black', label=f'Actual {risk}')
            
            for risk in ['Low', 'Medium', 'High']:
                if risk in predicted_counts.index:
                    count = predicted_counts[risk]
                    ax1.bar(risk, count, color=colors[risk], alpha=0.4, 
                           edgecolor='black', hatch='//', label=f'Predicted {risk}')
            
            ax1.set_xlabel('Risk Level', fontsize=12)
            ax1.set_ylabel('Count', fontsize=12)
            ax1.set_title('Risk Level Distribution (Side-by-side)', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for risk in x_labels:
                if risk in actual_counts.index:
                    height = actual_counts[risk]
                    ax1.text(x_pos[x_labels.index(risk)] - 0.15, height + 0.5, 
                            f'Actual: {int(height)}', ha='center', fontsize=9)
                if risk in predicted_counts.index:
                    height = predicted_counts[risk]
                    ax1.text(x_pos[x_labels.index(risk)] + 0.15, height + 0.5, 
                            f'Predicted: {int(height)}', ha='center', fontsize=9)
            
            # 2. Overlay Histogram (Bottom)
            # Create data for histogram
            actual_values = []
            predicted_values = []
            
            for risk in ['Low', 'Medium', 'High']:
                if risk in actual_counts.index:
                    actual_values.extend([risk] * int(actual_counts[risk]))
                if risk in predicted_counts.index:
                    predicted_values.extend([risk] * int(predicted_counts[risk]))
            
            # Create histogram
            bins = np.arange(4) - 0.5
            ax2.hist([actual_values, predicted_values], bins=bins, 
                    label=['Actual', 'Predicted'], 
                    color=['blue', 'orange'], alpha=0.7, edgecolor='black')
            
            ax2.set_xlabel('Risk Level', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Risk Level Frequency Distribution (Overlay)', fontsize=14, fontweight='bold')
            ax2.set_xticks([0, 1, 2])
            ax2.set_xticklabels(['Low', 'Medium', 'High'])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, risk in enumerate(['Low', 'Medium', 'High']):
                if risk in actual_counts.index:
                    height = actual_counts[risk]
                    ax2.text(i - 0.18, height + 0.5, str(int(height)), 
                            ha='center', fontsize=10, fontweight='bold', color='blue')
                if risk in predicted_counts.index:
                    height = predicted_counts[risk]
                    ax2.text(i + 0.18, height + 0.5, str(int(height)), 
                            ha='center', fontsize=10, fontweight='bold', color='orange')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show detailed comparison table
            st.write("**Detailed Risk Level Comparison:**")
            comparison_df = pd.DataFrame({
                'Risk Level': ['Low', 'Medium', 'High'],
                'Actual Count': [actual_counts.get('Low', 0), 
                                actual_counts.get('Medium', 0), 
                                actual_counts.get('High', 0)],
                'Predicted Count': [predicted_counts.get('Low', 0), 
                                   predicted_counts.get('Medium', 0), 
                                   predicted_counts.get('High', 0)],
                'Difference': [predicted_counts.get('Low', 0) - actual_counts.get('Low', 0),
                              predicted_counts.get('Medium', 0) - actual_counts.get('Medium', 0),
                              predicted_counts.get('High', 0) - actual_counts.get('High', 0)]
            })
            
            # Add percentage columns
            total_samples = len(df_clean)
            comparison_df['Actual %'] = (comparison_df['Actual Count'] / total_samples * 100).round(1)
            comparison_df['Predicted %'] = (comparison_df['Predicted Count'] / total_samples * 100).round(1)
            
            st.dataframe(comparison_df, use_container_width=True)
        
        # 5. MISCLASSIFICATION ANALYSIS
        st.header("🔍 Misclassification Analysis")
        
        # Identify misclassifications
        df_clean['Correct'] = df_clean['Actual_Risk'] == df_clean['Predicted_Risk']
        df_clean['Error_Type'] = df_clean.apply(
            lambda row: f"{row['Actual_Risk']} → {row['Predicted_Risk']}" 
            if row['Actual_Risk'] != row['Predicted_Risk'] else "Correct", 
            axis=1
        )
        
        # Calculate overall accuracy
        accuracy = df_clean['Correct'].mean()
        
        col_acc1, col_acc2 = st.columns(2)
        with col_acc1:
            st.metric("Overall Accuracy", f"{accuracy:.1%}")
        with col_acc2:
            st.metric("Misclassified Samples", f"{sum(~df_clean['Correct'])}")
        
        # Show misclassification patterns
        misclass_counts = df_clean[df_clean['Error_Type'] != 'Correct']['Error_Type'].value_counts()
        
        if len(misclass_counts) > 0:
            st.write("**Misclassification Patterns:**")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bars = ax.barh(misclass_counts.index, misclass_counts.values, 
                          color='#FF6B6B', alpha=0.7, edgecolor='black')
            
            ax.set_xlabel('Number of Samples', fontsize=11)
            ax.set_title('Types of Misclassifications', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                       f'{int(width)}', ha='left', va='center', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show sample misclassifications
            with st.expander("View Sample Misclassified Cases"):
                misclassified = df_clean[~df_clean['Correct']].head(10)
                display_cols = ['Actual_Risk', 'Predicted_Risk', 'Error_Type']
                if actual_col in df_clean.columns:
                    display_cols.insert(0, actual_col)
                if predicted_col in df_clean.columns:
                    display_cols.insert(1, predicted_col)
                
                st.dataframe(misclassified[display_cols], use_container_width=True)
        
        # 6. DOWNLOAD RESULTS
        st.header("💾 Download Analysis Results")
        
        # Create downloadable results
        results_csv = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Analysis Results",
            data=results_csv,
            file_name="risk_level_analysis_results.csv",
            mime="text/csv",
            help="Includes Actual and Predicted Risk Levels with error analysis"
        )
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.code(f"Error details: {e}", language='python')

else:
    # Show instructions
    st.info("👆 Upload your CSV file with Actual and Predicted Life Expectancy values")
    
    st.write("""
    ### Expected Data Format:
    
    **Your CSV should include:**
    - **Actual Life Expectancy** column (numeric values)
    - **Predicted Life Expectancy** column (numeric values)
    
    **Example format:**
    | Actual_Life_Expectancy | Predicted_Life_Expectancy |
    |------------------------|---------------------------|
    | 58.3 | 56.8 |
    | 72.1 | 71.5 |
    | 81.5 | 80.9 |
    | 65.8 | 67.2 |
    | 79.2 | 78.5 |
    
    **The app will:**
    1. Convert Life Expectancy values to Risk Levels
    2. Compare Actual vs Predicted Risk Levels
    3. Show distribution comparisons
    4. Analyze misclassifications
    
    **Risk Level Definition (from your notebook):**
    - **High Risk**: Life Expectancy < 60 years
    - **Medium Risk**: Life Expectancy 60-70 years  
    - **Low Risk**: Life Expectancy > 70 years
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Health Risk Level Analysis Tool</strong></p>
    <p>Converts Life Expectancy predictions to risk categories and compares with actual classifications</p>
</div>
""", unsafe_allow_html=True)