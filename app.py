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

# 1. UPLOAD YOUR DATA
st.header("📁 Upload Your Dataset")
uploaded_file = st.file_uploader("Upload your dataset CSV", type=['csv'])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Show preview
        with st.expander("👀 Preview your data"):
            st.dataframe(df.head(), use_container_width=True)
        
        # 2. SETUP RISK LEVELS
        st.header("🔧 Setup Risk Levels")
        
        # Find Life Expectancy column
        life_cols = [col for col in df.columns if any(word in col.lower() 
                     for word in ['life expectancy', 'life_expectancy', 'life', 'expectancy'])]
        
        if not life_cols:
            st.error("❌ Could not find Life Expectancy column")
            st.write("**Please select your Life Expectancy column:**")
            life_col = st.selectbox("Select Life Expectancy column:", df.columns)
        else:
            life_col = life_cols[0]
            st.info(f"Using column: '{life_col}' for Life Expectancy")
        
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
        
        # Create risk levels
        df['Risk_Level'] = df[life_col].apply(risk_level)
        
        # Remove any rows with NaN
        df_clean = df.dropna(subset=['Risk_Level'])
        
        # Get risk counts
        risk_counts = df_clean['Risk_Level'].value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0)
        
        # Display counts
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Low Risk", risk_counts.get('Low', 0))
        with col2:
            st.metric("Medium Risk", risk_counts.get('Medium', 0))
        with col3:
            st.metric("High Risk", risk_counts.get('High', 0))
        
        # 3. VISUALIZATIONS
        st.header("📈 Health Risk Level Analysis")
        
        # Create tabs
        tab1, tab2 = st.tabs(["Risk Level Distribution", "Comparative Analysis"])
        
        with tab1:
            # SINGLE DISTRIBUTION (ACTUAL RISK LEVELS)
            st.subheader("Actual Health Risk Level Distribution")
            
            # Create figure for actual risk levels
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Health Risk Level Distribution Analysis', fontsize=16, fontweight='bold')
            
            # Colors for risk levels
            colors = {'High': '#FF6B6B', 'Medium': '#FFD93D', 'Low': '#6BCF7F'}
            
            # 1. Pie Chart (Top Left)
            wedges1, texts1, autotexts1 = ax1.pie(risk_counts.values, 
                                                 labels=risk_counts.index, 
                                                 colors=[colors[risk] for risk in risk_counts.index],
                                                 autopct='%1.1f%%', 
                                                 startangle=90,
                                                 explode=[0.05, 0.05, 0.05])
            ax1.set_title('Risk Level Distribution (Pie)', fontsize=12, fontweight='bold')
            
            # Make percentages bold and white
            for autotext in autotexts1:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            # 2. Bar Chart (Top Right)
            bars2 = ax2.bar(risk_counts.index, risk_counts.values, 
                           color=[colors[risk] for risk in risk_counts.index],
                           alpha=0.8, edgecolor='black')
            ax2.set_xlabel('Risk Level', fontsize=11)
            ax2.set_ylabel('Count', fontsize=11)
            ax2.set_title('Risk Level Distribution (Bar)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                        f'{int(height)}', ha='center', fontsize=10, fontweight='bold')
            
            # 3. Donut Chart (Bottom Left)
            wedges3, texts3, autotexts3 = ax3.pie(risk_counts.values, 
                                                 labels=risk_counts.index, 
                                                 colors=[colors[risk] for risk in risk_counts.index],
                                                 autopct='%1.1f%%', 
                                                 startangle=90,
                                                 wedgeprops=dict(width=0.3, edgecolor='w'))
            ax3.set_title('Risk Level Distribution (Donut)', fontsize=12, fontweight='bold')
            
            # Make percentages bold and white
            for autotext in autotexts3:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            # 4. Horizontal Bar Chart (Bottom Right)
            bars4 = ax4.barh(range(len(risk_counts)), risk_counts.values, 
                            color=[colors[risk] for risk in risk_counts.index],
                            alpha=0.8, edgecolor='black')
            ax4.set_yticks(range(len(risk_counts)))
            ax4.set_yticklabels(risk_counts.index)
            ax4.set_xlabel('Count', fontsize=11)
            ax4.set_title('Risk Level Distribution (Horizontal Bar)', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, bar in enumerate(bars4):
                width = bar.get_width()
                ax4.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{int(width)}', ha='left', va='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show data table
            st.write("**Risk Level Summary:**")
            summary_df = pd.DataFrame({
                'Risk Level': risk_counts.index,
                'Count': risk_counts.values,
                'Percentage': (risk_counts.values / len(df_clean) * 100).round(1)
            })
            st.dataframe(summary_df, use_container_width=True)
        
        with tab2:
            # COMPARATIVE ANALYSIS (IF PREDICTED VALUES EXIST)
            st.subheader("Comparative Analysis")
            
            # Check if we have predicted values
            predicted_cols = [col for col in df.columns if any(word in col.lower() 
                            for word in ['predicted', 'pred', 'y_pred', 'predict', 'estimate'])]
            
            if predicted_cols:
                predicted_col = predicted_cols[0]
                
                # Create predicted risk levels
                df_clean['Predicted_Risk'] = df_clean[predicted_col].apply(risk_level)
                
                # Get predicted risk counts
                predicted_counts = df_clean['Predicted_Risk'].value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0)
                
                # Create figure for comparison
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle('Actual vs Predicted Risk Level Comparison', fontsize=16, fontweight='bold')
                
                # 1. Side-by-side Bar Chart (Top Left)
                x = np.arange(len(risk_counts))
                width = 0.35
                
                bars1 = ax1.bar(x - width/2, risk_counts.values, width, 
                               label='Actual', color=[colors[risk] for risk in risk_counts.index],
                               alpha=0.8, edgecolor='black')
                
                bars2 = ax1.bar(x + width/2, predicted_counts.values, width, 
                               label='Predicted', color=[colors[risk] for risk in predicted_counts.index],
                               alpha=0.6, edgecolor='black', hatch='//')
                
                ax1.set_xlabel('Risk Level', fontsize=11)
                ax1.set_ylabel('Count', fontsize=11)
                ax1.set_title('Actual vs Predicted Risk Counts', fontsize=12, fontweight='bold')
                ax1.set_xticks(x)
                ax1.set_xticklabels(['Low', 'Medium', 'High'])
                ax1.legend()
                ax1.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                                f'{int(height)}', ha='center', fontsize=9, fontweight='bold')
                
                # 2. Stacked Bar Chart (Top Right)
                bottom_vals = np.zeros(len(risk_counts))
                
                for i, risk in enumerate(['Low', 'Medium', 'High']):
                    if risk in risk_counts.index:
                        count = risk_counts[risk]
                        ax2.bar(risk, count, bottom=bottom_vals[i], 
                               color=colors[risk], alpha=0.8, 
                               edgecolor='black', label=f'Actual {risk}')
                        bottom_vals[i] += count
                
                bottom_vals = np.zeros(len(predicted_counts))
                
                for i, risk in enumerate(['Low', 'Medium', 'High']):
                    if risk in predicted_counts.index:
                        count = predicted_counts[risk]
                        ax2.bar(risk, count, bottom=bottom_vals[i], 
                               color=colors[risk], alpha=0.4, 
                               edgecolor='black', hatch='//', label=f'Predicted {risk}')
                        bottom_vals[i] += count
                
                ax2.set_xlabel('Risk Level', fontsize=11)
                ax2.set_ylabel('Count', fontsize=11)
                ax2.set_title('Stacked Risk Level Distribution', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')
                
                # 3. Grouped Bar Chart (Bottom Left)
                x_pos = np.arange(len(risk_counts))
                
                ax3.bar(x_pos - 0.2, risk_counts.values, 0.4, 
                       color=[colors[risk] for risk in risk_counts.index],
                       alpha=0.8, label='Actual')
                ax3.bar(x_pos + 0.2, predicted_counts.values, 0.4, 
                       color=[colors[risk] for risk in predicted_counts.index],
                       alpha=0.8, label='Predicted')
                
                ax3.set_xlabel('Risk Level', fontsize=11)
                ax3.set_ylabel('Count', fontsize=11)
                ax3.set_title('Grouped Risk Level Comparison', fontsize=12, fontweight='bold')
                ax3.set_xticks(x_pos)
                ax3.set_xticklabels(['Low', 'Medium', 'High'])
                ax3.legend()
                ax3.grid(True, alpha=0.3, axis='y')
                
                # 4. Line Plot Comparison (Bottom Right)
                ax4.plot(risk_counts.index, risk_counts.values, 'o-', 
                        label='Actual', linewidth=2, markersize=8, color='blue')
                ax4.plot(predicted_counts.index, predicted_counts.values, 's--', 
                        label='Predicted', linewidth=2, markersize=8, color='red')
                
                ax4.set_xlabel('Risk Level', fontsize=11)
                ax4.set_ylabel('Count', fontsize=11)
                ax4.set_title('Risk Level Trend Comparison', fontsize=12, fontweight='bold')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show comparison table
                st.write("**Actual vs Predicted Comparison:**")
                comparison_df = pd.DataFrame({
                    'Risk Level': ['Low', 'Medium', 'High'],
                    'Actual Count': [risk_counts.get('Low', 0), 
                                    risk_counts.get('Medium', 0), 
                                    risk_counts.get('High', 0)],
                    'Predicted Count': [predicted_counts.get('Low', 0), 
                                       predicted_counts.get('Medium', 0), 
                                       predicted_counts.get('High', 0)],
                    'Difference': [predicted_counts.get('Low', 0) - risk_counts.get('Low', 0),
                                  predicted_counts.get('Medium', 0) - risk_counts.get('Medium', 0),
                                  predicted_counts.get('High', 0) - risk_counts.get('High', 0)]
                })
                
                st.dataframe(comparison_df, use_container_width=True)
                
            else:
                st.info("No predicted values found in dataset. Showing only actual risk level analysis.")
                
                # Create a dummy comparison for visualization
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle('Risk Level Distribution Patterns', fontsize=16, fontweight='bold')
                
                # Show different visualizations of the same data
                # 1. Vertical Bars
                bars1 = ax1.bar(risk_counts.index, risk_counts.values, 
                               color=[colors[risk] for risk in risk_counts.index],
                               alpha=0.8, edgecolor='black')
                ax1.set_title('Vertical Distribution', fontsize=12)
                ax1.set_ylabel('Count')
                ax1.grid(True, alpha=0.3)
                
                # 2. Horizontal Bars
                bars2 = ax2.barh(range(len(risk_counts)), risk_counts.values, 
                               color=[colors[risk] for risk in risk_counts.index],
                               alpha=0.8, edgecolor='black')
                ax2.set_yticks(range(len(risk_counts)))
                ax2.set_yticklabels(risk_counts.index)
                ax2.set_title('Horizontal Distribution', fontsize=12)
                ax2.set_xlabel('Count')
                ax2.grid(True, alpha=0.3)
                
                # 3. Area Chart
                ax3.fill_between(range(len(risk_counts)), risk_counts.values, 
                                alpha=0.5, color='skyblue')
                ax3.plot(range(len(risk_counts)), risk_counts.values, 
                        'o-', color='blue', linewidth=2)
                ax3.set_xticks(range(len(risk_counts)))
                ax3.set_xticklabels(risk_counts.index)
                ax3.set_title('Area Distribution', fontsize=12)
                ax3.set_ylabel('Count')
                ax3.grid(True, alpha=0.3)
                
                # 4. Radar Chart
                angles = np.linspace(0, 2*np.pi, len(risk_counts), endpoint=False).tolist()
                values = risk_counts.values.tolist()
                values += values[:1]  # Close the loop
                angles += angles[:1]  # Close the loop
                
                ax4 = plt.subplot(2, 2, 4, projection='polar')
                ax4.plot(angles, values, 'o-', linewidth=2)
                ax4.fill(angles, values, alpha=0.25)
                ax4.set_xticks(angles[:-1])
                ax4.set_xticklabels(risk_counts.index)
                ax4.set_title('Radar Distribution', fontsize=12)
                ax4.grid(True)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        # 4. DOWNLOAD RESULTS
        st.header("💾 Download Results")
        
        # Create downloadable data
        results_csv = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Risk Level Analysis",
            data=results_csv,
            file_name="risk_level_analysis.csv",
            mime="text/csv",
            help="Includes original data with calculated Risk Levels"
        )
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.code(f"Error details: {e}", language='python')

else:
    # Show instructions
    st.info("👆 Upload your dataset CSV file")
    
    st.write("""
    ### Expected Dataset Format:
    
    **Your CSV should include:**
    - A **Life Expectancy** column (numeric values)
    - Optionally a **Predicted Life Expectancy** column for comparison
    
    **Example format:**
    | Country | Year | Life Expectancy | Predicted_Life_Expectancy | ...other features |
    |---------|------|-----------------|---------------------------|-------------------|
    | Afghanistan | 1990 | 50.331 | 50.5 | ... |
    | Afghanistan | 1991 | 50.999 | 51.2 | ... |
    
    **The app will:**
    1. Convert Life Expectancy to Risk Levels
    2. Show distribution of Risk Levels
    3. If predicted values exist, compare Actual vs Predicted Risk Levels
    
    **Risk Level Definition:**
    - **High Risk**: Life Expectancy < 60 years
    - **Medium Risk**: Life Expectancy 60-70 years  
    - **Low Risk**: Life Expectancy > 70 years
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Health Risk Level Analysis</strong></p>
    <p>Visualizing and comparing health risk categories based on life expectancy</p>
</div>
""", unsafe_allow_html=True)