# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Page setup
st.set_page_config(page_title="Life Expectancy Risk Analysis", layout="wide")
st.title("📊 Life Expectancy Risk Level Analysis")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

# 1. LOAD DATASET
st.header("📁 Load Dataset")
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=['csv'])

if uploaded_file is not None:
    try:
        # Load the dataset
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        
        st.success(f"✅ Dataset loaded successfully!")
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Show dataset preview
        with st.expander("👀 View first 10 rows"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Show column names
        st.write("**Available columns:**")
        st.write(", ".join(df.columns.tolist()))
        
        # Find life expectancy column
        life_exp_columns = [col for col in df.columns if any(keyword in col.lower() 
                           for keyword in ['life', 'expectancy', 'lifexp', 'life_exp'])]
        
        if life_exp_columns:
            life_exp_col = life_exp_columns[0]
            st.info(f"🔍 Detected life expectancy column: **{life_exp_col}**")
            
            if pd.api.types.is_numeric_dtype(df[life_exp_col]):
                # Process the data
                df_processed = df.copy()
                
                # 2. CREATE RISK CATEGORIES BASED ON LIFE EXPECTANCY
                st.header("🎯 Risk Categories")
                
                # Define risk categories based on life expectancy
                df_processed['Actual_Risk'] = pd.cut(
                    df_processed[life_exp_col],
                    bins=[0, 60, 75, 100],
                    labels=['High Risk', 'Medium Risk', 'Low Risk']
                )
                
                # Show distribution
                risk_counts = df_processed['Actual_Risk'].value_counts()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("High Risk", risk_counts.get('High Risk', 0))
                with col2:
                    st.metric("Medium Risk", risk_counts.get('Medium Risk', 0))
                with col3:
                    st.metric("Low Risk", risk_counts.get('Low Risk', 0))
                
                # Show risk classification rules
                with st.expander("📋 Risk Classification Rules"):
                    st.write("""
                    **Risk Levels based on Life Expectancy:**
                    - **High Risk**: Life Expectancy < 60 years
                    - **Medium Risk**: 60 ≤ Life Expectancy ≤ 75 years  
                    - **Low Risk**: Life Expectancy > 75 years
                    """)
                
                # 3. GENERATE PREDICTIONS (Simple simulation)
                st.header("🔮 Generate Predictions")
                
                if st.button("Generate Risk Predictions", type="primary"):
                    with st.spinner("Generating predictions..."):
                        # Simulate predictions (85% accuracy)
                        np.random.seed(42)
                        n = len(df_processed)
                        
                        # Start with actual values
                        predictions = df_processed['Actual_Risk'].copy()
                        
                        # Introduce some prediction errors (15% error rate)
                        wrong_count = int(0.15 * n)
                        wrong_indices = np.random.choice(n, wrong_count, replace=False)
                        
                        for idx in wrong_indices:
                            actual = df_processed.loc[idx, 'Actual_Risk']
                            # Predict a different risk level
                            if actual == 'High Risk':
                                predictions[idx] = np.random.choice(['Medium Risk', 'Low Risk'])
                            elif actual == 'Medium Risk':
                                predictions[idx] = np.random.choice(['High Risk', 'Low Risk'])
                            else:  # Low Risk
                                predictions[idx] = np.random.choice(['High Risk', 'Medium Risk'])
                        
                        df_processed['Predicted_Risk'] = predictions
                        
                        # Store in session state
                        st.session_state.df_processed = df_processed
                        
                        st.success("✅ Predictions generated successfully!")
                        
                        # Show sample of predictions
                        with st.expander("📊 View sample predictions"):
                            sample_cols = [life_exp_col, 'Actual_Risk', 'Predicted_Risk']
                            sample_df = df_processed[sample_cols].head(15).copy()
                            sample_df['Match'] = sample_df['Actual_Risk'] == sample_df['Predicted_Risk']
                            st.dataframe(sample_df, use_container_width=True)
                        
                        # 4. VISUALIZATIONS
                        st.header("📈 Visualizations")
                        
                        categories = ['High Risk', 'Medium Risk', 'Low Risk']
                        actual_counts = df_processed['Actual_Risk'].value_counts().reindex(categories, fill_value=0)
                        predicted_counts = df_processed['Predicted_Risk'].value_counts().reindex(categories, fill_value=0)
                        
                        # Tab layout for visualizations
                        tab1, tab2, tab3 = st.tabs(["Predicted vs Actual", "Risk Distribution", "Classification Report"])
                        
                        with tab1:
                            # BAR GRAPH - Predicted vs Actual
                            st.subheader("Predicted vs Actual Risk Levels")
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            x = np.arange(len(categories))
                            width = 0.35
                            
                            # Create bars
                            bars_actual = ax.bar(x - width/2, actual_counts.values, width, 
                                                label='Actual', color='#2196F3', alpha=0.8)
                            bars_predicted = ax.bar(x + width/2, predicted_counts.values, width, 
                                                   label='Predicted', color='#FF5722', alpha=0.8)
                            
                            # Customize the chart
                            ax.set_xlabel('Risk Level', fontsize=12)
                            ax.set_ylabel('Count', fontsize=12)
                            ax.set_title('Predicted vs Actual Risk Levels', fontsize=14, fontweight='bold')
                            ax.set_xticks(x)
                            ax.set_xticklabels(['High', 'Medium', 'Low'])
                            ax.legend()
                            ax.grid(True, alpha=0.3, linestyle='--')
                            
                            # Add value labels on bars
                            def add_value_labels(bars):
                                for bar in bars:
                                    height = bar.get_height()
                                    ax.annotate(f'{int(height)}',
                                               xy=(bar.get_x() + bar.get_width() / 2, height),
                                               xytext=(0, 3),
                                               textcoords="offset points",
                                               ha='center', va='bottom',
                                               fontsize=10)
                            
                            add_value_labels(bars_actual)
                            add_value_labels(bars_predicted)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        with tab2:
                            # BAR GRAPH - Risk Distribution (ACTUAL ONLY)
                            st.subheader("Risk Distribution")
                            
                            # Create two columns for different visualizations
                            col_vis1, col_vis2 = st.columns(2)
                            
                            with col_vis1:
                                # Bar Chart
                                st.write("**Bar Chart**")
                                fig, ax = plt.subplots(figsize=(8, 6))
                                
                                # Define colors for each risk level
                                colors = ['#FF5252', '#FFA726', '#66BB6A']  # Red, Orange, Green
                                
                                bars = ax.bar(actual_counts.index, actual_counts.values, 
                                             color=colors, alpha=0.8, edgecolor='black', linewidth=1)
                                
                                # Customize the chart
                                ax.set_xlabel('Risk Level', fontsize=12)
                                ax.set_ylabel('Count', fontsize=12)
                                ax.set_title('Risk Distribution', fontsize=14, fontweight='bold')
                                ax.set_xticklabels(['High\n(<60 years)', 'Medium\n(60-75 years)', 'Low\n(>75 years)'])
                                ax.grid(True, alpha=0.3, axis='y')
                                
                                # Add value labels on bars
                                for bar in bars:
                                    height = bar.get_height()
                                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                                           f'{int(height)}', ha='center', va='bottom',
                                           fontsize=11, fontweight='bold')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                            
                            with col_vis2:
                                # Pie Chart
                                st.write("**Pie Chart**")
                                fig, ax = plt.subplots(figsize=(8, 6))
                                
                                colors = ['#FF5252', '#FFA726', '#66BB6A']
                                explode = (0.05, 0.05, 0.05)  # Slight explode for all slices
                                
                                wedges, texts, autotexts = ax.pie(
                                    actual_counts.values,
                                    labels=actual_counts.index,
                                    autopct='%1.1f%%',
                                    colors=colors,
                                    explode=explode,
                                    startangle=90,
                                    shadow=True
                                )
                                
                                # Customize the pie chart
                                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                                ax.set_title('Risk Distribution Percentage', fontsize=14, fontweight='bold')
                                
                                # Style the text
                                for autotext in autotexts:
                                    autotext.set_color('white')
                                    autotext.set_fontweight('bold')
                                    autotext.set_fontsize(11)
                                
                                for text in texts:
                                    text.set_fontsize(11)
                                
                                st.pyplot(fig)
                            
                            # Statistics section
                            st.write("---")
                            st.write("**Distribution Statistics:**")
                            
                            # Calculate percentages
                            total = actual_counts.sum()
                            percentages = (actual_counts / total * 100).round(1)
                            
                            col_stat1, col_stat2, col_stat3 = st.columns(3)
                            
                            with col_stat1:
                                st.markdown("**High Risk**")
                                st.markdown(f"<h3 style='color: #FF5252;'>{actual_counts['High Risk']}</h3>", 
                                           unsafe_allow_html=True)
                                st.write(f"{percentages['High Risk']}% of total")
                                st.write("Life Expectancy: < 60 years")
                            
                            with col_stat2:
                                st.markdown("**Medium Risk**")
                                st.markdown(f"<h3 style='color: #FFA726;'>{actual_counts['Medium Risk']}</h3>", 
                                           unsafe_allow_html=True)
                                st.write(f"{percentages['Medium Risk']}% of total")
                                st.write("Life Expectancy: 60-75 years")
                            
                            with col_stat3:
                                st.markdown("**Low Risk**")
                                st.markdown(f"<h3 style='color: #66BB6A;'>{actual_counts['Low Risk']}</h3>", 
                                           unsafe_allow_html=True)
                                st.write(f"{percentages['Low Risk']}% of total")
                                st.write("Life Expectancy: > 75 years")
                            
                            # Summary
                            st.write("---")
                            st.write("**Summary:**")
                            dominant_risk = actual_counts.idxmax()
                            dominant_percentage = percentages[dominant_risk]
                            st.write(f"The most common risk level is **{dominant_risk}** with **{dominant_percentage}%** of the dataset.")
                        
                        with tab3:
                            # CLASSIFICATION REPORT
                            st.subheader("Classification Report")
                            
                            # Calculate confusion matrix
                            cm = confusion_matrix(df_processed['Actual_Risk'], 
                                                  df_processed['Predicted_Risk'], 
                                                  labels=categories)
                            
                            # Create two columns for visualizations
                            col_report1, col_report2 = st.columns([1, 2])
                            
                            with col_report1:
                                # Confusion Matrix Heatmap
                                st.write("**Confusion Matrix**")
                                fig, ax = plt.subplots(figsize=(6, 5))
                                
                                sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                                           xticklabels=['High', 'Medium', 'Low'],
                                           yticklabels=['High', 'Medium', 'Low'],
                                           ax=ax)
                                
                                ax.set_xlabel('Predicted Risk', fontsize=11)
                                ax.set_ylabel('Actual Risk', fontsize=11)
                                ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
                                
                                st.pyplot(fig)
                            
                            with col_report2:
                                # Classification Report Table
                                st.write("**Classification Metrics**")
                                
                                # Generate classification report
                                report = classification_report(df_processed['Actual_Risk'], 
                                                              df_processed['Predicted_Risk'],
                                                              output_dict=True,
                                                              zero_division=0)
                                
                                # Convert to DataFrame
                                report_df = pd.DataFrame(report).transpose()
                                
                                # Display styled table
                                styled_df = report_df.style.format({
                                    'precision': '{:.3f}',
                                    'recall': '{:.3f}',
                                    'f1-score': '{:.3f}',
                                    'support': '{:.0f}'
                                }).background_gradient(subset=['precision', 'recall', 'f1-score'], 
                                                      cmap='RdYlGn', 
                                                      vmin=0, vmax=1)
                                
                                st.dataframe(styled_df, use_container_width=True)
                            
                            # Metrics Visualization
                            st.write("**Metrics by Risk Level**")
                            
                            # Extract metrics for visualization
                            metrics_data = []
                            for class_name in categories:
                                if class_name in report:
                                    metrics_data.append({
                                        'Risk Level': class_name,
                                        'Precision': report[class_name]['precision'],
                                        'Recall': report[class_name]['recall'],
                                        'F1-Score': report[class_name]['f1-score']
                                    })
                            
                            if metrics_data:
                                metrics_df = pd.DataFrame(metrics_data)
                                
                                fig, ax = plt.subplots(figsize=(10, 5))
                                x = np.arange(len(categories))
                                width = 0.25
                                
                                bars1 = ax.bar(x - width, metrics_df['Precision'], width, 
                                              label='Precision', color='#FF9800')
                                bars2 = ax.bar(x, metrics_df['Recall'], width, 
                                              label='Recall', color='#2196F3')
                                bars3 = ax.bar(x + width, metrics_df['F1-Score'], width, 
                                              label='F1-Score', color='#4CAF50')
                                
                                ax.set_xlabel('Risk Level', fontsize=11)
                                ax.set_ylabel('Score', fontsize=11)
                                ax.set_title('Performance Metrics by Risk Level', fontsize=13, fontweight='bold')
                                ax.set_xticks(x)
                                ax.set_xticklabels(['High', 'Medium', 'Low'])
                                ax.set_ylim([0, 1])
                                ax.legend()
                                ax.grid(True, alpha=0.3, axis='y')
                                
                                # Add value labels
                                def add_labels_to_bars(bars):
                                    for bar in bars:
                                        height = bar.get_height()
                                        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                                               f'{height:.2f}', ha='center', va='bottom', fontsize=9)
                                
                                add_labels_to_bars(bars1)
                                add_labels_to_bars(bars2)
                                add_labels_to_bars(bars3)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                        
                        # Show summary statistics
                        st.header("📊 Summary Statistics")
                        
                        col_sum1, col_sum2, col_sum3 = st.columns(3)
                        
                        with col_sum1:
                            accuracy = (df_processed['Actual_Risk'] == df_processed['Predicted_Risk']).mean()
                            st.metric("Overall Accuracy", f"{accuracy:.1%}")
                        
                        with col_sum2:
                            total_mismatches = (df_processed['Actual_Risk'] != df_processed['Predicted_Risk']).sum()
                            st.metric("Total Mismatches", total_mismatches)
                        
                        with col_sum3:
                            if 'Country' in df_processed.columns:
                                country_count = df_processed['Country'].nunique()
                                st.metric("Countries", country_count)
                            else:
                                st.metric("Total Samples", len(df_processed))
            
            else:
                st.error(f"❌ Column '{life_exp_col}' is not numeric. Please upload a dataset with numeric life expectancy values.")
        
        else:
            st.warning("⚠️ No life expectancy column found. Please ensure your dataset has a column with life expectancy data.")
            st.info("💡 Try columns with names like: 'Life_Expectancy', 'life_expectancy', 'LifeExp', 'Longevity'")
    
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")

else:
    # Show instructions when no file is uploaded
    st.info("👆 Please upload a CSV file to begin analysis")
    
    # Show example dataset structure
    st.write("**Example dataset structure:**")
    
    example_data = pd.DataFrame({
        'Country': ['United States', 'United Kingdom', 'Japan', 'India', 'Brazil'],
        'Life_Expectancy': [78.5, 81.2, 84.6, 69.7, 75.9],
        'Region': ['North America', 'Europe', 'Asia', 'Asia', 'South America']
    })
    
    st.dataframe(example_data, use_container_width=True)
    
    st.write("**How it works:**")
    st.write("""
    1. Upload a CSV file with a life expectancy column
    2. The app automatically creates risk categories:
       - **High Risk**: Life Expectancy < 60 years
       - **Medium Risk**: 60-75 years
       - **Low Risk**: > 75 years
    3. Generates sample predictions
    4. Shows visualizations:
       - **Tab 1**: Predicted vs Actual comparison
       - **Tab 2**: Risk Distribution (bar and pie charts)
       - **Tab 3**: Classification Report with metrics
    """)