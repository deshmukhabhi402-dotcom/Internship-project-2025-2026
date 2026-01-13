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
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

# Instructions
st.markdown("""
### Instructions:
1. **Prepare your CSV file in your notebook** with:
   - Your life expectancy column (e.g., 'Life_Expectancy')
   - Your actual risk column (e.g., 'Actual_Risk' - categories: High Risk, Medium Risk, Low Risk)
   - Your predicted risk column (e.g., 'Predicted_Risk')
2. **Export/save as CSV** from your notebook
3. **Upload it here** to see the same visualizations
""")

# 1. LOAD DATASET WITH PREDICTIONS
st.header("📁 Upload Your Model Results")
uploaded_file = st.file_uploader("Upload your CSV file with predictions", type=['csv'])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        
        st.success(f"✅ Dataset loaded successfully!")
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Show dataset preview
        with st.expander("👀 View first 10 rows"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Show column names
        st.write("**Available columns in your dataset:**")
        st.write(", ".join(df.columns.tolist()))
        
        # 2. CHECK FOR REQUIRED COLUMNS
        st.header("🔍 Check Required Columns")
        
        # Find columns automatically
        life_exp_col = None
        actual_risk_col = None
        predicted_risk_col = None
        
        # Try to find life expectancy column
        life_exp_candidates = [col for col in df.columns if any(keyword in col.lower() 
                           for keyword in ['life', 'expectancy', 'lifexp', 'life_exp'])]
        
        # Try to find risk columns
        actual_risk_candidates = [col for col in df.columns if any(keyword in col.lower() 
                               for keyword in ['actual', 'true', 'real', 'risk'])]
        
        predicted_risk_candidates = [col for col in df.columns if any(keyword in col.lower() 
                                  for keyword in ['predict', 'predicted', 'forecast', 'estimated'])]
        
        # Select boxes for columns
        col_select1, col_select2, col_select3 = st.columns(3)
        
        with col_select1:
            if life_exp_candidates:
                life_exp_col = st.selectbox(
                    "Select Life Expectancy column:",
                    df.columns,
                    index=list(df.columns).index(life_exp_candidates[0])
                )
            else:
                life_exp_col = st.selectbox(
                    "Select Life Expectancy column:",
                    df.columns
                )
        
        with col_select2:
            if actual_risk_candidates:
                actual_risk_col = st.selectbox(
                    "Select Actual Risk column:",
                    df.columns,
                    index=list(df.columns).index(actual_risk_candidates[0])
                )
            else:
                actual_risk_col = st.selectbox(
                    "Select Actual Risk column:",
                    df.columns
                )
        
        with col_select3:
            if predicted_risk_candidates:
                predicted_risk_col = st.selectbox(
                    "Select Predicted Risk column:",
                    df.columns,
                    index=list(df.columns).index(predicted_risk_candidates[0])
                )
            else:
                predicted_risk_col = st.selectbox(
                    "Select Predicted Risk column:",
                    df.columns
                )
        
        # Verify column selections
        if life_exp_col and actual_risk_col and predicted_risk_col:
            st.info(f"""
            **Selected columns:**
            - Life Expectancy: `{life_exp_col}`
            - Actual Risk: `{actual_risk_col}`
            - Predicted Risk: `{predicted_risk_col}`
            """)
            
            # Check if data is valid
            valid_data = True
            
            # Check if risk columns contain expected values
            risk_values = ['High Risk', 'Medium Risk', 'Low Risk']
            
            actual_unique = df[actual_risk_col].unique()
            predicted_unique = df[predicted_risk_col].unique()
            
            # Convert to string and check
            actual_risk_str = df[actual_risk_col].astype(str)
            predicted_risk_str = df[predicted_risk_col].astype(str)
            
            # Check for common risk values
            common_risk_terms = ['high', 'medium', 'low', 'risk']
            
            def contains_risk_term(value):
                value_lower = str(value).lower()
                return any(term in value_lower for term in common_risk_terms)
            
            actual_has_risk = any(contains_risk_term(val) for val in actual_unique[:5])
            predicted_has_risk = any(contains_risk_term(val) for val in predicted_unique[:5])
            
            if st.button("✅ Process Data", type="primary"):
                # Process the data
                df_processed = df.copy()
                
                # Standardize risk column names
                df_processed = df_processed.rename(columns={
                    actual_risk_col: 'Actual_Risk',
                    predicted_risk_col: 'Predicted_Risk'
                })
                
                # Convert to string for safety
                df_processed['Actual_Risk'] = df_processed['Actual_Risk'].astype(str)
                df_processed['Predicted_Risk'] = df_processed['Predicted_Risk'].astype(str)
                
                # Standardize risk categories
                risk_mapping = {
                    # Handle various formats
                    'high risk': 'High Risk',
                    'high': 'High Risk',
                    'high_risk': 'High Risk',
                    'high-risk': 'High Risk',
                    'medium risk': 'Medium Risk',
                    'medium': 'Medium Risk',
                    'medium_risk': 'Medium Risk',
                    'medium-risk': 'Medium Risk',
                    'low risk': 'Low Risk',
                    'low': 'Low Risk',
                    'low_risk': 'Low Risk',
                    'low-risk': 'Low Risk',
                }
                
                # Apply mapping
                for old, new in risk_mapping.items():
                    mask = df_processed['Actual_Risk'].str.lower() == old
                    df_processed.loc[mask, 'Actual_Risk'] = new
                    
                    mask = df_processed['Predicted_Risk'].str.lower() == old
                    df_processed.loc[mask, 'Predicted_Risk'] = new
                
                # Store processed data
                st.session_state.df_processed = df_processed
                st.session_state.life_exp_col = life_exp_col
                
                st.success("✅ Data processed successfully!")
                
                # Show sample of processed data
                with st.expander("📊 View processed data (first 10 rows)"):
                    display_cols = [life_exp_col, 'Actual_Risk', 'Predicted_Risk']
                    if 'Country' in df_processed.columns:
                        display_cols = ['Country'] + display_cols
                    sample_df = df_processed[display_cols].head(10).copy()
                    sample_df['Match'] = sample_df['Actual_Risk'] == sample_df['Predicted_Risk']
                    st.dataframe(sample_df, use_container_width=True)
                
                # Show basic statistics
                st.header("📊 Basic Statistics")
                
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                
                with col_stat1:
                    total_samples = len(df_processed)
                    st.metric("Total Samples", total_samples)
                
                with col_stat2:
                    accuracy = (df_processed['Actual_Risk'] == df_processed['Predicted_Risk']).mean()
                    st.metric("Accuracy", f"{accuracy:.1%}")
                
                with col_stat3:
                    mismatches = (df_processed['Actual_Risk'] != df_processed['Predicted_Risk']).sum()
                    st.metric("Mismatches", mismatches)
                
                # 3. VISUALIZATIONS
                st.header("📈 Visualizations")
                
                categories = ['High Risk', 'Medium Risk', 'Low Risk']
                
                # Ensure all categories exist
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
                    for i, (act, pred) in enumerate(zip(actual_counts.values, predicted_counts.values)):
                        ax.text(i - width/2, act + 0.5, str(int(act)), ha='center', va='bottom', fontsize=10)
                        ax.text(i + width/2, pred + 0.5, str(int(pred)), ha='center', va='bottom', fontsize=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show comparison table
                    st.write("**Count Comparison:**")
                    comparison_df = pd.DataFrame({
                        'Risk Level': categories,
                        'Actual': actual_counts.values,
                        'Predicted': predicted_counts.values,
                        'Difference': predicted_counts.values - actual_counts.values
                    })
                    st.dataframe(comparison_df, use_container_width=True)
                
                with tab2:
                    # BAR GRAPH - Actual Risk Distribution
                    st.subheader("Actual Risk Distribution")
                    
                    col_vis1, col_vis2 = st.columns(2)
                    
                    with col_vis1:
                        # Bar Chart
                        st.write("**Bar Chart**")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        colors = ['#FF5252', '#FFA726', '#66BB6A']
                        bars = ax.bar(categories, actual_counts.values, 
                                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)
                        
                        ax.set_xlabel('Risk Level', fontsize=12)
                        ax.set_ylabel('Count', fontsize=12)
                        ax.set_title('Actual Risk Distribution', fontsize=14, fontweight='bold')
                        ax.grid(True, alpha=0.3, axis='y')
                        
                        # Add value labels
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
                        explode = (0.05, 0.05, 0.05)
                        
                        wedges, texts, autotexts = ax.pie(
                            actual_counts.values,
                            labels=categories,
                            autopct='%1.1f%%',
                            colors=colors,
                            explode=explode,
                            startangle=90
                        )
                        
                        ax.axis('equal')
                        ax.set_title('Risk Distribution Percentage', fontsize=14, fontweight='bold')
                        
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontweight('bold')
                        
                        st.pyplot(fig)
                    
                    # Distribution statistics
                    st.write("**Distribution Statistics:**")
                    total = actual_counts.sum()
                    percentages = (actual_counts / total * 100).round(1)
                    
                    col_dist1, col_dist2, col_dist3 = st.columns(3)
                    
                    with col_dist1:
                        st.markdown("**High Risk**")
                        st.markdown(f"<h3 style='color: #FF5252;'>{actual_counts['High Risk']}</h3>", 
                                   unsafe_allow_html=True)
                        st.write(f"{percentages['High Risk']}%")
                    
                    with col_dist2:
                        st.markdown("**Medium Risk**")
                        st.markdown(f"<h3 style='color: #FFA726;'>{actual_counts['Medium Risk']}</h3>", 
                                   unsafe_allow_html=True)
                        st.write(f"{percentages['Medium Risk']}%")
                    
                    with col_dist3:
                        st.markdown("**Low Risk**")
                        st.markdown(f"<h3 style='color: #66BB6A;'>{actual_counts['Low Risk']}</h3>", 
                                   unsafe_allow_html=True)
                        st.write(f"{percentages['Low Risk']}%")
                
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
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    x = np.arange(len(categories))
                    width = 0.25
                    
                    # Extract metrics
                    precision_vals = []
                    recall_vals = []
                    f1_vals = []
                    
                    for class_name in categories:
                        if class_name in report:
                            precision_vals.append(report[class_name]['precision'])
                            recall_vals.append(report[class_name]['recall'])
                            f1_vals.append(report[class_name]['f1-score'])
                        else:
                            precision_vals.append(0)
                            recall_vals.append(0)
                            f1_vals.append(0)
                    
                    bars1 = ax.bar(x - width, precision_vals, width, 
                                  label='Precision', color='#FF9800')
                    bars2 = ax.bar(x, recall_vals, width, 
                                  label='Recall', color='#2196F3')
                    bars3 = ax.bar(x + width, f1_vals, width, 
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
                    for bars, values in zip([bars1, bars2, bars3], [precision_vals, recall_vals, f1_vals]):
                        for bar, val in zip(bars, values):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        
        else:
            st.warning("⚠️ Please select all three columns to proceed.")
    
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.info("Make sure your CSV file contains: 1) Life expectancy values, 2) Actual risk labels, 3) Predicted risk labels")

else:
    # Show instructions when no file is uploaded
    st.info("👆 Upload your CSV file with predictions")
    
    st.write("**Prepare your data in your notebook:**")
    st.code("""
    # In your Jupyter notebook, after making predictions:
    
    # 1. Create a DataFrame with your results
    results_df = pd.DataFrame({
        'Life_Expectancy': X_test['Life_Expectancy'],  # Your test data
        'Actual_Risk': y_test,                         # Actual labels
        'Predicted_Risk': y_pred                       # Model predictions
    })
    
    # 2. Add any other columns you want
    results_df['Country'] = test_countries  # Optional
    results_df['Year'] = test_years         # Optional
    
    # 3. Save to CSV
    results_df.to_csv('model_predictions.csv', index=False)
    """)
    
    st.write("**Example of what your CSV should contain:**")
    example_data = pd.DataFrame({
        'Life_Expectancy': [58.5, 72.3, 81.7, 65.8, 78.9],
        'Actual_Risk': ['High Risk', 'Medium Risk', 'Low Risk', 'Medium Risk', 'Low Risk'],
        'Predicted_Risk': ['High Risk', 'Medium Risk', 'Low Risk', 'High Risk', 'Low Risk'],
        'Country': ['Country A', 'Country B', 'Country C', 'Country D', 'Country E']
    })
    st.dataframe(example_data, use_container_width=True)