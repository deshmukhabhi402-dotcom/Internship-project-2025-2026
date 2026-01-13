# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Page setup
st.set_page_config(page_title="Life Expectancy Risk Analysis", layout="wide")
st.title("📊 Life Expectancy Risk Analysis Dashboard")

st.markdown("""
### Important: How to get matching results
1. **In your notebook**, save your **actual predictions** to CSV
2. **Upload that CSV here** to see matching visualizations
3. This app **does NOT generate predictions** - it shows YOUR results
""")

# 1. UPLOAD YOUR PREDICTIONS CSV
st.header("📁 Step 1: Upload Your Predictions CSV")
uploaded_file = st.file_uploader("Upload the CSV file with your model predictions", type=['csv'])

if uploaded_file is not None:
    try:
        # Load your predictions
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Show what columns are available
        st.write("**Columns in your file:**")
        st.write(", ".join(df.columns.tolist()))
        
        # Show preview
        with st.expander("👀 Preview your data"):
            st.dataframe(df.head(), use_container_width=True)
        
        # 2. SELECT THE RIGHT COLUMNS
        st.header("🔍 Step 2: Select Your Columns")
        
        st.info("""
        Your CSV should contain:
        1. **Life Expectancy values** (or your target variable)
        2. **Actual Risk labels** (from your test data)
        3. **Predicted Risk labels** (from your model)
        """)
        
        # Let user select columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Find life expectancy column
            life_cols = [col for col in df.columns if any(word in col.lower() 
                       for word in ['life', 'expectancy', 'target'])]
            if life_cols:
                life_exp_col = st.selectbox("Life Expectancy column:", 
                                           df.columns, 
                                           index=df.columns.get_loc(life_cols[0]))
            else:
                life_exp_col = st.selectbox("Life Expectancy column:", df.columns)
        
        with col2:
            # Find actual risk column
            actual_cols = [col for col in df.columns if any(word in col.lower() 
                          for word in ['actual', 'true', 'test', 'y_test'])]
            if actual_cols:
                actual_risk_col = st.selectbox("Actual Risk column:", 
                                              df.columns, 
                                              index=df.columns.get_loc(actual_cols[0]))
            else:
                actual_risk_col = st.selectbox("Actual Risk column:", df.columns)
        
        with col3:
            # Find predicted risk column
            pred_cols = [col for col in df.columns if any(word in col.lower() 
                         for word in ['predicted', 'pred', 'y_pred', 'predict'])]
            if pred_cols:
                predicted_risk_col = st.selectbox("Predicted Risk column:", 
                                                 df.columns, 
                                                 index=df.columns.get_loc(pred_cols[0]))
            else:
                predicted_risk_col = st.selectbox("Predicted Risk column:", df.columns)
        
        # 3. PROCESS THE DATA
        if st.button("📊 Show Visualizations", type="primary"):
            # Create a clean dataframe
            results_df = pd.DataFrame({
                'Life_Expectancy': df[life_exp_col],
                'Actual_Risk': df[actual_risk_col].astype(str).str.strip(),
                'Predicted_Risk': df[predicted_risk_col].astype(str).str.strip()
            })
            
            # Standardize risk labels (case-insensitive)
            risk_mapping = {
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
            
            # Apply standardization
            for old, new in risk_mapping.items():
                results_df.loc[results_df['Actual_Risk'].str.lower() == old, 'Actual_Risk'] = new
                results_df.loc[results_df['Predicted_Risk'].str.lower() == old, 'Predicted_Risk'] = new
            
            # Show processed data
            st.success("✅ Data processed successfully!")
            
            # Calculate basic metrics
            accuracy = (results_df['Actual_Risk'] == results_df['Predicted_Risk']).mean()
            total = len(results_df)
            mismatches = (results_df['Actual_Risk'] != results_df['Predicted_Risk']).sum()
            
            # Display metrics
            col_met1, col_met2, col_met3 = st.columns(3)
            with col_met1:
                st.metric("Accuracy", f"{accuracy:.1%}")
            with col_met2:
                st.metric("Total Samples", total)
            with col_met3:
                st.metric("Mismatches", mismatches)
            
            # 4. VISUALIZATIONS
            st.header("📈 Your Visualizations")
            
            # Define categories
            categories = ['High Risk', 'Medium Risk', 'Low Risk']
            
            # Get counts
            actual_counts = results_df['Actual_Risk'].value_counts().reindex(categories, fill_value=0)
            predicted_counts = results_df['Predicted_Risk'].value_counts().reindex(categories, fill_value=0)
            
            # Create tabs
            tab1, tab2, tab3 = st.tabs(["Predicted vs Actual", "Risk Distribution", "Classification Report"])
            
            with tab1:
                # BAR GRAPH - Predicted vs Actual
                st.subheader("📊 Predicted vs Actual (Bar Graph)")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(categories))
                width = 0.35
                
                ax.bar(x - width/2, actual_counts.values, width, 
                      label='Actual', color='blue', alpha=0.7)
                ax.bar(x + width/2, predicted_counts.values, width, 
                      label='Predicted', color='orange', alpha=0.7)
                
                ax.set_xlabel('Risk Level')
                ax.set_ylabel('Count')
                ax.set_title('Your Model: Predicted vs Actual Risk Levels')
                ax.set_xticks(x)
                ax.set_xticklabels(['High', 'Medium', 'Low'])
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add numbers on bars
                for i, (act, pred) in enumerate(zip(actual_counts.values, predicted_counts.values)):
                    ax.text(i - width/2, act + 0.5, str(int(act)), ha='center', fontsize=10)
                    ax.text(i + width/2, pred + 0.5, str(int(pred)), ha='center', fontsize=10)
                
                st.pyplot(fig)
                
                # Show data table
                with st.expander("📋 View comparison data"):
                    comp_df = pd.DataFrame({
                        'Risk Level': categories,
                        'Actual Count': actual_counts.values,
                        'Predicted Count': predicted_counts.values
                    })
                    st.dataframe(comp_df, use_container_width=True)
            
            with tab2:
                # BAR GRAPH - Risk Distribution
                st.subheader("📊 Risk Distribution (Bar Graph)")
                
                # Two visualizations side by side
                col_risk1, col_risk2 = st.columns(2)
                
                with col_risk1:
                    # Bar chart
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = ['#FF6B6B', '#FFD93D', '#6BCF7F']
                    
                    bars = ax.bar(categories, actual_counts.values, 
                                 color=colors, alpha=0.8, edgecolor='black')
                    
                    ax.set_xlabel('Risk Level')
                    ax.set_ylabel('Count')
                    ax.set_title('Actual Risk Distribution')
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                               f'{int(height)}', ha='center', fontweight='bold')
                    
                    st.pyplot(fig)
                
                with col_risk2:
                    # Pie chart
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    colors = ['#FF6B6B', '#FFD93D', '#6BCF7F']
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
                    ax.set_title('Risk Distribution Percentage')
                    
                    st.pyplot(fig)
                
                # Show statistics
                st.write("**Distribution Statistics:**")
                total_count = actual_counts.sum()
                
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    perc = (actual_counts['High Risk'] / total_count * 100)
                    st.metric("High Risk", f"{actual_counts['High Risk']}", 
                             f"{perc:.1f}%")
                with col_stat2:
                    perc = (actual_counts['Medium Risk'] / total_count * 100)
                    st.metric("Medium Risk", f"{actual_counts['Medium Risk']}", 
                             f"{perc:.1f}%")
                with col_stat3:
                    perc = (actual_counts['Low Risk'] / total_count * 100)
                    st.metric("Low Risk", f"{actual_counts['Low Risk']}", 
                             f"{perc:.1f}%")
            
            with tab3:
                # CLASSIFICATION REPORT
                st.subheader("📋 Classification Report")
                
                # Calculate classification report
                report = classification_report(results_df['Actual_Risk'], 
                                              results_df['Predicted_Risk'],
                                              output_dict=True,
                                              zero_division=0)
                
                # Create two columns
                col_cm, col_table = st.columns([1, 2])
                
                with col_cm:
                    # Confusion Matrix
                    cm = confusion_matrix(results_df['Actual_Risk'], 
                                         results_df['Predicted_Risk'], 
                                         labels=categories)
                    
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                               xticklabels=['High', 'Medium', 'Low'],
                               yticklabels=['High', 'Medium', 'Low'],
                               ax=ax)
                    
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix')
                    
                    st.pyplot(fig)
                
                with col_table:
                    # Classification report table
                    report_df = pd.DataFrame(report).transpose()
                    
                    # Style the table
                    styled_df = report_df.style.format({
                        'precision': '{:.3f}',
                        'recall': '{:.3f}',
                        'f1-score': '{:.3f}',
                        'support': '{:.0f}'
                    }).background_gradient(subset=['precision', 'recall', 'f1-score'], 
                                          cmap='YlOrRd', low=0.3, high=0.7)
                    
                    st.dataframe(styled_df, use_container_width=True)
                
                # Metrics visualization
                st.write("**Metrics Visualization:**")
                
                # Extract metrics for each class
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
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 5))
                x = np.arange(len(categories))
                width = 0.25
                
                bars1 = ax.bar(x - width, precision_vals, width, 
                              label='Precision', color='#FF9800')
                bars2 = ax.bar(x, recall_vals, width, 
                              label='Recall', color='#2196F3')
                bars3 = ax.bar(x + width, f1_vals, width, 
                              label='F1-Score', color='#4CAF50')
                
                ax.set_xlabel('Risk Level')
                ax.set_ylabel('Score')
                ax.set_title('Metrics by Risk Level')
                ax.set_xticks(x)
                ax.set_xticklabels(['High', 'Medium', 'Low'])
                ax.set_ylim([0, 1])
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for i, (prec, rec, f1) in enumerate(zip(precision_vals, recall_vals, f1_vals)):
                    ax.text(i - width, prec + 0.02, f'{prec:.2f}', ha='center', fontsize=9)
                    ax.text(i, rec + 0.02, f'{rec:.2f}', ha='center', fontsize=9)
                    ax.text(i + width, f1 + 0.02, f'{f1:.2f}', ha='center', fontsize=9)
                
                st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Make sure you've selected the correct columns.")

else:
    # Show instructions
    st.info("👆 Upload your predictions CSV file to get started")
    
    st.write("""
    ### How to prepare your data in the notebook:
    
    **Step 1: After training your model, save predictions to CSV:**
    ```python
    # Assuming you have these from your notebook:
    # y_test = actual test labels
    # y_pred = model predictions
    # X_test = test features (with life expectancy)
    
    # Create a results dataframe
    results = pd.DataFrame({
        'Life_Expectancy': X_test['Life_Expectancy'].values,  # Your feature
        'Actual_Risk': y_test,                                # True labels
        'Predicted_Risk': y_pred                              # Model predictions
    })
    
    # Save to CSV
    results.to_csv('my_predictions.csv', index=False)
    ```
    
    **Step 2: Upload that CSV file here**
    
    **Step 3: Select the correct columns when prompted**
    
    **Step 4: See your exact results visualized**
    """)
    
    # Show example
    st.write("**Example of what your CSV should look like:**")
    example = pd.DataFrame({
        'Life_Expectancy': [58.3, 72.1, 81.5, 65.8, 79.2],
        'Actual_Risk': ['High Risk', 'Medium Risk', 'Low Risk', 'Medium Risk', 'Low Risk'],
        'Predicted_Risk': ['High Risk', 'Medium Risk', 'Low Risk', 'High Risk', 'Low Risk']
    })
    st.dataframe(example, use_container_width=True)