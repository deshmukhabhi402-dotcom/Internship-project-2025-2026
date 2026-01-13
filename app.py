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
        
        # Get prediction accuracy from user
        accuracy = st.slider("Set prediction accuracy:", 0.5, 1.0, 0.85, 0.05)
        
        if st.button("Generate Predictions", type="primary"):
            # Generate sample predictions
            np.random.seed(42)
            n = len(df)
            
            # Start with actual values
            predictions = df['Actual_Risk'].copy()
            
            # Make some wrong predictions based on accuracy
            wrong_count = int((1 - accuracy) * n)
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
            
            # Calculate actual accuracy
            actual_accuracy = (df['Actual_Risk'] == df['Predicted_Risk']).mean()
            
            # Display metrics
            st.success(f"✓ Predictions generated with {actual_accuracy:.1%} accuracy")
            
            # Show sample predictions
            with st.expander("View predictions (first 10 rows)"):
                display_cols = [life_exp_col, 'Actual_Risk', 'Predicted_Risk']
                if 'Country' in df.columns:
                    display_cols = ['Country'] + display_cols
                st.dataframe(df[display_cols].head(10))
            
            # 5. VISUALIZATIONS
            st.header("5. Visualizations")
            
            # Tab layout
            tab1, tab2, tab3 = st.tabs(["Predicted vs Actual", "Risk Distribution", "Classification Report"])
            
            # Count values
            categories = ['High Risk', 'Medium Risk', 'Low Risk']
            actual_counts = df['Actual_Risk'].value_counts().reindex(categories, fill_value=0)
            predicted_counts = df['Predicted_Risk'].value_counts().reindex(categories, fill_value=0)
            
            # TAB 1: Predicted vs Actual
            with tab1:
                st.subheader("Predicted vs Actual Risk Levels")
                
                col_graph1, col_graph2 = st.columns(2)
                
                with col_graph1:
                    # BAR GRAPH
                    st.write("**Bar Graph**")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    x = np.arange(len(categories))
                    width = 0.35
                    
                    ax.bar(x - width/2, actual_counts.values, width, label='Actual', color='blue', alpha=0.7)
                    ax.bar(x + width/2, predicted_counts.values, width, label='Predicted', color='orange', alpha=0.7)
                    
                    ax.set_xlabel('Risk Level')
                    ax.set_ylabel('Count')
                    ax.set_title('Predicted vs Actual')
                    ax.set_xticks(x)
                    ax.set_xticklabels(['High', 'Medium', 'Low'])
                    ax.legend()
                    
                    # Add numbers on bars
                    for i, (act, pred) in enumerate(zip(actual_counts.values, predicted_counts.values)):
                        ax.text(i - width/2, act + 0.5, str(int(act)), ha='center')
                        ax.text(i + width/2, pred + 0.5, str(int(pred)), ha='center')
                    
                    st.pyplot(fig)
                
                with col_graph2:
                    # LINE GRAPH
                    st.write("**Line Graph**")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    # Take first 50 samples for clarity
                    sample_size = min(50, len(df))
                    sample_df = df.head(sample_size).copy()
                    sample_df = sample_df.sort_values(life_exp_col).reset_index(drop=True)
                    
                    # Convert risk to numeric for line plot
                    risk_to_num = {'High Risk': 0, 'Medium Risk': 1, 'Low Risk': 2}
                    actual_numeric = sample_df['Actual_Risk'].map(risk_to_num)
                    predicted_numeric = sample_df['Predicted_Risk'].map(risk_to_num)
                    
                    ax.plot(actual_numeric.index, actual_numeric.values, 
                           label='Actual', marker='o', linewidth=2, color='blue')
                    ax.plot(predicted_numeric.index, predicted_numeric.values, 
                           label='Predicted', marker='s', linewidth=2, color='orange', alpha=0.7)
                    
                    # Mark mismatches
                    mismatches = sample_df[sample_df['Actual_Risk'] != sample_df['Predicted_Risk']].index
                    if len(mismatches) > 0:
                        ax.scatter(mismatches, predicted_numeric[mismatches], 
                                  color='red', s=100, zorder=5, label='Mismatch')
                    
                    ax.set_xlabel('Sample Index (sorted)')
                    ax.set_ylabel('Risk Level (0=High, 1=Medium, 2=Low)')
                    ax.set_title('Risk Prediction Across Samples')
                    ax.set_yticks([0, 1, 2])
                    ax.set_yticklabels(['High', 'Medium', 'Low'])
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
            
            # TAB 2: Risk Distribution
            with tab2:
                st.subheader("Risk Distribution")
                
                col_dist1, col_dist2 = st.columns(2)
                
                with col_dist1:
                    # Actual Distribution
                    st.write("**Actual Risk Distribution**")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    
                    colors = ['#FF6B6B', '#FFD93D', '#6BCF7F']  # Red, Yellow, Green
                    wedges, texts, autotexts = ax.pie(
                        actual_counts.values,
                        labels=['High', 'Medium', 'Low'],
                        autopct='%1.1f%%',
                        colors=colors,
                        startangle=90
                    )
                    ax.axis('equal')
                    ax.set_title('Actual Risk Distribution')
                    st.pyplot(fig)
                
                with col_dist2:
                    # Predicted Distribution
                    st.write("**Predicted Risk Distribution**")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    
                    wedges, texts, autotexts = ax.pie(
                        predicted_counts.values,
                        labels=['High', 'Medium', 'Low'],
                        autopct='%1.1f%%',
                        colors=colors,
                        startangle=90
                    )
                    ax.axis('equal')
                    ax.set_title('Predicted Risk Distribution')
                    st.pyplot(fig)
                
                # Bar chart comparison
                st.write("**Distribution Comparison**")
                fig, ax = plt.subplots(figsize=(10, 5))
                
                x = np.arange(len(categories))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, actual_counts.values, width, 
                              label='Actual', color='blue', alpha=0.7)
                bars2 = ax.bar(x + width/2, predicted_counts.values, width, 
                              label='Predicted', color='orange', alpha=0.7)
                
                ax.set_xlabel('Risk Level')
                ax.set_ylabel('Count')
                ax.set_title('Risk Distribution: Actual vs Predicted')
                ax.set_xticks(x)
                ax.set_xticklabels(['High', 'Medium', 'Low'])
                ax.legend()
                
                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                               f'{int(height)}', ha='center', va='bottom')
                
                st.pyplot(fig)
            
            # TAB 3: Classification Report
            with tab3:
                st.subheader("Classification Report")
                
                # Calculate metrics
                from sklearn.metrics import confusion_matrix, classification_report
                
                # Create confusion matrix
                cm = confusion_matrix(df['Actual_Risk'], df['Predicted_Risk'], 
                                      labels=categories)
                
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
                
                # Calculate and display classification report
                st.write("**Classification Metrics**")
                report_dict = classification_report(df['Actual_Risk'], df['Predicted_Risk'], 
                                                   output_dict=True, zero_division=0)
                
                # Convert to DataFrame
                report_df = pd.DataFrame(report_dict).transpose()
                
                # Display as table
                st.dataframe(report_df.style.format({
                    'precision': '{:.3f}',
                    'recall': '{:.3f}',
                    'f1-score': '{:.3f}',
                    'support': '{:.0f}'
                }), use_container_width=True)
                
                # Show accuracy by class
                st.write("**Accuracy by Risk Level**")
                class_accuracies = []
                for class_name in categories:
                    class_mask = df['Actual_Risk'] == class_name
                    if class_mask.sum() > 0:
                        class_acc = (df.loc[class_mask, 'Actual_Risk'] == 
                                   df.loc[class_mask, 'Predicted_Risk']).mean()
                        class_accuracies.append(class_acc)
                    else:
                        class_accuracies.append(0)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                bars = ax.bar(['High', 'Medium', 'Low'], class_accuracies, 
                             color=['#FF6B6B', '#FFD93D', '#6BCF7F'])
                ax.set_ylim([0, 1])
                ax.set_ylabel('Accuracy')
                ax.set_title('Prediction Accuracy by Risk Level')
                
                # Add value labels
                for bar, acc in zip(bars, class_accuracies):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                           f'{acc:.1%}', ha='center')
                
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