# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page setup
st.set_page_config(page_title="Life Expectancy Analysis", layout="wide")
st.title("📊 Life Expectancy Prediction & Risk Analysis")

st.markdown("""
### This app analyzes both Regression and Classification results from your notebook
**Your notebook shows:**
1. **Regression**: Predicting Life Expectancy values
2. **Classification**: Categorizing into Health Risk Levels (High, Medium, Low)
""")

# 1. UPLOAD YOUR DATASET
st.header("📁 Upload Your Dataset")
uploaded_file = st.file_uploader("Upload your dataset CSV (like UnifiedDataset.csv)", type=['csv'])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Show preview
        with st.expander("👀 Preview your dataset"):
            st.dataframe(df.head(), use_container_width=True)
            st.write(f"**Dataset Info:** {df.shape[0]} samples, {df.shape[1]} features")
        
        # 2. DATA PREPROCESSING (Matching your notebook)
        st.header("🔧 Data Preprocessing")
        
        with st.expander("Data Processing Steps (from your notebook)"):
            st.markdown("""
            **From your notebook:**
            1. Fill missing numerical values with median
            2. Fill missing categorical values with mode
            3. Encode categorical columns
            4. Create Health Risk Levels:
               - High Risk: Life Expectancy < 60
               - Medium Risk: Life Expectancy 60-70
               - Low Risk: Life Expectancy > 70
            """)
        
        # Check if Life Expectancy column exists
        if 'Life Expectancy' not in df.columns:
            # Try to find similar column names
            life_cols = [col for col in df.columns if 'life' in col.lower() or 'expect' in col.lower()]
            if life_cols:
                df = df.rename(columns={life_cols[0]: 'Life Expectancy'})
                st.info(f"Renamed '{life_cols[0]}' to 'Life Expectancy'")
            else:
                st.error("❌ Could not find Life Expectancy column in your dataset")
                st.stop()
        
        # Create Health Risk Level (same as notebook)
        def risk_level(le):
            if pd.isna(le):
                return np.nan
            if le < 60:
                return "High"
            elif le < 70:
                return "Medium"
            else:
                return "Low"
        
        df["Health_Risk_Level"] = df["Life Expectancy"].apply(risk_level)
        
        # Show risk distribution
        risk_counts = df["Health_Risk_Level"].value_counts()
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("Total Samples", df.shape[0])
        with col_stat2:
            if 'High' in risk_counts:
                st.metric("High Risk", risk_counts['High'])
        with col_stat3:
            if 'Medium' in risk_counts:
                st.metric("Medium Risk", risk_counts['Medium'])
        with col_stat4:
            if 'Low' in risk_counts:
                st.metric("Low Risk", risk_counts['Low'])
        
        # 3. VISUALIZE DISTRIBUTIONS
        st.header("📊 Dataset Analysis")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "Life Expectancy Distribution", 
            "Risk Level Analysis", 
            "Feature Correlation",
            "Model Simulation"
        ])
        
        with tab1:
            # Life Expectancy Distribution
            st.subheader("Life Expectancy Distribution")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Histogram
            ax1.hist(df['Life Expectancy'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='skyblue')
            ax1.axvline(x=60, color='red', linestyle='--', alpha=0.5, label='High/Medium Threshold')
            ax1.axvline(x=70, color='orange', linestyle='--', alpha=0.5, label='Medium/Low Threshold')
            ax1.set_xlabel('Life Expectancy (years)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of Life Expectancy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot(df['Life Expectancy'].dropna())
            ax2.set_ylabel('Life Expectancy (years)')
            ax2.set_title('Box Plot of Life Expectancy')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f"""
            Statistics:
            Mean: {df['Life Expectancy'].mean():.2f}
            Median: {df['Life Expectancy'].median():.2f}
            Std Dev: {df['Life Expectancy'].std():.2f}
            Min: {df['Life Expectancy'].min():.2f}
            Max: {df['Life Expectancy'].max():.2f}
            """
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                    fontsize=9, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Statistics table
            st.write("**Life Expectancy Statistics by Risk Level:**")
            if 'Health_Risk_Level' in df.columns:
                stats_by_risk = df.groupby('Health_Risk_Level')['Life Expectancy'].agg([
                    'count', 'mean', 'std', 'min', 'max', 'median'
                ]).round(2)
                st.dataframe(stats_by_risk, use_container_width=True)
        
        with tab2:
            # Risk Level Analysis
            st.subheader("Health Risk Level Analysis")
            
            col_risk1, col_risk2 = st.columns(2)
            
            with col_risk1:
                # Pie chart
                fig, ax = plt.subplots(figsize=(8, 8))
                
                colors = {'High': '#FF6B6B', 'Medium': '#FFD93D', 'Low': '#6BCF7F'}
                risk_colors = [colors.get(risk, 'gray') for risk in risk_counts.index]
                
                wedges, texts, autotexts = ax.pie(risk_counts.values, 
                                                 labels=risk_counts.index, 
                                                 colors=risk_colors,
                                                 autopct='%1.1f%%', 
                                                 startangle=90,
                                                 explode=[0.05, 0.05, 0.05] if len(risk_counts) == 3 else [0.05]*len(risk_counts))
                
                ax.set_title('Distribution of Health Risk Levels', fontsize=14, fontweight='bold')
                
                # Make percentages bold
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                st.pyplot(fig)
            
            with col_risk2:
                # Bar chart
                fig, ax = plt.subplots(figsize=(8, 8))
                
                bars = ax.bar(risk_counts.index, risk_counts.values, 
                             color=[colors.get(risk, 'gray') for risk in risk_counts.index],
                             alpha=0.8, edgecolor='black')
                
                ax.set_xlabel('Health Risk Level', fontsize=12)
                ax.set_ylabel('Number of Samples', fontsize=12)
                ax.set_title('Health Risk Level Counts', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                           f'{int(height)}', ha='center', fontweight='bold', fontsize=11)
                
                st.pyplot(fig)
            
            # Show sample data for each risk level
            st.write("**Sample Data by Risk Level:**")
            sample_size = st.slider("Samples per risk level:", 1, 10, 3)
            
            for risk_level in ['High', 'Medium', 'Low']:
                if risk_level in df['Health_Risk_Level'].values:
                    with st.expander(f"{risk_level} Risk Samples"):
                        samples = df[df['Health_Risk_Level'] == risk_level].head(sample_size)
                        # Select only relevant columns
                        display_cols = ['Life Expectancy', 'Health_Risk_Level']
                        # Add some additional columns if they exist
                        additional_cols = ['Country', 'Year', 'Gender', 'Infant Mortality Rate']
                        for col in additional_cols:
                            if col in df.columns:
                                display_cols.append(col)
                        
                        st.dataframe(samples[display_cols], use_container_width=True)
        
        with tab3:
            # Feature Correlation
            st.subheader("Feature Correlation with Life Expectancy")
            
            # Select numerical columns
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(num_cols) > 1 and 'Life Expectancy' in num_cols:
                # Calculate correlation with Life Expectancy
                correlations = df[num_cols].corr()['Life Expectancy'].sort_values(ascending=False)
                
                # Remove Life Expectancy itself
                correlations = correlations[correlations.index != 'Life Expectancy']
                
                # Take top and bottom correlations
                top_n = st.slider("Number of features to show:", 5, 20, 10)
                top_features = pd.concat([correlations.head(top_n//2), correlations.tail(top_n//2)])
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                colors = ['green' if x > 0 else 'red' for x in top_features.values]
                bars = ax.barh(range(len(top_features)), top_features.values, color=colors, alpha=0.7)
                
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features.index)
                ax.set_xlabel('Correlation with Life Expectancy')
                ax.set_title(f'Top {top_n} Most Correlated Features', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                
                # Add correlation values on bars
                for i, (bar, val) in enumerate(zip(bars, top_features.values)):
                    ax.text(val + (0.01 if val >= 0 else -0.03), bar.get_y() + bar.get_height()/2,
                           f'{val:.3f}', va='center', fontsize=9,
                           color='black' if abs(val) > 0.1 else 'gray')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show correlation table
                st.write("**Correlation Details:**")
                corr_table = pd.DataFrame({
                    'Feature': top_features.index,
                    'Correlation': top_features.values,
                    'Strength': ['Strong' if abs(x) > 0.5 else 'Moderate' if abs(x) > 0.3 else 'Weak' for x in top_features.values],
                    'Direction': ['Positive' if x > 0 else 'Negative' for x in top_features.values]
                })
                st.dataframe(corr_table.style.format({'Correlation': '{:.3f}'}), 
                           use_container_width=True)
            else:
                st.warning("Not enough numerical columns for correlation analysis")
        
        with tab4:
            # Model Simulation
            st.subheader("Simulate Model Predictions")
            
            st.markdown("""
            **Simulating your notebook's Random Forest predictions:**
            This section simulates what your model might predict based on the dataset.
            """)
            
            # Simulate predictions (for demo purposes)
            np.random.seed(42)
            
            # Create simulated predictions with some error
            actual_values = df['Life Expectancy'].dropna().values
            
            # Add random error to simulate predictions
            error_std = st.slider("Simulated prediction error (std dev):", 0.1, 5.0, 1.5)
            simulated_predictions = actual_values + np.random.normal(0, error_std, len(actual_values))
            
            # Clip predictions to realistic range
            simulated_predictions = np.clip(simulated_predictions, 40, 90)
            
            # Calculate metrics
            mse = mean_squared_error(actual_values, simulated_predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual_values, simulated_predictions)
            r2 = r2_score(actual_values, simulated_predictions)
            
            # Create simulated risk predictions
            actual_risk = [risk_level(x) for x in actual_values]
            predicted_risk = [risk_level(x) for x in simulated_predictions]
            
            # Calculate risk accuracy
            risk_accuracy = sum(1 for a, p in zip(actual_risk, predicted_risk) if a == p) / len(actual_risk)
            
            # Display metrics
            col_sim1, col_sim2, col_sim3, col_sim4 = st.columns(4)
            with col_sim1:
                st.metric("R² Score", f"{r2:.3f}")
            with col_sim2:
                st.metric("RMSE", f"{rmse:.2f}")
            with col_sim3:
                st.metric("MAE", f"{mae:.2f}")
            with col_sim4:
                st.metric("Risk Accuracy", f"{risk_accuracy:.1%}")
            
            # Plot simulated results
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Scatter plot
            ax1.scatter(actual_values, simulated_predictions, alpha=0.5, s=20)
            ax1.plot([40, 90], [40, 90], 'r--', label='Perfect Prediction')
            ax1.set_xlabel('Actual Life Expectancy')
            ax1.set_ylabel('Simulated Prediction')
            ax1.set_title('Simulated Model Predictions')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal', adjustable='box')
            
            # Confusion matrix for risk levels
            risk_levels = ['High', 'Medium', 'Low']
            cm = confusion_matrix(actual_risk, predicted_risk, labels=risk_levels)
            
            im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax2.figure.colorbar(im, ax=ax2)
            
            # Set labels
            ax2.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=risk_levels, yticklabels=risk_levels,
                   title='Risk Level Confusion Matrix',
                   ylabel='Actual Risk',
                   xlabel='Predicted Risk')
            
            # Rotate tick labels
            plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax2.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info("**Note:** This is a simulation. Upload your actual model predictions for accurate analysis.")
        
        # 4. DOWNLOAD PROCESSED DATA
        st.header("💾 Download Processed Data")
        
        processed_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Processed Dataset with Risk Levels",
            data=processed_csv,
            file_name="processed_dataset_with_risk_levels.csv",
            mime="text/csv",
            help="Includes original data plus calculated Health Risk Levels"
        )
        
        # Show column information
        with st.expander("📋 Dataset Column Information"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.notnull().sum(),
                'Null Count': df.isnull().sum(),
                'Unique Values': [df[col].nunique() if df[col].dtype == 'object' else '-' for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True, height=400)
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.code(f"Error details: {e}", language='python')

else:
    # Show instructions
    st.info("👆 Upload your dataset CSV file (like UnifiedDataset.csv from your notebook)")
    
    # Show example of what the app creates
    st.write("**Example of created Risk Levels:**")
    example_df = pd.DataFrame({
        'Life_Expectancy': [55, 62, 78, 45, 85, 68],
        'Health_Risk_Level': ['High', 'Medium', 'Low', 'High', 'Low', 'Medium']
    })
    st.dataframe(example_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>This app analyzes Life Expectancy data and creates Health Risk Level classifications.</p>
    <p>Based on your notebook: Random Forest regression for Life Expectancy prediction.</p>
</div>
""", unsafe_allow_html=True)