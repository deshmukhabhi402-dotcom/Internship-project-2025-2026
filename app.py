# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Life Expectancy Risk Analysis",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #764ba2;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    .actual { color: #2196F3; font-weight: bold; }
    .predicted { color: #FF5722; font-weight: bold; }
    .risk-high { color: #FF5252; }
    .risk-medium { color: #FFA726; }
    .risk-low { color: #4CAF50; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📈 Life Expectancy Risk Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Compare Predicted vs Actual Risk Levels")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'risk_predictions' not in st.session_state:
    st.session_state.risk_predictions = None

# Sidebar for navigation
with st.sidebar:
    st.title("📋 Navigation")
    page = st.radio(
        "Go to:",
        ["📥 Load Dataset", "📊 Risk Analysis Dashboard"]
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This dashboard shows:
    1. Predicted vs Actual Risk Levels
    2. Risk Distribution
    3. Classification Report
    """)

# Page 1: Load Dataset
if page == "📥 Load Dataset":
    st.markdown("## 📥 Load Your Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file", 
            type=['csv'],
            help="Upload your dataset with 'Life_Expectancy' column"
        )
        
        if uploaded_file is not None:
            try:
                # Load the dataset
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.data_loaded = True
                
                st.success(f"✅ Dataset loaded successfully!")
                st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Show first 5 rows
                with st.expander("👁️ Preview first 5 rows"):
                    st.dataframe(df.head(), use_container_width=True)
                
                # Check for required columns
                if 'Life_Expectancy' in df.columns:
                    st.success("✅ 'Life_Expectancy' column found!")
                    
                    # Generate sample predictions (replace with your actual model)
                    np.random.seed(42)
                    
                    # Create actual risk levels based on life expectancy
                    df['Actual_Risk'] = pd.cut(
                        df['Life_Expectancy'],
                        bins=[0, 60, 75, 100],
                        labels=['High Risk', 'Medium Risk', 'Low Risk']
                    )
                    
                    # Generate sample predictions (85% accuracy)
                    n_samples = len(df)
                    correct_predictions = int(0.85 * n_samples)
                    wrong_predictions = n_samples - correct_predictions
                    
                    predicted = df['Actual_Risk'].copy()
                    
                    # Introduce some errors
                    error_indices = np.random.choice(n_samples, wrong_predictions, replace=False)
                    for idx in error_indices:
                        actual = df.loc[idx, 'Actual_Risk']
                        # Predict a different risk level
                        options = ['High Risk', 'Medium Risk', 'Low Risk']
                        options.remove(actual)
                        predicted[idx] = np.random.choice(options)
                    
                    df['Predicted_Risk'] = predicted
                    
                    # Calculate accuracy and classification metrics
                    accuracy = (df['Actual_Risk'] == df['Predicted_Risk']).mean()
                    
                    # Store in session state
                    st.session_state.risk_predictions = {
                        'df': df,
                        'accuracy': accuracy,
                        'actual_counts': df['Actual_Risk'].value_counts(),
                        'predicted_counts': df['Predicted_Risk'].value_counts()
                    }
                    st.session_state.predictions_made = True
                    
                    st.success("✅ Sample predictions generated!")
                    st.metric("Sample Model Accuracy", f"{accuracy:.1%}")
                    
                else:
                    st.warning("⚠️ 'Life_Expectancy' column not found. Please upload a dataset with this column.")
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    with col2:
        st.markdown("### 📋 Data Requirements")
        st.info("""
        Your dataset should include:
        - **Life_Expectancy** column
        - Health indicators
        - Socioeconomic factors
        - Any other relevant features
        """)
        
        # Sample data option
        with st.expander("🎲 Generate Sample Data"):
            if st.button("Generate Sample Dataset"):
                np.random.seed(42)
                
                # Create sample data
                sample_data = pd.DataFrame({
                    'Country': ['Country_' + str(i) for i in range(100)],
                    'Year': np.random.randint(2000, 2020, 100),
                    'Life_Expectancy': np.random.uniform(45, 85, 100),
                    'Adult_Mortality': np.random.uniform(50, 300, 100),
                    'BMI': np.random.uniform(15, 40, 100),
                    'Alcohol': np.random.uniform(0.1, 15, 100),
                    'GDP': np.random.uniform(100, 50000, 100),
                    'Schooling': np.random.uniform(5, 20, 100),
                    'HIV_AIDS': np.random.uniform(0, 5, 100)
                })
                
                # Create risk levels
                sample_data['Actual_Risk'] = pd.cut(
                    sample_data['Life_Expectancy'],
                    bins=[0, 60, 75, 100],
                    labels=['High Risk', 'Medium Risk', 'Low Risk']
                )
                
                # Generate predictions with some errors
                predicted = sample_data['Actual_Risk'].copy()
                error_indices = np.random.choice(100, 15, replace=False)
                for idx in error_indices:
                    actual = sample_data.loc[idx, 'Actual_Risk']
                    options = ['High Risk', 'Medium Risk', 'Low Risk']
                    options.remove(actual)
                    predicted[idx] = np.random.choice(options)
                
                sample_data['Predicted_Risk'] = predicted
                
                st.session_state.df = sample_data
                st.session_state.data_loaded = True
                
                # Calculate metrics
                accuracy = (sample_data['Actual_Risk'] == sample_data['Predicted_Risk']).mean()
                
                st.session_state.risk_predictions = {
                    'df': sample_data,
                    'accuracy': accuracy,
                    'actual_counts': sample_data['Actual_Risk'].value_counts(),
                    'predicted_counts': sample_data['Predicted_Risk'].value_counts()
                }
                st.session_state.predictions_made = True
                
                st.success("✅ Sample dataset created!")
                st.metric("Sample Size", "100 records")
                st.metric("Sample Accuracy", f"{accuracy:.1%}")

# Page 2: Risk Analysis Dashboard
elif page == "📊 Risk Analysis Dashboard":
    if not st.session_state.predictions_made:
        st.warning("⚠️ Please load a dataset first and generate predictions.")
        if st.button("Go to Load Dataset"):
            st.switch_page("📥 Load Dataset")
    else:
        predictions = st.session_state.risk_predictions
        df = predictions['df']
        
        # Display key metrics at top
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Model Accuracy", f"{predictions['accuracy']:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Records", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Features", df.shape[1])
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab layout for different visualizations
        tab1, tab2, tab3 = st.tabs([
            "📊 Predicted vs Actual", 
            "📈 Risk Distribution", 
            "📋 Classification Report"
        ])
        
        # Tab 1: Predicted vs Actual
        with tab1:
            st.subheader("Predicted vs Actual Risk Levels")
            
            col1_graph, col2_graph = st.columns(2)
            
            with col1_graph:
                st.markdown("#### Bar Graph")
                
                # Prepare data for bar chart
                risk_levels = ['High Risk', 'Medium Risk', 'Low Risk']
                actual_counts = [predictions['actual_counts'].get(level, 0) for level in risk_levels]
                predicted_counts = [predictions['predicted_counts'].get(level, 0) for level in risk_levels]
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(risk_levels))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, actual_counts, width, 
                              label='Actual', color='#2196F3', alpha=0.8)
                bars2 = ax.bar(x + width/2, predicted_counts, width, 
                              label='Predicted', color='#FF5722', alpha=0.8)
                
                ax.set_xlabel('Risk Level')
                ax.set_ylabel('Count')
                ax.set_title('Predicted vs Actual Risk Levels')
                ax.set_xticks(x)
                ax.set_xticklabels(risk_levels)
                ax.legend()
                
                # Add value labels on bars
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.annotate(f'{int(height)}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),  # 3 points vertical offset
                                   textcoords="offset points",
                                   ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display comparison table
                st.write("**Count Comparison:**")
                comparison_df = pd.DataFrame({
                    'Risk Level': risk_levels,
                    'Actual': actual_counts,
                    'Predicted': predicted_counts,
                    'Difference': np.array(predicted_counts) - np.array(actual_counts)
                })
                st.dataframe(comparison_df, use_container_width=True)
            
            with col2_graph:
                st.markdown("#### Line Graph")
                
                # Prepare data for line chart
                # Sort by life expectancy for better visualization
                sorted_df = df.sort_values('Life_Expectancy').reset_index(drop=True)
                
                # Create line chart
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot actual risk (convert to numeric for line)
                risk_mapping = {'High Risk': 0, 'Medium Risk': 1, 'Low Risk': 2}
                actual_numeric = sorted_df['Actual_Risk'].map(risk_mapping)
                predicted_numeric = sorted_df['Predicted_Risk'].map(risk_mapping)
                
                ax.plot(sorted_df.index, actual_numeric, 
                       label='Actual Risk', color='#2196F3', linewidth=2, marker='o', markersize=4)
                ax.plot(sorted_df.index, predicted_numeric, 
                       label='Predicted Risk', color='#FF5722', linewidth=2, marker='s', markersize=4, alpha=0.7)
                
                # Highlight mismatches
                mismatches = sorted_df[sorted_df['Actual_Risk'] != sorted_df['Predicted_Risk']].index
                if len(mismatches) > 0:
                    ax.scatter(mismatches, predicted_numeric[mismatches], 
                              color='red', s=100, zorder=5, label='Mismatch', alpha=0.6)
                
                # Customize y-axis
                ax.set_yticks([0, 1, 2])
                ax.set_yticklabels(['High Risk', 'Medium Risk', 'Low Risk'])
                ax.set_xlabel('Sample Index (sorted by Life Expectancy)')
                ax.set_ylabel('Risk Level')
                ax.set_title('Predicted vs Actual Risk (Sorted by Life Expectancy)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show mismatches
                mismatch_count = len(mismatches)
                st.write(f"**Mismatches:** {mismatch_count} out of {len(df)} samples ({mismatch_count/len(df)*100:.1f}%)")
        
        # Tab 2: Risk Distribution
        with tab2:
            st.subheader("Risk Level Distribution")
            
            col1_dist, col2_dist = st.columns(2)
            
            with col1_dist:
                st.markdown("#### Actual Risk Distribution")
                
                # Prepare data
                actual_counts = predictions['actual_counts']
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#FF5252', '#FFA726', '#4CAF50']  # Red, Orange, Green
                
                bars = ax.bar(actual_counts.index, actual_counts.values, 
                             color=colors, alpha=0.8)
                
                ax.set_xlabel('Risk Level')
                ax.set_ylabel('Count')
                ax.set_title('Actual Risk Distribution')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{int(height)}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show percentages
                st.write("**Actual Distribution (%):**")
                percentages = (actual_counts / actual_counts.sum() * 100).round(1)
                for risk, percent in percentages.items():
                    if risk == 'High Risk':
                        st.markdown(f'<span class="risk-high">{risk}: {percent}%</span>', unsafe_allow_html=True)
                    elif risk == 'Medium Risk':
                        st.markdown(f'<span class="risk-medium">{risk}: {percent}%</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="risk-low">{risk}: {percent}%</span>', unsafe_allow_html=True)
            
            with col2_dist:
                st.markdown("#### Predicted Risk Distribution")
                
                # Prepare data
                predicted_counts = predictions['predicted_counts']
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#FF5252', '#FFA726', '#4CAF50']
                
                bars = ax.bar(predicted_counts.index, predicted_counts.values, 
                             color=colors, alpha=0.8)
                
                ax.set_xlabel('Risk Level')
                ax.set_ylabel('Count')
                ax.set_title('Predicted Risk Distribution')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{int(height)}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show percentages
                st.write("**Predicted Distribution (%):**")
                percentages = (predicted_counts / predicted_counts.sum() * 100).round(1)
                for risk, percent in percentages.items():
                    if risk == 'High Risk':
                        st.markdown(f'<span class="risk-high">{risk}: {percent}%</span>', unsafe_allow_html=True)
                    elif risk == 'Medium Risk':
                        st.markdown(f'<span class="risk-medium">{risk}: {percent}%</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="risk-low">{risk}: {percent}%</span>', unsafe_allow_html=True)
            
            # Combined distribution comparison
            st.markdown("---")
            st.markdown("#### Distribution Comparison")
            
            # Create comparison chart
            fig, ax = plt.subplots(figsize=(12, 6))
            
            risk_levels = ['High Risk', 'Medium Risk', 'Low Risk']
            actual_percent = [(predictions['actual_counts'].get(level, 0) / len(df) * 100) for level in risk_levels]
            predicted_percent = [(predictions['predicted_counts'].get(level, 0) / len(df) * 100) for level in risk_levels]
            
            x = np.arange(len(risk_levels))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, actual_percent, width, 
                          label='Actual %', color='#2196F3', alpha=0.8)
            bars2 = ax.bar(x + width/2, predicted_percent, width, 
                          label='Predicted %', color='#FF5722', alpha=0.8)
            
            ax.set_xlabel('Risk Level')
            ax.set_ylabel('Percentage (%)')
            ax.set_title('Percentage Distribution: Actual vs Predicted')
            ax.set_xticks(x)
            ax.set_xticklabels(risk_levels)
            ax.legend()
            
            # Add percentage labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.1f}%',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Tab 3: Classification Report
        with tab3:
            st.subheader("Classification Report")
            
            # Calculate classification metrics
            from sklearn.metrics import classification_report
            
            # Generate classification report
            report = classification_report(df['Actual_Risk'], df['Predicted_Risk'], 
                                          output_dict=True, zero_division=0)
            
            # Convert to DataFrame
            report_df = pd.DataFrame(report).transpose()
            
            # Display as table
            st.write("**Detailed Classification Metrics:**")
            
            # Style the dataframe
            styled_df = report_df.style.format({
                'precision': '{:.2f}',
                'recall': '{:.2f}',
                'f1-score': '{:.2f}',
                'support': '{:.0f}'
            }).background_gradient(subset=['precision', 'recall', 'f1-score'], 
                                  cmap='YlOrRd', low=0.3, high=0.7)
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Visualize metrics
            st.markdown("#### Metrics Visualization")
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            metrics = ['precision', 'recall', 'f1-score']
            colors = ['#FF9800', '#2196F3', '#4CAF50']
            
            for idx, (metric, color) in enumerate(zip(metrics, colors)):
                ax = axes[idx]
                
                # Get values for each class
                values = []
                labels = []
                for class_name in ['High Risk', 'Medium Risk', 'Low Risk']:
                    if class_name in report:
                        values.append(report[class_name][metric])
                        labels.append(class_name[:1])  # First letter
                
                if values:
                    bars = ax.bar(range(len(values)), values, color=color, alpha=0.8)
                    ax.set_xticks(range(len(values)))
                    ax.set_xticklabels(labels)
                    ax.set_ylim([0, 1])
                    ax.set_title(f'{metric.capitalize()} by Class')
                    ax.set_ylabel('Score')
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        ax.annotate(f'{height:.2f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),
                                   textcoords="offset points",
                                   ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Confusion Matrix
            st.markdown("#### Confusion Matrix")
            
            from sklearn.metrics import confusion_matrix
            
            cm = confusion_matrix(df['Actual_Risk'], df['Predicted_Risk'], 
                                  labels=['High Risk', 'Medium Risk', 'Low Risk'])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['High', 'Medium', 'Low'],
                       yticklabels=['High', 'Medium', 'Low'],
                       ax=ax)
            ax.set_xlabel('Predicted Risk')
            ax.set_ylabel('Actual Risk')
            ax.set_title('Confusion Matrix')
            
            st.pyplot(fig)
            
            # Interpretation
            st.markdown("#### Interpretation")
            col_interpret1, col_interpret2 = st.columns(2)
            
            with col_interpret1:
                st.info("""
                **High Risk Class:**
                - Most critical to predict correctly
                - False negatives are costly
                - Precision indicates correct high-risk identification
                """)
            
            with col_interpret2:
                st.info("""
                **Overall Performance:**
                - Accuracy: Overall correct predictions
                - Weighted avg: Accounts for class imbalance
                - Support: Number of actual occurrences
                """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>📈 Life Expectancy Risk Analysis Dashboard | Streamlit</p>
        <p>Actual vs Predicted Risk Level Comparison</p>
    </div>
    """,
    unsafe_allow_html=True
)