import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Health Risk Classification Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px 0;
    }
    
    /* Sub-header styling */
    .sub-header {
        font-size: 1.8rem;
        color: #1E40AF;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 700;
        border-bottom: 3px solid #3B82F6;
        padding-bottom: 0.5rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        color: #374151;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
        font-weight: 600;
        background-color: #F3F4F6;
        padding: 8px 12px;
        border-radius: 8px;
        border-left: 4px solid #10B981;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #F3F4F6, #E5E7EB);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #D1D5DB;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #6B7280;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    
    .metric-value {
        font-size: 2rem;
        color: #1F2937;
        font-weight: 800;
    }
    
    /* Success and error indicators */
    .success-badge {
        display: inline-block;
        background-color: #10B981;
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 8px;
    }
    
    .error-badge {
        display: inline-block;
        background-color: #EF4444;
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 8px;
    }
    
    /* Styled tables */
    .styled-table {
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.9em;
        font-family: sans-serif;
        min-width: 400px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        border-radius: 10px;
        overflow: hidden;
    }
    
    .styled-table thead tr {
        background-color: #3B82F6;
        color: #ffffff;
        text-align: left;
    }
    
    .styled-table th,
    .styled-table td {
        padding: 12px 15px;
    }
    
    .styled-table tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    
    .styled-table tbody tr:last-of-type {
        border-bottom: 2px solid #3B82F6;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>🏥 Health Risk Classification Dashboard</h1>", unsafe_allow_html=True)

# Sidebar for data upload and model configuration
with st.sidebar:
    st.markdown("### 📁 Data & Model Configuration")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Dataset", type=['csv'], help="Upload your health dataset in CSV format")
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            # Display file info
            st.success(f"✅ File loaded successfully!")
            st.info(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
            
            # Select target variable (assuming 'Life Expectancy' exists)
            if 'Life Expectancy' in df.columns:
                st.session_state.target_column = 'Life Expectancy'
                st.success("✅ Target variable detected: 'Life Expectancy'")
            else:
                st.error("❌ 'Life Expectancy' column not found in dataset")
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    else:
        st.warning("⚠️ Please upload a CSV file to begin")
        # Display sample data structure
        st.markdown("""
        ### Expected Data Format:
        Your dataset should include:
        - Country, Year, Gender
        - **Life Expectancy** (target variable)
        - Infant Mortality Rate
        - Under 5 Mortality Rate
        - Various health indicators...
        """)

# Main content area
if 'df' in st.session_state and st.session_state.df is not None:
    df = st.session_state.df
    
    # Create health risk categories
    def categorize_risk(life_expectancy):
        if pd.isna(life_expectancy):
            return None
        if life_expectancy < 55:
            return 'High Risk'
        elif life_expectancy < 70:
            return 'Medium Risk'
        else:
            return 'Low Risk'
    
    # Apply categorization
    df['Health_Risk_Level'] = df[st.session_state.target_column].apply(categorize_risk)
    df_model = df.dropna(subset=['Health_Risk_Level'])
    
    # Display data overview
    st.markdown("<h2 class='sub-header'>📊 Dataset Overview</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Samples</div>
            <div class="metric-value">{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        risk_counts = df_model['Health_Risk_Level'].value_counts()
        low_risk = risk_counts.get('Low Risk', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Low Risk Cases</div>
            <div class="metric-value">{low_risk:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        medium_risk = risk_counts.get('Medium Risk', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Medium Risk Cases</div>
            <div class="metric-value">{medium_risk:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        high_risk = risk_counts.get('High Risk', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">High Risk Cases</div>
            <div class="metric-value">{high_risk:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Data preview
    with st.expander("📋 View Data Sample", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Model training section
    st.markdown("<h2 class='sub-header'>🤖 Model Training & Results</h2>", unsafe_allow_html=True)
    
    if st.button("🚀 Train Classification Model", type="primary", use_container_width=True):
        with st.spinner("Training model and generating results..."):
            try:
                # Define feature columns (similar to your notebook)
                feature_columns = [
                    'Year', 'Gender', 'Infant Mortality Rate', 'Under 5 Mortality Rate',
                    'Suicides Rate', 'Alcohol Abuse', 'Tobacco Prevalence', 
                    '% Death Cardiovascular', 'Incidence of Malaria', 'Incidence of Tuberculosis',
                    '% of Births Attended By Skilled Personal', 'Universal Heath Care Coverage',
                    'Air Pollution Death Rate Total', 'GDP per Capita', '% Population $1.90 a day',
                    'Doctors', 'Nurses and Midwifes', '% Population Aged 0-14', '% Population Aged 65+'
                ]
                
                # Only use columns that exist in the dataframe
                available_features = [col for col in feature_columns if col in df_model.columns]
                
                # Prepare data
                X = df_model[available_features]
                y = df_model['Health_Risk_Level']
                
                # Encode categorical variables
                label_encoder = LabelEncoder()
                if 'Gender' in X.columns:
                    X = X.copy()
                    X['Gender_encoded'] = label_encoder.fit_transform(X['Gender'])
                    X = X.drop('Gender', axis=1)
                
                # Encode target
                y_encoded = label_encoder.fit_transform(y)
                st.session_state.label_encoder = label_encoder
                
                # Handle missing values
                imputer = SimpleImputer(strategy='median')
                X_imputed = imputer.fit_transform(X)
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_imputed)
                
                # Split data
                X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                    X_scaled, y_encoded, df_model.index, test_size=0.2, 
                    random_state=42, stratify=y_encoded
                )
                
                # Train Random Forest model
                rf_classifier = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight='balanced'
                )
                rf_classifier.fit(X_train, y_train)
                
                # Make predictions
                y_pred = rf_classifier.predict(X_test)
                y_test_labels = label_encoder.inverse_transform(y_test)
                y_pred_labels = label_encoder.inverse_transform(y_pred)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test_labels, y_pred_labels)
                report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
                
                # Store results
                st.session_state.results = {
                    'y_test': y_test_labels,
                    'y_pred': y_pred_labels,
                    'accuracy': accuracy,
                    'report': report,
                    'class_names': label_encoder.classes_,
                    'model': rf_classifier,
                    'test_indices': idx_test
                }
                
                st.success("✅ Model trained successfully! Showing results below...")
                
            except Exception as e:
                st.error(f"❌ Error during model training: {str(e)}")
    
    # Display results if available
    if 'results' in st.session_state and st.session_state.results is not None:
        results = st.session_state.results
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Performance Overview", 
            "📈 Risk Distribution", 
            "📋 Classification Report", 
            "📉 Detailed Analysis"
        ])
        
        with tab1:
            st.markdown("<h3 class='section-header'>Model Performance Summary</h3>", unsafe_allow_html=True)
            
            # Performance metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Overall Accuracy</div>
                    <div class="metric-value">{results['accuracy']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                correct_predictions = (results['y_test'] == results['y_pred']).sum()
                total_predictions = len(results['y_test'])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Correct Predictions</div>
                    <div class="metric-value">{correct_predictions:,}/{total_predictions:,}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Risk Categories</div>
                    <div class="metric-value">{len(results['class_names'])}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Confusion Matrix
            st.markdown("<h3 class='section-header'>Confusion Matrix</h3>", unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(results['y_test'], results['y_pred'])
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=results['class_names'],
                       yticklabels=results['class_names'],
                       ax=ax)
            ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
            ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
            
            st.pyplot(fig)
        
        with tab2:
            st.markdown("<h3 class='section-header'>Risk Level Distribution Analysis</h3>", unsafe_allow_html=True)
            
            # Calculate distribution
            actual_counts = pd.Series(results['y_test']).value_counts()
            predicted_counts = pd.Series(results['y_pred']).value_counts()
            
            # Ensure all classes are present
            for cls in results['class_names']:
                if cls not in actual_counts:
                    actual_counts[cls] = 0
                if cls not in predicted_counts:
                    predicted_counts[cls] = 0
            
            # Sort by class order
            actual_counts = actual_counts[results['class_names']]
            predicted_counts = predicted_counts[results['class_names']]
            
            # Create side-by-side bar chart
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                x = np.arange(len(results['class_names']))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, actual_counts.values, width, 
                             label='Actual', color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
                bars2 = ax.bar(x + width/2, predicted_counts.values, width, 
                             label='Predicted', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
                
                # Customize the chart
                ax.set_xlabel('Risk Level', fontsize=12, fontweight='bold')
                ax.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
                ax.set_title('Actual vs Predicted Risk Distribution', fontsize=14, fontweight='bold', pad=20)
                ax.set_xticks(x)
                ax.set_xticklabels(results['class_names'], fontsize=11)
                ax.legend(fontsize=11)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                
                # Add value labels on bars
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.annotate(f'{int(height)}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),  # 3 points vertical offset
                                   textcoords="offset points",
                                   ha='center', va='bottom',
                                   fontsize=10, fontweight='bold')
                
                # Add percentage labels
                for i, (actual, predicted) in enumerate(zip(actual_counts.values, predicted_counts.values)):
                    actual_pct = (actual / len(results['y_test']) * 100)
                    predicted_pct = (predicted / len(results['y_pred']) * 100)
                    
                    ax.text(i - width/2, actual + max(actual_counts.max(), predicted_counts.max())*0.02,
                           f'{actual_pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
                    ax.text(i + width/2, predicted + max(actual_counts.max(), predicted_counts.max())*0.02,
                           f'{predicted_pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                st.pyplot(fig)
            
            with col2:
                # Display distribution table
                st.markdown("#### 📊 Distribution Table")
                dist_data = []
                for risk in results['class_names']:
                    actual_count = actual_counts[risk]
                    predicted_count = predicted_counts[risk]
                    actual_pct = (actual_count / len(results['y_test']) * 100)
                    predicted_pct = (predicted_count / len(results['y_pred']) * 100)
                    
                    dist_data.append({
                        'Risk Level': risk,
                        'Actual': f"{actual_count} ({actual_pct:.1f}%)",
                        'Predicted': f"{predicted_count} ({predicted_pct:.1f}%)"
                    })
                
                dist_df = pd.DataFrame(dist_data)
                st.dataframe(dist_df, use_container_width=True)
                
                # Calculate and display accuracy by class
                st.markdown("#### 🎯 Accuracy by Risk Level")
                accuracy_by_class = []
                for risk in results['class_names']:
                    mask = results['y_test'] == risk
                    if mask.sum() > 0:
                        class_accuracy = (results['y_test'][mask] == results['y_pred'][mask]).mean()
                        accuracy_by_class.append({
                            'Risk Level': risk,
                            'Accuracy': f"{class_accuracy:.1%}"
                        })
                
                accuracy_df = pd.DataFrame(accuracy_by_class)
                st.dataframe(accuracy_df, use_container_width=True)
        
        with tab3:
            st.markdown("<h3 class='section-header'>Detailed Classification Report</h3>", unsafe_allow_html=True)
            
            # Convert classification report to DataFrame
            report_df = pd.DataFrame(results['report']).transpose()
            
            # Display the report with styling
            st.dataframe(
                report_df.style.format({
                    'precision': '{:.2f}',
                    'recall': '{:.2f}',
                    'f1-score': '{:.2f}',
                    'support': '{:.0f}'
                }).background_gradient(subset=['precision', 'recall', 'f1-score'], cmap='YlOrRd'),
                use_container_width=True
            )
            
            # Create visual representation of precision, recall, f1-score
            st.markdown("<h3 class='section-header'>Performance Metrics Visualization</h3>", unsafe_allow_html=True)
            
            # Filter out 'accuracy', 'macro avg', 'weighted avg' for the bar chart
            metrics_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
            
            if not metrics_df.empty:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                metrics = ['precision', 'recall', 'f1-score']
                colors = ['#3498db', '#2ecc71', '#e74c3c']
                
                for ax, metric, color in zip(axes, metrics, colors):
                    if metric in metrics_df.columns:
                        bars = ax.bar(metrics_df.index, metrics_df[metric], color=color, alpha=0.8, edgecolor='black')
                        ax.set_title(f'{metric.title()} by Class', fontsize=12, fontweight='bold')
                        ax.set_xlabel('Risk Level', fontsize=10)
                        ax.set_ylabel(metric.title(), fontsize=10)
                        ax.set_ylim([0, 1.1])
                        ax.grid(axis='y', alpha=0.3)
                        
                        # Add value labels on bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.annotate(f'{height:.2f}',
                                       xy=(bar.get_x() + bar.get_width() / 2, height),
                                       xytext=(0, 3),
                                       textcoords="offset points",
                                       ha='center', va='bottom',
                                       fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        with tab4:
            st.markdown("<h3 class='section-header'>Detailed Analysis & Sample Predictions</h3>", unsafe_allow_html=True)
            
            # Sample predictions comparison
            st.markdown("#### 🔍 Sample Predictions (15 Random Samples)")
            
            # Create comparison DataFrame
            comparison_df = pd.DataFrame({
                'Index': results['test_indices'],
                'Actual_Risk': results['y_test'],
                'Predicted_Risk': results['y_pred']
            })
            
            # Take sample
            sample_comparison = comparison_df.sample(min(15, len(comparison_df)), random_state=42)
            sample_comparison['Match'] = sample_comparison['Actual_Risk'] == sample_comparison['Predicted_Risk']
            
            # Display each sample in a styled card
            for idx, row in sample_comparison.iterrows():
                match_symbol = "✅" if row['Match'] else "❌"
                match_class = "success-badge" if row['Match'] else "error-badge"
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>Index {row['Index']}</strong><br>
                                <span style="color: #4B5563; font-size: 0.9rem;">Actual: <strong style="color: #1E3A8A">{row['Actual_Risk']}</strong></span><br>
                                <span style="color: #4B5563; font-size: 0.9rem;">Predicted: <strong style="color: #059669">{row['Predicted_Risk']}</strong></span>
                            </div>
                            <span class="{match_class}">{match_symbol}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Detailed error analysis
            st.markdown("#### 📉 Error Analysis")
            
            # Calculate confusion summary
            confusion_summary = {}
            for actual in results['class_names']:
                for predicted in results['class_names']:
                    count = ((results['y_test'] == actual) & (results['y_pred'] == predicted)).sum()
                    if count > 0 and actual != predicted:
                        confusion_summary[f"{actual} → {predicted}"] = count
            
            if confusion_summary:
                error_df = pd.DataFrame({
                    'Misclassification': list(confusion_summary.keys()),
                    'Count': list(confusion_summary.values())
                }).sort_values('Count', ascending=False)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.barh(error_df['Misclassification'], error_df['Count'], color='#e74c3c', alpha=0.8)
                    ax.set_xlabel('Number of Misclassifications', fontsize=11)
                    ax.set_title('Top Misclassifications', fontsize=13, fontweight='bold')
                    ax.invert_yaxis()  # Highest on top
                    
                    # Add value labels
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                               f'{int(width)}', va='center', fontsize=10, fontweight='bold')
                    
                    st.pyplot(fig)
                
                with col2:
                    st.dataframe(error_df, use_container_width=True)
            else:
                st.success("🎉 No misclassifications found!")
    
    else:
        # Instructions when no model is trained yet
        st.markdown("""
        <div class="info-box">
            <h4>📋 Instructions:</h4>
            <ol>
                <li>Upload your health dataset using the sidebar</li>
                <li>Click the "Train Classification Model" button above</li>
                <li>View the results in the tabs below</li>
            </ol>
            <p>The model will classify health risk into three categories:</p>
            <ul>
                <li><strong>Low Risk:</strong> Life Expectancy ≥ 70</li>
                <li><strong>Medium Risk:</strong> 55 ≤ Life Expectancy < 70</li>
                <li><strong>High Risk:</strong> Life Expectancy < 55</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Display sample visualization
        st.markdown("<h3 class='section-header'>📊 Expected Output Visualizations</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("https://via.placeholder.com/600x400/3498db/ffffff?text=Risk+Distribution+Chart", 
                    caption="Risk Distribution Bar Chart")
        
        with col2:
            st.image("https://via.placeholder.com/600x400/2ecc71/ffffff?text=Confusion+Matrix", 
                    caption="Confusion Matrix Heatmap")

else:
    # Welcome screen when no data is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
        <h2 style="font-size: 2.5rem; margin-bottom: 1rem;">Welcome to Health Risk Classification Dashboard</h2>
        <p style="font-size: 1.2rem; margin-bottom: 2rem;">Upload your health dataset to analyze and predict health risk levels</p>
        
        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 3rem;">
            <div style="text-align: center;">
                <div style="font-size: 3rem;">📊</div>
                <h3>Data Analysis</h3>
                <p>Upload and explore your health data</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 3rem;">🤖</div>
                <h3>Model Training</h3>
                <p>Train Random Forest classifier</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 3rem;">📈</div>
                <h3>Visualizations</h3>
                <p>View comprehensive charts and graphs</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 3rem;">📋</div>
                <h3>Reports</h3>
                <p>Detailed classification reports</p>
            </div>
        </div>
    </div>
    
    <div style="margin-top: 3rem;">
        <h3>🚀 Get Started:</h3>
        <ol>
            <li>Use the sidebar on the left to upload your CSV dataset</li>
            <li>Ensure your dataset includes a 'Life Expectancy' column</li>
            <li>Click the 'Train Classification Model' button</li>
            <li>Explore the results through interactive visualizations</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem; padding: 1rem;">
    <p style="margin-bottom: 0.5rem;">🏥 <strong>Health Risk Classification Dashboard</strong> • Powered by Streamlit & Scikit-learn</p>
    <p style="font-size: 0.8rem;">Classify health risk levels based on life expectancy and various health indicators</p>
</div>
""", unsafe_allow_html=True)