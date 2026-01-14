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

# Set page configuration
st.set_page_config(
    page_title="Health Risk Classification Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E40AF;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .success {
        color: #10B981;
        font-weight: bold;
    }
    .error {
        color: #EF4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>🏥 Health Risk Classification Dashboard</h1>", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Model Training", "Predictions", "Visualizations", "Download Results"])

# Initialize session state for storing data and model
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None

# Page 1: Data Overview
if page == "Data Overview":
    st.markdown("<h2 class='sub-header'>📊 Data Overview</h2>", unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            # Display basic info
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Rows", df.shape[0])
                st.metric("Total Columns", df.shape[1])
                
            with col2:
                st.metric("Missing Values", df.isnull().sum().sum())
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                st.metric("Numeric Columns", len(numeric_cols))
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Data info
            st.subheader("Data Information")
            buffer = BytesIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            
            # Column selector for target variable
            st.subheader("Select Features and Target")
            
            all_columns = df.columns.tolist()
            default_features = [
                'Year', 'Gender', 'Infant Mortality Rate', 'Under 5 Mortality Rate',
                'Suicides Rate', 'Alcohol Abuse', 'Tobacco Prevalence', 
                '% Death Cardiovascular', 'Incidence of Malaria', 'Incidence of Tuberculosis',
                '% of Births Attended By Skilled Personal', 'Universal Heath Care Coverage',
                'Air Pollution Death Rate Total', 'GDP per Capita', '% Population $1.90 a day',
                'Doctors', 'Nurses and Midwifes', '% Population Aged 0-14', '% Population Aged 65+'
            ]
            
            # Only include columns that exist in the dataframe
            available_features = [col for col in default_features if col in all_columns]
            
            selected_features = st.multiselect(
                "Select features for the model",
                all_columns,
                default=available_features
            )
            
            target_options = [col for col in all_columns if col not in selected_features]
            selected_target = st.selectbox(
                "Select target variable (Life Expectancy)",
                target_options
            )
            
            if st.button("Prepare Data for Modeling"):
                st.session_state.selected_features = selected_features
                st.session_state.selected_target = selected_target
                st.success("Data prepared successfully! Navigate to 'Model Training' to continue.")
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    else:
        st.info("Please upload a CSV file to begin analysis")
        st.markdown("""
        ### Expected Data Format:
        The dataset should contain health-related features including:
        - Country
        - Year
        - Gender
        - Life Expectancy
        - Infant Mortality Rate
        - Under 5 Mortality Rate
        - And other health indicators...
        """)

# Page 2: Model Training
elif page == "Model Training":
    st.markdown("<h2 class='sub-header'>🤖 Model Training</h2>", unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload data first from the 'Data Overview' page")
    else:
        df = st.session_state.df
        
        # Define risk categorization function
        def categorize_risk(life_expectancy):
            if pd.isna(life_expectancy):
                return None
            if life_expectancy < 55:
                return 'High Risk'
            elif life_expectancy < 70:
                return 'Medium Risk'
            else:
                return 'Low Risk'
        
        # Create health risk level
        df['Health_Risk_Level'] = df[st.session_state.selected_target].apply(categorize_risk)
        
        # Model parameters
        st.subheader("Model Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_estimators = st.slider("Number of Trees", 50, 500, 100, 50)
            max_depth = st.slider("Max Depth", 5, 50, 10, 5)
            
        with col2:
            min_samples_split = st.slider("Min Samples Split", 2, 20, 5, 1)
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 2, 1)
            
        with col3:
            test_size = st.slider("Test Size (%)", 10, 40, 20, 5) / 100
            random_state = st.number_input("Random State", 0, 100, 42)
        
        if st.button("Train Model", type="primary"):
            with st.spinner("Training model..."):
                try:
                    # Prepare data
                    features = st.session_state.selected_features
                    df_model = df[features + ['Health_Risk_Level']].copy()
                    df_model = df_model.dropna(subset=['Health_Risk_Level'])
                    
                    # Encode categorical variables
                    if 'Gender' in features:
                        label_encoder = LabelEncoder()
                        df_model['Gender_encoded'] = label_encoder.fit_transform(df_model['Gender'])
                        features_to_use = [col for col in df_model.columns if col not in ['Health_Risk_Level', 'Gender']]
                    else:
                        features_to_use = [col for col in features if col in df_model.columns]
                    
                    X = df_model[features_to_use]
                    y = df_model['Health_Risk_Level']
                    
                    # Encode target
                    target_encoder = LabelEncoder()
                    y_encoded = target_encoder.fit_transform(y)
                    st.session_state.label_encoder = target_encoder
                    
                    # Handle missing values
                    imputer = SimpleImputer(strategy='median')
                    X_imputed = imputer.fit_transform(X)
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_imputed)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y_encoded, test_size=test_size, 
                        random_state=random_state, stratify=y_encoded
                    )
                    
                    # Train model
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_state,
                        class_weight='balanced'
                    )
                    
                    model.fit(X_train, y_train)
                    st.session_state.model = model
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_test_labels = target_encoder.inverse_transform(y_test)
                    y_pred_labels = target_encoder.inverse_transform(y_pred)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test_labels, y_pred_labels)
                    report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
                    
                    # Store results
                    st.session_state.results = {
                        'X_test': X_test,
                        'y_test': y_test_labels,
                        'y_pred': y_pred_labels,
                        'accuracy': accuracy,
                        'report': report,
                        'class_names': target_encoder.classes_,
                        'feature_names': features_to_use
                    }
                    
                    # Display results
                    st.success("Model trained successfully!")
                    
                    # Show metrics
                    st.subheader("Model Performance")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.2%}")
                    with col2:
                        st.metric("Total Samples", len(y_test))
                    with col3:
                        st.metric("Classes", len(target_encoder.classes_))
                    
                    # Feature importance
                    st.subheader("Top 10 Feature Importances")
                    feature_importance = pd.DataFrame({
                        'feature': features_to_use,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(feature_importance.head(10)['feature'][::-1], 
                           feature_importance.head(10)['importance'][::-1])
                    ax.set_xlabel('Importance')
                    ax.set_title('Top 10 Feature Importances')
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")

# Page 3: Predictions
elif page == "Predictions":
    st.markdown("<h2 class='sub-header'>🔮 Model Predictions</h2>", unsafe_allow_html=True)
    
    if st.session_state.results is None:
        st.warning("Please train the model first from the 'Model Training' page")
    else:
        results = st.session_state.results
        
        # Show sample predictions
        st.subheader("Sample Predictions (15 random samples)")
        
        comparison_df = pd.DataFrame({
            'Actual_Risk': results['y_test'],
            'Predicted_Risk': results['y_pred']
        })
        
        sample_comparison = comparison_df.sample(min(15, len(comparison_df)), random_state=42)
        sample_comparison['Match'] = sample_comparison['Actual_Risk'] == sample_comparison['Predicted_Risk']
        
        # Display sample predictions in a table
        for idx, (_, row) in enumerate(sample_comparison.iterrows()):
            match_symbol = "✓" if row['Match'] else "✗"
            match_class = "success" if row['Match'] else "error"
            st.markdown(f"""
            <div class="metric-card">
                <strong>Sample {idx+1}:</strong><br>
                Actual: <strong>{row['Actual_Risk']}</strong> | 
                Predicted: <strong>{row['Predicted_Risk']}</strong> | 
                <span class="{match_class}">{match_symbol}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Overall accuracy
        correct_predictions = (comparison_df['Actual_Risk'] == comparison_df['Predicted_Risk']).sum()
        total_predictions = len(comparison_df)
        accuracy = correct_predictions / total_predictions
        
        st.subheader("Overall Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Correct Predictions", f"{correct_predictions}/{total_predictions}")
        with col2:
            st.metric("Accuracy", f"{accuracy:.2%}")
        
        # Detailed classification report
        st.subheader("Detailed Classification Report")
        
        report_df = pd.DataFrame(results['report']).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))

# Page 4: Visualizations
elif page == "Visualizations":
    st.markdown("<h2 class='sub-header'>📈 Data Visualizations</h2>", unsafe_allow_html=True)
    
    if st.session_state.results is None:
        st.warning("Please train the model first from the 'Model Training' page")
    else:
        results = st.session_state.results
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Risk Distribution", "Performance Metrics"])
        
        with tab1:
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            
            cm = confusion_matrix(results['y_test'], results['y_pred'])
            cm_df = pd.DataFrame(cm, 
                                index=results['class_names'], 
                                columns=results['class_names'])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            
        with tab2:
            # Risk Distribution
            st.subheader("Risk Level Distribution")
            
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
            
            # Create bar chart
            x = np.arange(len(results['class_names']))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x - width/2, actual_counts.values, width, label='Actual', 
                  color='#3498db', edgecolor='black', alpha=0.8)
            ax.bar(x + width/2, predicted_counts.values, width, label='Predicted', 
                  color='#2ecc71', edgecolor='black', alpha=0.8)
            
            ax.set_xlabel('Risk Level')
            ax.set_ylabel('Number of Cases')
            ax.set_title('Actual vs Predicted Risk Distribution')
            ax.set_xticks(x)
            ax.set_xticklabels(results['class_names'])
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, (actual_val, predicted_val) in enumerate(zip(actual_counts.values, predicted_counts.values)):
                ax.text(i - width/2, actual_val, str(actual_val), 
                       ha='center', va='bottom', fontweight='bold')
                ax.text(i + width/2, predicted_val, str(predicted_val), 
                       ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
            
            # Display distribution table
            st.subheader("Distribution Details")
            dist_data = []
            for risk in results['class_names']:
                actual_count = actual_counts[risk]
                predicted_count = predicted_counts[risk]
                actual_percent = (actual_count / len(results['y_test']) * 100)
                predicted_percent = (predicted_count / len(results['y_pred']) * 100)
                
                dist_data.append({
                    'Risk Level': risk,
                    'Actual Count': actual_count,
                    'Actual %': f"{actual_percent:.1f}%",
                    'Predicted Count': predicted_count,
                    'Predicted %': f"{predicted_percent:.1f}%"
                })
            
            st.dataframe(pd.DataFrame(dist_data))
            
        with tab3:
            # Performance Metrics by Class
            st.subheader("Performance Metrics by Risk Level")
            
            precision, recall, f1, support = precision_recall_fscore_support(
                results['y_test'], results['y_pred'], labels=results['class_names']
            )
            
            metrics_df = pd.DataFrame({
                'Risk Level': results['class_names'],
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Support': support
            })
            
            st.dataframe(metrics_df.style.format({
                'Precision': '{:.3f}',
                'Recall': '{:.3f}',
                'F1-Score': '{:.3f}'
            }))
            
            # Radar chart for metrics
            st.subheader("Performance Radar Chart")
            
            # Prepare data for radar chart
            categories = ['Precision', 'Recall', 'F1-Score']
            fig = go.Figure()
            
            for i, risk_level in enumerate(results['class_names']):
                values = [
                    metrics_df.loc[metrics_df['Risk Level'] == risk_level, 'Precision'].values[0],
                    metrics_df.loc[metrics_df['Risk Level'] == risk_level, 'Recall'].values[0],
                    metrics_df.loc[metrics_df['Risk Level'] == risk_level, 'F1-Score'].values[0]
                ]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=risk_level
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Performance Metrics by Risk Level"
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Page 5: Download Results
elif page == "Download Results":
    st.markdown("<h2 class='sub-header'>📥 Download Results</h2>", unsafe_allow_html=True)
    
    if st.session_state.results is None:
        st.warning("No results available. Please train the model first.")
    else:
        results = st.session_state.results
        
        # Create comprehensive results dataframe
        comparison_df = pd.DataFrame({
            'Actual_Risk': results['y_test'],
            'Predicted_Risk': results['y_pred'],
            'Correct': results['y_test'] == results['y_pred']
        })
        
        # Create summary statistics
        summary_df = pd.DataFrame(results['report']).transpose()
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Download predictions
            csv = comparison_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions (CSV)",
                data=csv,
                file_name="health_risk_predictions.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download summary
            csv_summary = summary_df.to_csv()
            st.download_button(
                label="Download Summary Report (CSV)",
                data=csv_summary,
                file_name="health_risk_summary.csv",
                mime="text/csv"
            )
        
        # Display sample of downloadable data
        st.subheader("Sample of Downloadable Data")
        st.dataframe(comparison_df.head())
        
        # Information about the data
        st.info("""
        The downloadable files contain:
        1. **Predictions CSV**: Individual predictions with actual vs predicted values
        2. **Summary CSV**: Complete classification report with precision, recall, and F1-scores
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <p>Health Risk Classification Dashboard • Built with Streamlit</p>
    <p>Upload your health dataset to predict risk levels based on various health indicators</p>
</div>
""", unsafe_allow_html=True)