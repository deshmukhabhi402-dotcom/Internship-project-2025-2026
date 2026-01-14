# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                            r2_score, classification_report, confusion_matrix)
import plotly.graph_objects as go
import plotly.express as px
import joblib
import io
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Health Risk Prediction Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class HealthRiskPredictor:
    def __init__(self):
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        
    def load_data(self, data_path="UnifiedDataset.csv"):
        """Load and preprocess the dataset"""
        try:
            self.df = pd.read_csv(data_path)
            st.success(f"Data loaded successfully! Shape: {self.df.shape}")
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def preprocess_data(self):
        """Preprocess the data"""
        if self.df is None:
            return False
        
        with st.spinner("Preprocessing data..."):
            # Create a copy for processing
            df_processed = self.df.copy()
            
            # Handle missing values
            # Numerical columns
            num_cols = df_processed.select_dtypes(include=np.number).columns
            df_processed[num_cols] = df_processed[num_cols].fillna(df_processed[num_cols].median())
            
            # Categorical columns
            cat_cols = df_processed.select_dtypes(exclude=np.number).columns
            for col in cat_cols:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
            
            # Encode categorical variables
            for col in cat_cols:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
            
            # Create Health Risk Level from Life Expectancy
            def risk_level(le):
                if le < 60:
                    return "High"
                elif le < 70:
                    return "Medium"
                else:
                    return "Low"
            
            df_processed["Health_Risk_Level"] = df_processed["Life Expectancy"].apply(risk_level)
            
            # Prepare features and target
            X = df_processed.drop(["Life Expectancy", "Health_Risk_Level"], axis=1)
            y = df_processed["Life Expectancy"]
            
            # Split the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            self.df_processed = df_processed
            
        st.success("Data preprocessing completed!")
        return True
    
    def train_model(self):
        """Train the Random Forest model"""
        with st.spinner("Training model..."):
            # Train Random Forest Regressor
            self.model = RandomForestRegressor(
                n_estimators=150,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            
            self.model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            self.y_pred = self.model.predict(self.X_test_scaled)
            
            # Calculate metrics
            self.mse = mean_squared_error(self.y_test, self.y_pred)
            self.mae = mean_absolute_error(self.y_test, self.y_pred)
            self.r2 = r2_score(self.y_test, self.y_pred)
            
            # Convert predictions to risk levels
            def risk_level(le):
                if le < 60:
                    return "High"
                elif le < 70:
                    return "Medium"
                else:
                    return "Low"
            
            y_test_risk = self.y_test.apply(risk_level)
            y_pred_risk = pd.Series(self.y_pred).apply(risk_level)
            
            # Generate classification report
            self.class_report = classification_report(
                y_test_risk, 
                y_pred_risk, 
                output_dict=True
            )
            
            # Generate confusion matrix
            self.conf_matrix = confusion_matrix(y_test_risk, y_pred_risk)
            
        st.success("Model training completed!")
        return True
    
    def get_feature_importance(self, top_n=20):
        """Get top N feature importances"""
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        })
        
        feature_importance = feature_importance.sort_values(
            'importance', ascending=False
        ).head(top_n)
        
        return feature_importance

def main():
    # Title
    st.markdown('<h1 class="main-header">🏥 Health Risk Prediction Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = HealthRiskPredictor()
    
    predictor = st.session_state.predictor
    
    # Sidebar
    with st.sidebar:
        st.image("🏥", width=100)
        st.title("Navigation")
        
        menu = st.selectbox(
            "Choose a section:",
            ["📊 Data Overview", "🔍 Data Analysis", "🤖 Model Training", 
             "📈 Results & Metrics", "🔮 Make Predictions"]
        )
        
        st.divider()
        
        if st.button("🔄 Reset Session"):
            st.session_state.clear()
            st.rerun()
    
    # Main content based on menu selection
    if menu == "📊 Data Overview":
        show_data_overview(predictor)
    
    elif menu == "🔍 Data Analysis":
        show_data_analysis(predictor)
    
    elif menu == "🤖 Model Training":
        show_model_training(predictor)
    
    elif menu == "📈 Results & Metrics":
        show_results_metrics(predictor)
    
    elif menu == "🔮 Make Predictions":
        show_predictions(predictor)

def show_data_overview(predictor):
    st.header("📊 Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📁 Load Data", type="primary"):
            if predictor.load_data():
                st.rerun()
    
    with col2:
        if st.button("🔄 Preprocess Data", type="secondary"):
            if predictor.preprocess_data():
                st.rerun()
    
    if predictor.df is not None:
        st.subheader("Dataset Preview")
        
        # Show basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", predictor.df.shape[0])
        with col2:
            st.metric("Total Columns", predictor.df.shape[1])
        with col3:
            missing_values = predictor.df.isnull().sum().sum()
            st.metric("Missing Values", missing_values)
        
        # Data preview
        st.dataframe(predictor.df.head(), use_container_width=True)
        
        # Data info
        with st.expander("📋 Dataset Information"):
            buffer = io.StringIO()
            predictor.df.info(buf=buffer)
            st.text(buffer.getvalue())
        
        # Missing values
        with st.expander("⚠️ Missing Values Analysis"):
            missing_df = pd.DataFrame({
                'Column': predictor.df.columns,
                'Missing_Values': predictor.df.isnull().sum(),
                'Percentage': (predictor.df.isnull().sum() / len(predictor.df)) * 100
            }).sort_values('Percentage', ascending=False)
            
            st.dataframe(missing_df[missing_df['Missing_Values'] > 0], 
                        use_container_width=True)
            
            # Plot missing values
            if len(missing_df[missing_df['Missing_Values'] > 0]) > 0:
                fig = px.bar(
                    missing_df[missing_df['Missing_Values'] > 0],
                    x='Column',
                    y='Percentage',
                    title='Missing Values Percentage by Column',
                    color='Percentage',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

def show_data_analysis(predictor):
    st.header("🔍 Data Analysis")
    
    if predictor.df is None:
        st.warning("Please load the data first from the Data Overview section.")
        return
    
    # Life Expectancy Distribution
    st.subheader("Life Expectancy Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig = px.histogram(
            predictor.df,
            x='Life Expectancy',
            nbins=50,
            title='Distribution of Life Expectancy',
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(
            xaxis_title='Life Expectancy',
            yaxis_title='Count'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot
        fig = px.box(
            predictor.df,
            y='Life Expectancy',
            title='Box Plot of Life Expectancy',
            color_discrete_sequence=['#ff7f0e']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    
    # Select numerical columns for correlation
    num_cols = predictor.df.select_dtypes(include=[np.number]).columns
    
    # Get top 15 features correlated with Life Expectancy
    if 'Life Expectancy' in num_cols:
        corr_with_target = predictor.df[num_cols].corr()['Life Expectancy'].abs().sort_values(ascending=False)
        top_corr_features = corr_with_target.index[1:16]  # Exclude Life Expectancy itself
        
        # Create correlation matrix for top features
        corr_matrix = predictor.df[top_corr_features.tolist() + ['Life Expectancy']].corr()
        
        # Plot heatmap
        fig = px.imshow(
            corr_matrix,
            title='Correlation Heatmap (Top 15 features with Life Expectancy)',
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1
        )
        fig.update_layout(width=800, height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk level distribution (after preprocessing)
    if hasattr(predictor, 'df_processed'):
        st.subheader("Health Risk Level Distribution")
        
        risk_counts = predictor.df_processed['Health_Risk_Level'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                names=risk_counts.index,
                values=risk_counts.values,
                title='Health Risk Level Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                title='Health Risk Level Counts',
                color=risk_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(
                xaxis_title='Risk Level',
                yaxis_title='Count'
            )
            st.plotly_chart(fig, use_container_width=True)

def show_model_training(predictor):
    st.header("🤖 Model Training")
    
    if predictor.df is None:
        st.warning("Please load and preprocess the data first.")
        return
    
    if not hasattr(predictor, 'df_processed'):
        st.warning("Please preprocess the data first from the Data Overview section.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_estimators = st.slider(
            "Number of Trees",
            min_value=50,
            max_value=500,
            value=150,
            step=50,
            help="Number of trees in the Random Forest"
        )
    
    with col2:
        max_depth = st.selectbox(
            "Max Depth",
            options=[None, 10, 20, 30, 50],
            index=0,
            help="Maximum depth of the tree"
        )
    
    with col3:
        min_samples_split = st.slider(
            "Min Samples Split",
            min_value=2,
            max_value=10,
            value=2,
            help="Minimum number of samples required to split an internal node"
        )
    
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Update model parameters
            predictor.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=1,
                random_state=42
            )
            
            predictor.model.fit(predictor.X_train_scaled, predictor.y_train)
            
            # Make predictions
            predictor.y_pred = predictor.model.predict(predictor.X_test_scaled)
            
            # Calculate metrics
            predictor.mse = mean_squared_error(predictor.y_test, predictor.y_pred)
            predictor.mae = mean_absolute_error(predictor.y_test, predictor.y_pred)
            predictor.r2 = r2_score(predictor.y_test, predictor.y_pred)
            
            # Convert predictions to risk levels
            def risk_level(le):
                if le < 60:
                    return "High"
                elif le < 70:
                    return "Medium"
                else:
                    return "Low"
            
            y_test_risk = predictor.y_test.apply(risk_level)
            y_pred_risk = pd.Series(predictor.y_pred).apply(risk_level)
            
            # Generate classification report
            predictor.class_report = classification_report(
                y_test_risk, 
                y_pred_risk, 
                output_dict=True
            )
            
            # Generate confusion matrix
            predictor.conf_matrix = confusion_matrix(y_test_risk, y_pred_risk)
            
            st.success("Model trained successfully!")
            st.balloons()
    
    # Show model information if trained
    if hasattr(predictor, 'model') and predictor.model is not None:
        st.subheader("Model Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Type", "Random Forest Regressor")
        
        with col2:
            st.metric("Number of Trees", predictor.model.n_estimators)
        
        with col3:
            st.metric("Features Used", predictor.X_train.shape[1])
        
        # Feature Importance
        st.subheader("Feature Importance")
        
        feature_importance = predictor.get_feature_importance(top_n=15)
        
        fig = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 15 Feature Importances',
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title='Importance',
            yaxis_title='Feature'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_results_metrics(predictor):
    st.header("📈 Results & Metrics")
    
    if not hasattr(predictor, 'model') or predictor.model is None:
        st.warning("Please train the model first from the Model Training section.")
        return
    
    # Regression Metrics
    st.subheader("Regression Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Squared Error", f"{predictor.mse:.4f}")
    
    with col2:
        st.metric("Mean Absolute Error", f"{predictor.mae:.4f}")
    
    with col3:
        st.metric("R² Score", f"{predictor.r2:.4f}")
    
    with col4:
        rmse = np.sqrt(predictor.mse)
        st.metric("Root MSE", f"{rmse:.4f}")
    
    # Actual vs Predicted Plot
    st.subheader("Actual vs Predicted Life Expectancy")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=predictor.y_test,
        y=predictor.y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(color='blue', opacity=0.6)
    ))
    
    # Add perfect prediction line
    min_val = min(predictor.y_test.min(), predictor.y_pred.min())
    max_val = max(predictor.y_test.max(), predictor.y_pred.max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='Actual vs Predicted Values',
        xaxis_title='Actual Life Expectancy',
        yaxis_title='Predicted Life Expectancy',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.subheader("Classification Report (Risk Levels)")
    
    if hasattr(predictor, 'class_report'):
        # Convert classification report to DataFrame for display
        class_report_df = pd.DataFrame(predictor.class_report).transpose()
        
        # Display as a styled table
        st.dataframe(
            class_report_df.style.format({
                'precision': '{:.2f}',
                'recall': '{:.2f}',
                'f1-score': '{:.2f}',
                'support': '{:.0f}'
            }).background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']),
            use_container_width=True
        )
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    
    if hasattr(predictor, 'conf_matrix'):
        # Get class labels (assuming they're in order)
        classes = ['High', 'Low', 'Medium']  # Based on your classification report
        
        # Create heatmap
        fig = px.imshow(
            predictor.conf_matrix,
            text_auto=True,
            color_continuous_scale='Blues',
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=classes,
            y=classes,
            title="Confusion Matrix"
        )
        
        fig.update_layout(
            xaxis_title='Predicted Label',
            yaxis_title='True Label'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Residual Analysis
    st.subheader("Residual Analysis")
    
    residuals = predictor.y_test - predictor.y_pred
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Residuals distribution
        fig = px.histogram(
            residuals,
            nbins=50,
            title='Distribution of Residuals',
            color_discrete_sequence=['#ff7f0e']
        )
        fig.update_layout(
            xaxis_title='Residuals',
            yaxis_title='Count'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Residuals vs Predicted
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=predictor.y_pred,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='green', opacity=0.6)
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title='Residuals vs Predicted Values',
            xaxis_title='Predicted Life Expectancy',
            yaxis_title='Residuals'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Error metrics by risk level
    st.subheader("Error Analysis by Risk Level")
    
    def risk_level(le):
        if le < 60:
            return "High"
        elif le < 70:
            return "Medium"
        else:
            return "Low"
    
    y_test_risk = predictor.y_test.apply(risk_level)
    errors = abs(predictor.y_test - predictor.y_pred)
    
    error_by_risk = pd.DataFrame({
        'Actual Risk': y_test_risk,
        'Error': errors
    })
    
    fig = px.box(
        error_by_risk,
        x='Actual Risk',
        y='Error',
        color='Actual Risk',
        title='Prediction Error Distribution by Risk Level',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        xaxis_title='Actual Risk Level',
        yaxis_title='Absolute Error'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_predictions(predictor):
    st.header("🔮 Make Predictions")
    
    if not hasattr(predictor, 'model') or predictor.model is None:
        st.warning("Please train the model first from the Model Training section.")
        return
    
    st.subheader("Enter Feature Values for Prediction")
    
    # Create input fields for top important features
    if hasattr(predictor, 'X_train'):
        # Get top 10 important features
        important_features = predictor.get_feature_importance(top_n=10)
        top_features = important_features['feature'].tolist()
        
        # Create input form
        input_data = {}
        
        cols = st.columns(2)
        
        for idx, feature in enumerate(top_features):
            with cols[idx % 2]:
                # Get min and max from training data for reference
                min_val = float(predictor.X_train[feature].min())
                max_val = float(predictor.X_train[feature].max())
                mean_val = float(predictor.X_train[feature].mean())
                
                input_data[feature] = st.number_input(
                    label=f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100,
                    help=f"Range: {min_val:.2f} to {max_val:.2f}"
                )
        
        # Fill remaining features with their means
        for feature in predictor.X_train.columns:
            if feature not in input_data:
                input_data[feature] = float(predictor.X_train[feature].mean())
        
        # Create prediction button
        if st.button("🎯 Predict Life Expectancy", type="primary"):
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure correct column order
            input_df = input_df[predictor.X_train.columns]
            
            # Scale the input
            input_scaled = predictor.scaler.transform(input_df)
            
            # Make prediction
            prediction = predictor.model.predict(input_scaled)[0]
            
            # Determine risk level
            if prediction < 60:
                risk = "High"
                risk_color = "red"
            elif prediction < 70:
                risk = "Medium"
                risk_color = "orange"
            else:
                risk = "Low"
                risk_color = "green"
            
            # Display results
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Predicted Life Expectancy</h3>
                    <h2 style="color: #1f77b4; font-size: 2.5rem;">{prediction:.2f} years</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Health Risk Level</h3>
                    <h2 style="color: {risk_color}; font-size: 2.5rem;">{risk}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Show feature importance for this prediction
            st.subheader("Feature Contribution to This Prediction")
            
            # Get feature importances
            feature_importance = predictor.get_feature_importance(top_n=10)
            
            # Display as horizontal bar chart
            fig = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title='Feature Importance (Global)',
                color='importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title='Importance',
                yaxis_title='Feature',
                height=400