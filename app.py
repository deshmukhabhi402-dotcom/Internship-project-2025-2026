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
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(
                    missing_df[missing_df['Missing_Values'] > 0]['Column'][:15],
                    missing_df[missing_df['Missing_Values'] > 0]['Percentage'][:15]
                )
                ax.set_xlabel('Percentage Missing')
                ax.set_title('Missing Values Percentage by Column (Top 15)')
                plt.tight_layout()
                st.pyplot(fig)

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
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(predictor.df['Life Expectancy'], bins=50, color='#1f77b4', alpha=0.7)
        ax.set_xlabel('Life Expectancy')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Life Expectancy')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        # Box plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(predictor.df['Life Expectancy'], vert=False, patch_artist=True,
                  boxprops=dict(facecolor='#ff7f0e'))
        ax.set_xlabel('Life Expectancy')
        ax.set_title('Box Plot of Life Expectancy')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
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
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Correlation Heatmap (Top 15 features with Life Expectancy)')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Risk level distribution (after preprocessing)
    if hasattr(predictor, 'df_processed'):
        st.subheader("Health Risk Level Distribution")
        
        risk_counts = predictor.df_processed['Health_Risk_Level'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = plt.cm.Set3(np.linspace(0, 1, len(risk_counts)))
            ax.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                  colors=colors, startangle=90)
            ax.set_title('Health Risk Level Distribution')
            ax.axis('equal')
            st.pyplot(fig)
        
        with col2:
            # Bar chart
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = plt.cm.Set3(np.linspace(0, 1, len(risk_counts)))
            bars = ax.bar(risk_counts.index, risk_counts.values, color=colors)
            ax.set_xlabel('Risk Level')
            ax.set_ylabel('Count')
            ax.set_title('Health Risk Level Counts')
            
            # Add count labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)

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
        
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(feature_importance))
        bars = ax.barh(y_pos, feature_importance['importance'].values)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_importance['feature'].values)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Importance')
        ax.set_title('Top 15 Feature Importances')
        
        # Color bars by importance
        cmap = plt.cm.viridis
        for i, bar in enumerate(bars):
            bar.set_color(cmap(i / len(bars)))
        
        plt.tight_layout()
        st.pyplot(fig)

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
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(predictor.y_test, predictor.y_pred, alpha=0.6, color='blue', label='Predictions')
    
    # Add perfect prediction line
    min_val = min(predictor.y_test.min(), predictor.y_pred.min())
    max_val = max(predictor.y_test.max(), predictor.y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    ax.set_xlabel('Actual Life Expectancy')
    ax.set_ylabel('Predicted Life Expectancy')
    ax.set_title('Actual vs Predicted Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add R² text
    ax.text(0.05, 0.95, f'R² = {predictor.r2:.3f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    st.pyplot(fig)
    
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
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(predictor.conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Residual Analysis
    st.subheader("Residual Analysis")
    
    residuals = predictor.y_test - predictor.y_pred
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Residuals distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(residuals, bins=50, color='#ff7f0e', alpha=0.7)
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Residuals')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        # Residuals vs Predicted
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(predictor.y_pred, residuals, alpha=0.6, color='green')
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Life Expectancy')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Predicted Values')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
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
    
    fig, ax = plt.subplots(figsize=(10, 6))
    boxplot = ax.boxplot([error_by_risk[error_by_risk['Actual Risk'] == risk]['Error'].values 
                         for risk in ['High', 'Medium', 'Low']],
                        labels=['High', 'Medium', 'Low'],
                        patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, 3))
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xlabel('Actual Risk Level')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Prediction Error Distribution by Risk Level')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

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
            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(feature_importance))
            bars = ax.barh(y_pos, feature_importance['importance'].values)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_importance['feature'].values)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance (Global)')
            
            # Color bars by importance
            cmap = plt.cm.viridis
            for i, bar in enumerate(bars):
                bar.set_color(cmap(i / len(bars)))
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show the input values for top features
            with st.expander("📋 Input Values Used"):
                top_input_data = {k: input_data[k] for k in top_features}
                st.json(top_input_data)

if __name__ == "__main__":
    main()