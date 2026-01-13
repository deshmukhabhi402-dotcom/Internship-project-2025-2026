# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Life Expectancy Risk Analyzer",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        background: linear-gradient(45deg, #1E88E5, #00ACC1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .risk-high { color: #FF5252; font-weight: bold; }
    .risk-medium { color: #FFA726; font-weight: bold; }
    .risk-low { color: #66BB6A; font-weight: bold; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .section-header {
        background: linear-gradient(90deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 5px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">📈 Life Expectancy Risk Level Analyzer</h1>', unsafe_allow_html=True)
st.markdown("### Predict Health Risk Levels Based on Life Expectancy Factors")

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'risk_predictions' not in st.session_state:
    st.session_state.risk_predictions = None

# Sidebar Navigation
with st.sidebar:
    st.title("🔍 Navigation")
    page = st.radio(
        "Select a page:",
        ["🏠 Home & Data Upload", "📊 Data Processing", "🎯 Risk Analysis", "📈 Visualizations"]
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This tool analyzes life expectancy data and 
    categorizes risk levels (High/Medium/Low) 
    based on various health and socioeconomic factors.
    """)

# Page 1: Home & Data Upload
if page == "🏠 Home & Data Upload":
    st.markdown('<div class="section-header"><h2>📥 Upload Your Life Expectancy Data</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Data Source Options")
        
        # Option 1: File Upload
        uploaded_file = st.file_uploader(
            "Upload your CSV file", 
            type=['csv'],
            help="Upload a dataset with life expectancy and related factors"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.raw_data = df
                st.success(f"✅ Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns")
                
                # Quick preview
                with st.expander("📋 Preview First 5 Rows"):
                    st.dataframe(df.head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        # Option 2: Sample Data
        with st.expander("📊 Use Sample Data"):
            if st.button("Load Sample Life Expectancy Dataset"):
                # Create sample data similar to your project
                np.random.seed(42)
                n_samples = 200
                
                sample_data = pd.DataFrame({
                    'Country': ['Country_' + str(i) for i in range(n_samples)],
                    'Year': np.random.randint(2000, 2020, n_samples),
                    'Life_Expectancy': np.random.uniform(50, 85, n_samples),
                    'Adult_Mortality': np.random.uniform(50, 300, n_samples),
                    'Infant_Deaths': np.random.randint(0, 100, n_samples),
                    'Alcohol': np.random.uniform(0.1, 15, n_samples),
                    'Percentage_Expenditure': np.random.uniform(0, 2000, n_samples),
                    'Hepatitis_B': np.random.uniform(50, 100, n_samples),
                    'Measles': np.random.randint(0, 1000, n_samples),
                    'BMI': np.random.uniform(15, 40, n_samples),
                    'Polio': np.random.uniform(50, 100, n_samples),
                    'Diphtheria': np.random.uniform(50, 100, n_samples),
                    'HIV_AIDS': np.random.uniform(0, 5, n_samples),
                    'GDP': np.random.uniform(100, 50000, n_samples),
                    'Population': np.random.uniform(10000, 10000000, n_samples),
                    'Schooling': np.random.uniform(5, 20, n_samples)
                })
                
                st.session_state.raw_data = sample_data
                st.success("✅ Sample dataset loaded successfully!")
                st.dataframe(sample_data.head(), use_container_width=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Expected Columns", "15+")
        st.write("Including Life Expectancy, Mortality Rates, Health Indicators")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 📋 Data Requirements")
        st.info("""
        Your dataset should include:
        - **Life_Expectancy** (Target Variable)
        - Health indicators (BMI, Alcohol, etc.)
        - Mortality rates
        - Socioeconomic factors
        - Country/Year identifiers
        """)

# Page 2: Data Processing
elif page == "📊 Data Processing":
    st.markdown('<div class="section-header"><h2>⚙️ Data Processing Pipeline</h2></div>', unsafe_allow_html=True)
    
    if 'raw_data' not in st.session_state or st.session_state.raw_data is None:
        st.warning("⚠️ Please upload data first on the Home page.")
    else:
        df = st.session_state.raw_data.copy()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Original Data Shape", f"{df.shape[0]} × {df.shape[1]}")
        
        with col2:
            missing_values = df.isnull().sum().sum()
            st.metric("Missing Values", f"{missing_values}")
        
        with col3:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Columns", f"{numeric_cols}")
        
        # Processing steps
        st.subheader("🔧 Processing Steps")
        
        # Step 1: Handle missing values
        with st.expander("1️⃣ Handle Missing Values", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                missing_strategy = st.selectbox(
                    "Strategy for numeric columns:",
                    ['Mean', 'Median', 'Zero']
                )
            
            with col2:
                if st.button("Apply Missing Value Treatment"):
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if missing_strategy == 'Mean':
                        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                    elif missing_strategy == 'Median':
                        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                    else:
                        df[numeric_cols] = df[numeric_cols].fillna(0)
                    
                    st.success(f"✅ Filled {missing_values} missing values using {missing_strategy}")
        
        # Step 2: Identify target variable
        with st.expander("2️⃣ Target Variable Configuration", expanded=True):
            if 'Life_Expectancy' in df.columns:
                target_col = 'Life_Expectancy'
                st.success(f"✅ Target variable found: **{target_col}**")
            else:
                target_col = st.selectbox("Select life expectancy column:", df.columns)
            
            # Create risk categories based on life expectancy
            if st.button("Create Risk Categories"):
                if target_col in df.columns:
                    # Categorize into risk levels
                    df['Risk_Level'] = pd.cut(
                        df[target_col],
                        bins=[0, 60, 75, 100],
                        labels=['High Risk', 'Medium Risk', 'Low Risk']
                    )
                    st.success("✅ Risk categories created!")
                    
                    # Show distribution
                    risk_counts = df['Risk_Level'].value_counts()
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.bar(risk_counts.index, risk_counts.values, 
                                  color=['#FF5252', '#FFA726', '#66BB6A'])
                    ax.set_ylabel('Count')
                    ax.set_title('Distribution of Risk Levels')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
        
        # Step 3: Feature selection
        with st.expander("3️⃣ Feature Selection"):
            all_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in all_columns:
                all_columns.remove(target_col)
            
            selected_features = st.multiselect(
                "Select features for analysis:",
                all_columns,
                default=all_columns[:min(10, len(all_columns))]
            )
            
            if selected_features:
                st.write(f"**Selected {len(selected_features)} features:**")
                st.write(", ".join(selected_features))
        
        # Step 4: Process and store data
        with st.expander("4️⃣ Final Processing"):
            if st.button("🚀 Complete Processing", type="primary"):
                # Store processed data
                st.session_state.processed_data = df
                
                # Create feature matrix
                features_to_keep = selected_features + [target_col, 'Risk_Level'] if 'Risk_Level' in df.columns else selected_features + [target_col]
                processed_df = df[features_to_keep].copy()
                
                # Calculate correlations with life expectancy
                if target_col in processed_df.columns:
                    correlations = processed_df.corr()[target_col].sort_values(ascending=False)
                    
                    # Display correlation results
                    st.subheader("📊 Correlation with Life Expectancy")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    correlations.drop(target_col).plot(kind='bar', ax=ax, color='steelblue')
                    ax.set_title(f'Feature Correlations with {target_col}')
                    ax.set_ylabel('Correlation Coefficient')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Display correlation table
                    st.dataframe(
                        correlations.to_frame('Correlation').style.background_gradient(
                            cmap='RdYlGn', vmin=-1, vmax=1
                        ),
                        use_container_width=True
                    )
                
                st.success("✅ Data processing completed successfully!")
                st.balloons()

# Page 3: Risk Analysis
elif page == "🎯 Risk Analysis":
    st.markdown('<div class="section-header"><h2>🎯 Life Expectancy Risk Analysis</h2></div>', unsafe_allow_html=True)
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please process your data first on the Data Processing page.")
    else:
        df = st.session_state.processed_data
        
        # Display risk distribution
        if 'Risk_Level' in df.columns:
            st.subheader("📊 Risk Level Distribution")
            
            col1, col2, col3 = st.columns(3)
            
            risk_counts = df['Risk_Level'].value_counts()
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("High Risk", f"{risk_counts.get('High Risk', 0)}")
                st.markdown('<p class="risk-high">🚨 Immediate attention needed</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Medium Risk", f"{risk_counts.get('Medium Risk', 0)}")
                st.markdown('<p class="risk-medium">⚠️ Monitor closely</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Low Risk", f"{risk_counts.get('Low Risk', 0)}")
                st.markdown('<p class="risk-low">✅ Good health status</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Interactive analysis
            st.subheader("🔍 Analyze Specific Cases")
            
            col_analysis1, col_analysis2 = st.columns(2)
            
            with col_analysis1:
                # Filter by risk level
                selected_risk = st.selectbox(
                    "Filter by Risk Level:",
                    ['All', 'High Risk', 'Medium Risk', 'Low Risk']
                )
                
                if selected_risk != 'All':
                    filtered_df = df[df['Risk_Level'] == selected_risk]
                else:
                    filtered_df = df
                
                st.write(f"**Showing {len(filtered_df)} records**")
                st.dataframe(filtered_df.head(10), use_container_width=True)
            
            with col_analysis2:
                # Find high-risk factors
                st.write("**Top Factors for High Risk:**")
                
                if 'Life_Expectancy' in df.columns:
                    high_risk_data = df[df['Risk_Level'] == 'High Risk']
                    
                    if len(high_risk_data) > 0:
                        # Calculate mean values for high risk group
                        numeric_cols = high_risk_data.select_dtypes(include=[np.number]).columns
                        avg_values = high_risk_data[numeric_cols].mean().sort_values(ascending=False)
                        
                        # Display top 5 factors
                        top_factors = avg_values.head(5)
                        for factor, value in top_factors.items():
                            if factor != 'Life_Expectancy':
                                progress = min(value / top_factors.max() * 100, 100)
                                st.write(f"**{factor}**: {value:.2f}")
                                st.progress(progress/100)
        
            # Risk prediction for new data
            st.subheader("🎯 Predict Risk for New Entry")
            
            with st.form("risk_prediction_form"):
                col_pred1, col_pred2, col_pred3 = st.columns(3)
                
                with col_pred1:
                    adult_mortality = st.slider("Adult Mortality", 0, 500, 200)
                    alcohol = st.slider("Alcohol Consumption", 0.0, 20.0, 5.0)
                
                with col_pred2:
                    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
                    hiv_aids = st.slider("HIV/AIDS Prevalence", 0.0, 10.0, 0.5)
                
                with col_pred3:
                    gdp = st.slider("GDP per capita", 0, 50000, 10000)
                    schooling = st.slider("Schooling Years", 0, 20, 12)
                
                predict_button = st.form_submit_button("🔮 Predict Risk Level")
                
                if predict_button:
                    # Simple rule-based prediction (replace with your actual model)
                    life_expectancy_score = (
                        (100 - adult_mortality/5) * 0.3 +
                        (20 - alcohol) * 0.1 +
                        bmi * 0.1 +
                        (10 - hiv_aids*2) * 0.2 +
                        (gdp/1000) * 0.1 +
                        schooling * 0.2
                    )
                    
                    # Categorize based on score
                    if life_expectancy_score < 60:
                        predicted_risk = "High Risk"
                        risk_color = "risk-high"
                    elif life_expectancy_score < 75:
                        predicted_risk = "Medium Risk"
                        risk_color = "risk-medium"
                    else:
                        predicted_risk = "Low Risk"
                        risk_color = "risk-low"
                    
                    st.markdown(f'<h3 class="{risk_color}">🎯 Predicted Risk Level: {predicted_risk}</h3>', unsafe_allow_html=True)
                    st.metric("Estimated Life Expectancy Score", f"{life_expectancy_score:.1f}")

# Page 4: Visualizations
elif page == "📈 Visualizations":
    st.markdown('<div class="section-header"><h2>📈 Data Visualizations</h2></div>', unsafe_allow_html=True)
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please process your data first to generate visualizations.")
    else:
        df = st.session_state.processed_data
        
        # Visualization options
        viz_type = st.selectbox(
            "Select Visualization Type:",
            ["Risk Distribution", "Life Expectancy Trends", "Feature Relationships", "Geographic Patterns"]
        )
        
        if viz_type == "Risk Distribution":
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if 'Risk_Level' in df.columns:
                    # Pie chart
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Pie chart
                    risk_counts = df['Risk_Level'].value_counts()
                    colors = ['#FF5252', '#FFA726', '#66BB6A']
                    ax1.pie(risk_counts.values, labels=risk_counts.index, 
                           colors=colors, autopct='%1.1f%%', startangle=90)
                    ax1.set_title('Risk Level Distribution')
                    
                    # Bar chart
                    ax2.bar(risk_counts.index, risk_counts.values, color=colors)
                    ax2.set_title('Risk Level Counts')
                    ax2.set_ylabel('Count')
                    plt.xticks(rotation=45)
                    
                    st.pyplot(fig)
            
            with col2:
                st.subheader("📊 Statistics")
                if 'Life_Expectancy' in df.columns:
                    st.metric("Average Life Expectancy", f"{df['Life_Expectancy'].mean():.1f} years")
                    st.metric("Minimum", f"{df['Life_Expectancy'].min():.1f} years")
                    st.metric("Maximum", f"{df['Life_Expectancy'].max():.1f} years")
        
        elif viz_type == "Life Expectancy Trends":
            st.subheader("Life Expectancy Analysis")
            
            # Select column to compare with life expectancy
            if 'Life_Expectancy' in df.columns:
                compare_col = st.selectbox(
                    "Compare Life Expectancy with:",
                    [col for col in df.select_dtypes(include=[np.number]).columns if col != 'Life_Expectancy']
                )
                
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(df[compare_col], df['Life_Expectancy'], 
                                    c=df['Life_Expectancy'], cmap='RdYlGn', alpha=0.6)
                ax.set_xlabel(compare_col)
                ax.set_ylabel('Life Expectancy')
                ax.set_title(f'Life Expectancy vs {compare_col}')
                plt.colorbar(scatter, label='Life Expectancy')
                st.pyplot(fig)
        
        elif viz_type == "Feature Relationships":
            st.subheader("Feature Correlation Matrix")
            
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                correlation_matrix = numeric_df.corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                           center=0, ax=ax, fmt='.2f')
                ax.set_title('Feature Correlation Matrix')
                st.pyplot(fig)
        
        elif viz_type == "Geographic Patterns":
            st.subheader("Country/Region Analysis")
            
            if 'Country' in df.columns and 'Life_Expectancy' in df.columns:
                # Show top and bottom countries
                col_top, col_bottom = st.columns(2)
                
                with col_top:
                    top_countries = df.nlargest(10, 'Life_Expectancy')[['Country', 'Life_Expectancy']]
                    st.write("**🏆 Top 10 Countries (Highest Life Expectancy)**")
                    st.dataframe(top_countries, use_container_width=True)
                
                with col_bottom:
                    bottom_countries = df.nsmallest(10, 'Life_Expectancy')[['Country', 'Life_Expectancy']]
                    st.write("**⚠️ Bottom 10 Countries (Lowest Life Expectancy)**")
                    st.dataframe(bottom_countries, use_container_width=True)
                
                # Bar chart comparison
                fig, ax = plt.subplots(figsize=(12, 6))
                combined = pd.concat([top_countries, bottom_countries])
                colors = ['green'] * 10 + ['red'] * 10
                ax.bar(combined['Country'], combined['Life_Expectancy'], color=colors)
                ax.set_ylabel('Life Expectancy')
                ax.set_title('Top vs Bottom Countries by Life Expectancy')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Life Expectancy Risk Analyzer v1.0 | Made with ❤️ using Streamlit</p>
        <p>For educational and research purposes</p>
    </div>
    """,
    unsafe_allow_html=True
)