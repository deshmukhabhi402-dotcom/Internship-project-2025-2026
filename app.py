# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page setup
st.set_page_config(page_title="Health Risk Level Analysis", layout="wide")
st.title("📊 Health Risk Level Analysis")

st.markdown("""
### Analyzing Health Risk Levels from Life Expectancy Data
**Risk categories based on Life Expectancy:**
- **High Risk**: Life Expectancy < 60 years
- **Medium Risk**: Life Expectancy 60-70 years  
- **Low Risk**: Life Expectancy > 70 years
""")

# Function to create risk levels (same as notebook)
def risk_level(le):
    if pd.isna(le):
        return np.nan
    if le < 60:
        return "High"
    elif le < 70:
        return "Medium"
    else:
        return "Low"

# 1. UPLOAD YOUR DATA
uploaded_file = st.file_uploader("📁 Upload your dataset CSV with Life Expectancy columns", type=['csv'])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Show preview
        with st.expander("👀 Preview your dataset"):
            st.dataframe(df.head(), use_container_width=True)
        
        # 2. FIND LIFE EXPECTANCY COLUMNS
        st.header("🔍 Life Expectancy Columns Found")
        
        # Find all columns with 'life' in the name (case-insensitive)
        life_cols = []
        for col in df.columns:
            col_lower = str(col).lower()
            if 'life' in col_lower:
                life_cols.append(col)
        
        if not life_cols:
            st.error("❌ No Life Expectancy columns found in the dataset")
            st.stop()
        
        # Display found columns
        st.write(f"Found **{len(life_cols)}** Life Expectancy column(s):")
        for i, col in enumerate(life_cols, 1):
            st.write(f"{i}. `{col}`")
        
        # 3. CREATE RISK LEVELS FOR EACH COLUMN
        st.header("📊 Creating Health Risk Levels")
        
        # Create risk levels for each life expectancy column
        risk_data = {}
        
        for col in life_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Create risk levels
                df[f'{col}_Risk'] = df[col].apply(risk_level)
                
                # Get counts
                counts = df[f'{col}_Risk'].value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0)
                risk_data[col] = counts
                
                # Display statistics for this column
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(f"{col} - Low", counts.get('Low', 0))
                with col2:
                    st.metric(f"{col} - Medium", counts.get('Medium', 0))
                with col3:
                    st.metric(f"{col} - High", counts.get('High', 0))
                with col4:
                    total = counts.sum()
                    st.metric(f"{col} - Total", total)
            else:
                st.warning(f"Column '{col}' is not numeric. Skipping.")
        
        # 4. VISUALIZATIONS - PAIRED GRAPHS
        st.header("📈 Health Risk Level Distributions")
        
        # Create two main columns for paired visualizations
        col_left, col_right = st.columns(2)
        
        with col_left:
            # FIGURE 1: Individual Bar Charts (Stacked)
            st.subheader("Individual Risk Level Distributions")
            
            if len(life_cols) == 1:
                # Single column - show detailed view
                col = life_cols[0]
                counts = risk_data[col]
                
                fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
                
                # Colors for risk levels
                colors = {'High': '#FF6B6B', 'Medium': '#FFD93D', 'Low': '#6BCF7F'}
                
                # Top: Bar chart
                bars = ax1.bar(counts.index, counts.values,
                              color=[colors[risk] for risk in counts.index],
                              alpha=0.8, edgecolor='black')
                ax1.set_title(f'Risk Levels for {col}', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Count', fontsize=12)
                ax1.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                            f'{int(height)}', ha='center', fontsize=11, fontweight='bold')
                
                # Bottom: Pie chart
                wedges, texts, autotexts = ax2.pie(counts.values,
                                                  labels=counts.index,
                                                  colors=[colors[risk] for risk in counts.index],
                                                  autopct='%1.1f%%',
                                                  startangle=90)
                ax2.set_title(f'Distribution for {col}', fontsize=14, fontweight='bold')
                
                # Make percentages bold
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                plt.tight_layout()
                st.pyplot(fig1)
                
            elif len(life_cols) == 2:
                # Two columns - show side-by-side comparison
                fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                colors = {'High': '#FF6B6B', 'Medium': '#FFD93D', 'Low': '#6BCF7F'}
                
                for idx, col in enumerate(life_cols[:2]):  # Show first 2 columns
                    counts = risk_data[col]
                    
                    # Bar chart
                    bars = axes[idx, 0].bar(counts.index, counts.values,
                                          color=[colors[risk] for risk in counts.index],
                                          alpha=0.8, edgecolor='black')
                    axes[idx, 0].set_title(f'{col} - Bar Chart', fontsize=11)
                    axes[idx, 0].set_ylabel('Count')
                    axes[idx, 0].grid(True, alpha=0.3, axis='y')
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        axes[idx, 0].text(bar.get_x() + bar.get_width()/2, height + 0.5,
                                         f'{int(height)}', ha='center', fontsize=9)
                    
                    # Pie chart
                    wedges, texts, autotexts = axes[idx, 1].pie(counts.values,
                                                              labels=counts.index,
                                                              colors=[colors[risk] for risk in counts.index],
                                                              autopct='%1.1f%%',
                                                              startangle=90)
                    axes[idx, 1].set_title(f'{col} - Pie Chart', fontsize=11)
                
                plt.tight_layout()
                st.pyplot(fig1)
                
            else:
                # Multiple columns - show grid
                num_cols = min(len(life_cols), 4)  # Show max 4 columns
                fig1, axes = plt.subplots(num_cols, 2, figsize=(12, 4*num_cols))
                
                colors = {'High': '#FF6B6B', 'Medium': '#FFD93D', 'Low': '#6BCF7F'}
                
                for idx, col in enumerate(life_cols[:num_cols]):
                    counts = risk_data[col]
                    
                    # Bar chart
                    if num_cols == 1:
                        ax_bar = axes[0]
                        ax_pie = axes[1]
                    else:
                        ax_bar = axes[idx, 0]
                        ax_pie = axes[idx, 1]
                    
                    bars = ax_bar.bar(counts.index, counts.values,
                                     color=[colors[risk] for risk in counts.index],
                                     alpha=0.8, edgecolor='black')
                    ax_bar.set_title(f'{col}', fontsize=11)
                    ax_bar.set_ylabel('Count')
                    ax_bar.grid(True, alpha=0.3, axis='y')
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        ax_bar.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                                   f'{int(height)}', ha='center', fontsize=9)
                    
                    # Pie chart
                    wedges, texts, autotexts = ax_pie.pie(counts.values,
                                                         colors=[colors[risk] for risk in counts.index],
                                                         autopct='%1.1f%%',
                                                         startangle=90)
                    ax_pie.set_title(f'{col} - Distribution', fontsize=11)
                
                plt.tight_layout()
                st.pyplot(fig1)
        
        with col_right:
            # FIGURE 2: Comparative Analysis
            st.subheader("Comparative Analysis")
            
            if len(life_cols) >= 2:
                # Compare first two columns
                col1, col2 = life_cols[0], life_cols[1]
                counts1 = risk_data[col1]
                counts2 = risk_data[col2]
                
                fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
                
                colors = {'High': '#FF6B6B', 'Medium': '#FFD93D', 'Low': '#6BCF7F'}
                
                # Top: Side-by-side comparison
                x = np.arange(len(counts1))
                width = 0.35
                
                bars1 = ax1.bar(x - width/2, counts1.values, width,
                               label=col1, color=[colors[risk] for risk in counts1.index],
                               alpha=0.8, edgecolor='black')
                
                bars2 = ax1.bar(x + width/2, counts2.values, width,
                               label=col2, color=[colors[risk] for risk in counts2.index],
                               alpha=0.6, edgecolor='black', hatch='//')
                
                ax1.set_xlabel('Risk Level', fontsize=12)
                ax1.set_ylabel('Count', fontsize=12)
                ax1.set_title(f'{col1} vs {col2} Comparison', fontsize=14, fontweight='bold')
                ax1.set_xticks(x)
                ax1.set_xticklabels(['Low', 'Medium', 'High'])
                ax1.legend()
                ax1.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                                f'{int(height)}', ha='center', fontsize=10)
                
                # Bottom: Stacked area comparison
                risk_levels = ['Low', 'Medium', 'High']
                values1 = [counts1.get(risk, 0) for risk in risk_levels]
                values2 = [counts2.get(risk, 0) for risk in risk_levels]
                
                x_pos = np.arange(len(risk_levels))
                ax2.bar(x_pos - 0.2, values1, 0.4, label=col1, alpha=0.7, color='blue')
                ax2.bar(x_pos + 0.2, values2, 0.4, label=col2, alpha=0.7, color='orange')
                
                ax2.set_xlabel('Risk Level', fontsize=12)
                ax2.set_ylabel('Count', fontsize=12)
                ax2.set_title('Risk Level Comparison', fontsize=14, fontweight='bold')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(risk_levels)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig2)
                
                # Show comparison table
                st.write("**Comparison Table:**")
                comparison_df = pd.DataFrame({
                    'Risk Level': risk_levels,
                    f'{col1}_Count': [counts1.get(risk, 0) for risk in risk_levels],
                    f'{col2}_Count': [counts2.get(risk, 0) for risk in risk_levels],
                    'Difference': [counts2.get(risk, 0) - counts1.get(risk, 0) for risk in risk_levels]
                })
                st.dataframe(comparison_df, use_container_width=True)
                
            else:
                # Single column - show additional visualization
                col = life_cols[0]
                counts = risk_data[col]
                
                fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
                
                colors = {'High': '#FF6B6B', 'Medium': '#FFD93D', 'Low': '#6BCF7F'}
                
                # Top: Horizontal bar chart
                bars = ax1.barh(range(len(counts)), counts.values,
                              color=[colors[risk] for risk in counts.index],
                              alpha=0.8, edgecolor='black')
                ax1.set_yticks(range(len(counts)))
                ax1.set_yticklabels(counts.index)
                ax1.set_xlabel('Count', fontsize=12)
                ax1.set_title(f'{col} - Horizontal View', fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3, axis='x')
                
                # Add value labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                            f'{int(width)}', ha='left', va='center', fontsize=10, fontweight='bold')
                
                # Bottom: Donut chart
                wedges, texts, autotexts = ax2.pie(counts.values,
                                                  labels=counts.index,
                                                  colors=[colors[risk] for risk in counts.index],
                                                  autopct='%1.1f%%',
                                                  startangle=90,
                                                  wedgeprops=dict(width=0.3, edgecolor='w'))
                ax2.set_title(f'{col} - Donut Chart', fontsize=14, fontweight='bold')
                
                # Make percentages bold
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                plt.tight_layout()
                st.pyplot(fig2)
                
                # Show statistics
                st.write("**Statistics:**")
                stats_df = pd.DataFrame({
                    'Risk Level': counts.index,
                    'Count': counts.values,
                    'Percentage': (counts.values / counts.sum() * 100).round(1),
                    'Cumulative %': (counts.values.cumsum() / counts.sum() * 100).round(1)
                })
                st.dataframe(stats_df, use_container_width=True)
        
        # 5. CLASSIFICATION REPORT (PLAIN TEXT)
        st.header("📋 Classification Report")
        
        if len(life_cols) >= 2:
            # Generate classification report comparing first two columns
            col1, col2 = life_cols[0], life_cols[1]
            
            # Ensure we have risk columns
            if f'{col1}_Risk' in df.columns and f'{col2}_Risk' in df.columns:
                # Get non-null values
                mask = df[f'{col1}_Risk'].notna() & df[f'{col2}_Risk'].notna()
                y_true = df.loc[mask, f'{col1}_Risk']
                y_pred = df.loc[mask, f'{col2}_Risk']
                
                # Generate classification report
                st.write(f"**Classification Report: {col1} vs {col2}**")
                
                report_dict = classification_report(y_true, y_pred, 
                                                   labels=['Low', 'Medium', 'High'], 
                                                   output_dict=True)
                
                # Display as plain text
                report_text = classification_report(y_true, y_pred, 
                                                   labels=['Low', 'Medium', 'High'])
                
                st.code(report_text, language='text')
                
                # Also show as formatted metrics
                st.write("**Key Metrics:**")
                
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                with metrics_col1:
                    accuracy = report_dict['accuracy']
                    st.metric("Accuracy", f"{accuracy:.2%}")
                
                with metrics_col2:
                    macro_avg_precision = report_dict['macro avg']['precision']
                    st.metric("Macro Avg Precision", f"{macro_avg_precision:.3f}")
                
                with metrics_col3:
                    macro_avg_recall = report_dict['macro avg']['recall']
                    st.metric("Macro Avg Recall", f"{macro_avg_recall:.3f}")
                
                with metrics_col4:
                    macro_avg_f1 = report_dict['macro avg']['f1-score']
                    st.metric("Macro Avg F1-Score", f"{macro_avg_f1:.3f}")
                
                # Show per-class metrics in a table
                st.write("**Per-Class Metrics:**")
                
                class_metrics = []
                for class_name in ['Low', 'Medium', 'High']:
                    if class_name in report_dict:
                        class_metrics.append({
                            'Class': class_name,
                            'Precision': report_dict[class_name]['precision'],
                            'Recall': report_dict[class_name]['recall'],
                            'F1-Score': report_dict[class_name]['f1-score'],
                            'Support': int(report_dict[class_name]['support'])
                        })
                
                class_metrics_df = pd.DataFrame(class_metrics)
                st.dataframe(class_metrics_df.style.format({
                    'Precision': '{:.3f}',
                    'Recall': '{:.3f}',
                    'F1-Score': '{:.3f}'
                }), use_container_width=True)
        
        # 6. SAMPLE DATA WITH RISK LEVELS
        st.header("🔍 Sample Data with Risk Levels")
        
        # Show sample of data with risk levels
        sample_size = 10
        
        # Create display dataframe with risk levels
        display_cols = []
        for col in life_cols[:2]:  # Show first 2 life expectancy columns
            display_cols.append(col)
            if f'{col}_Risk' in df.columns:
                display_cols.append(f'{col}_Risk')
        
        # Add any additional important columns
        other_cols = [col for col in df.columns if col not in display_cols][:3]  # First 3 other columns
        display_cols = other_cols + display_cols