import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import pickle
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# =========================================================================
# PAGE CONFIGURATION
# =========================================================================
st.set_page_config(
    page_title="Pizza Delivery Time Prediction",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff6b35;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin: 1.5rem 0;
        border-bottom: 2px solid #ff6b35;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ff6b35;
        margin: 0.8rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .feature-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #2c3e50;
        font-weight: 500;
    }
    .performance-excellent {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #00d4aa;
    }
    .performance-good {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff8a80;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff6b35;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# LOAD MODELS AND ARTIFACTS
# =========================================================================
@st.cache_resource
def load_model_artifacts():
    """Load trained model and related artifacts"""
    try:
        # Load model
        model = joblib.load('best_model.pkl')
        
        # Load scaler
        scaler = joblib.load('scaler.pkl')
        
        # Load metadata
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Load correlations
        with open('feature_correlations.pkl', 'rb') as f:
            correlations = pickle.load(f)
        
        # Load dataset stats
        with open('dataset_stats.pkl', 'rb') as f:
            dataset_stats = pickle.load(f)
        
        # Load model comparison results
        results_df = pd.read_csv('model_comparison_results.csv')
        
        return model, scaler, metadata, correlations, dataset_stats, results_df
    
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.info("Please run the training script first to generate the required model files.")
        st.stop()

@st.cache_data
def load_visualization_images():
    """Load saved visualization images"""
    images = {}
    try:
        images['model_comparison'] = Image.open('model_comparison.png')
        images['correlation_heatmap'] = Image.open('correlation_heatmap.png')
        images['target_analysis'] = Image.open('target_analysis.png')
    except Exception as e:
        st.warning(f"Some visualization images could not be loaded: {e}")
    
    return images

# =========================================================================
# HELPER FUNCTIONS
# =========================================================================
def make_prediction(model, scaler, features, input_data, is_scaled):
    """Make prediction with given model"""
    # Debug: Print untuk troubleshooting
    print(f"Features expected: {len(features)} -> {features}")
    print(f"Input data received: {len(input_data)} -> {input_data}")
    
    # Validasi input
    if len(input_data) != len(features):
        st.error(f"❌ Input mismatch: Expected {len(features)} features, got {len(input_data)}")
        st.write("Expected features:", features)
        st.write("Input data:", input_data)
        return None
    
    # Create DataFrame
    try:
        input_df = pd.DataFrame([input_data], columns=features)
        st.write("✅ Input DataFrame created successfully:")
        st.dataframe(input_df)
        
        if is_scaled:
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            st.write("✅ Used scaled features for prediction")
        else:
            prediction = model.predict(input_df)[0]
            st.write("✅ Used unscaled features for prediction")
        
        return prediction
    
    except Exception as e:
        st.error(f"❌ Error in prediction: {str(e)}")
        st.write("Debug info:")
        st.write(f"Features: {features}")
        st.write(f"Input data: {input_data}")
        st.write(f"Data types: {[type(x) for x in input_data]}")
        return None

def get_delivery_category(prediction):
    """Categorize delivery time"""
    if prediction <= 20:
        return "🚀 Fast Delivery", "success"
    elif prediction <= 30:
        return "⏱️ Normal Delivery", "info"
    else:
        return "🐌 Slow Delivery", "warning"

def get_feature_ranges():
    """Get realistic ranges for input features based on original categories"""
    return {
        'Pizza Type': {
            'options': list(range(1, 12)),
            'labels': {
                1: 'Vegan',
                2: 'Non-Vegan', 
                3: 'Cheese Burst',
                4: 'Gluten-Free',
                5: 'Stuffed Crust',
                6: 'Thin Crust',
                7: 'Deep Fish',
                8: 'Thai Chicken',
                9: 'Sicilian',
                10: 'BBQ Chicken',
                11: 'Margarita'
            }
        },
        'Distance (km)': {
            'min': 2.0,
            'max': 6.0,
            'step': 0.5,
            'default': 4.0
        },
        'Is Weekend': [0, 1],
        'Topping Density': {
            'min': 0.3,
            'max': 1.3,
            'step': 0.05,
            'default': 0.8
        },
        'Order Month': list(range(1, 13)),
        'Pizza Complexity': {
            'min': 1,
            'max': 20,
            'step': 1,
            'default': 10
        },
        'Traffic Impact': {
            'min': 1.0,
            'max': 10.0,
            'step': 0.5,
            'default': 5.0
        },
        'Order Hour': list(range(0, 24))
    }

# =========================================================================
# MAIN APP
# =========================================================================
def main():
    # Load artifacts
    model, scaler, metadata, correlations, dataset_stats, results_df = load_model_artifacts()
    images = load_visualization_images()
    
    # Debug section (dapat diaktifkan jika perlu)
    if st.sidebar.checkbox("🔧 Show Debug Info"):
        st.sidebar.markdown("### 🔍 Debug Information")
        st.sidebar.write(f"**Features ({len(metadata['features'])}):**")
        for i, feature in enumerate(metadata['features']):
            st.sidebar.write(f"{i+1}. {feature}")
        st.sidebar.write(f"**Scaled Model:** {metadata['scaled_data']}")
        st.sidebar.write(f"**Model Type:** {metadata['model_name']}")
    
    # Header
    st.markdown('<div class="main-header">🍕 Pizza Delivery Time Prediction</div>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Delivery Time Estimation System")
    
    # Key metrics in header
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏆 Best Model", metadata['model_name'])
    with col2:
        st.metric("📊 R² Score", f"{metadata['performance']['r2_score']:.4f}")
    with col3:
        st.metric("🎯 RMSE", f"{metadata['performance']['rmse']:.2f} min")
    with col4:
        st.metric("📏 MAE", f"{metadata['performance']['mae']:.2f} min")
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Make Prediction", "📊 Model Performance", "📈 Data Analysis", "ℹ️ About"])
    
    # =====================================================================
    # TAB 1: MAKE PREDICTION
    # =====================================================================
    with tab1:
        st.markdown('<div class="sub-header">🎯 Delivery Time Prediction</div>', unsafe_allow_html=True)
        
        # Model info
        st.markdown(f"""
        <div class="feature-box">
            <strong>Active Model:</strong> {metadata['model_name']} 
            {'(Hyperparameter Tuned)' if metadata.get('tuned', False) else '(Default Parameters)'}
        </div>
        """, unsafe_allow_html=True)
        
        # Input form - HANYA 6 FEATURES YANG DIGUNAKAN MODEL
        st.markdown("#### 📝 Enter Order Details")
        
        # Display which features are used
        st.info(f"ℹ️ This model uses the top 6 most correlated features: {', '.join(metadata['features'])}")
        
        # Get feature ranges
        feature_ranges = get_feature_ranges()
        
        # Dynamic input form berdasarkan features yang ada di model
        input_values = {}
        
        # Create input columns
        col1, col2 = st.columns(2)
        
        # Split features into two columns
        features_col1 = metadata['features'][:len(metadata['features'])//2]
        features_col2 = metadata['features'][len(metadata['features'])//2:]
        
        with col1:
            for feature in features_col1:
                if feature == 'Pizza Type':
                    pizza_types = feature_ranges['Pizza Type']
                    selected_type = st.selectbox(
                        "🍕 Pizza Type",
                        options=pizza_types['options'],
                        index=3,  # Default to Gluten-Free
                        format_func=lambda x: f"{x}. {pizza_types['labels'][x]}",
                        help="Select the type of pizza"
                    )
                    input_values[feature] = selected_type
                    
                elif feature == 'Distance (km)':
                    distance_config = feature_ranges['Distance (km)']
                    input_values[feature] = st.slider(
                        "📍 Distance (km)",
                        min_value=distance_config['min'],
                        max_value=distance_config['max'],
                        value=distance_config['default'],
                        step=distance_config['step'],
                        help="Distance from restaurant to delivery location (2-6 km)"
                    )
                    
                elif feature == 'Is Weekend':
                    input_values[feature] = st.selectbox(
                        "📅 Is Weekend?", 
                        feature_ranges['Is Weekend'], 
                        format_func=lambda x: "🟢 Yes" if x else "🔴 No",
                        help="Weekend orders may take longer"
                    )
                    
                elif feature == 'Topping Density':
                    topping_config = feature_ranges['Topping Density']
                    input_values[feature] = st.slider(
                        "🧀 Topping Density",
                        min_value=topping_config['min'],
                        max_value=topping_config['max'],
                        value=topping_config['default'],
                        step=topping_config['step'],
                        help="Density of toppings on pizza (0.3-1.3 scale)"
                    )
                    
                elif feature == 'Order Month':
                    input_values[feature] = st.selectbox(
                        "📆 Order Month",
                        options=feature_ranges['Order Month'],
                        index=5,  # Default to June
                        format_func=lambda x: f"{x} - {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1]}",
                        help="Month of the year"
                    )
                    
                elif feature == 'Pizza Complexity':
                    complexity_config = feature_ranges['Pizza Complexity']
                    input_values[feature] = st.slider(
                        "🎛️ Pizza Complexity",
                        min_value=complexity_config['min'],
                        max_value=complexity_config['max'],
                        value=complexity_config['default'],
                        step=complexity_config['step'],
                        help="How complex the pizza is to prepare (1-20 scale)"
                    )
                    
                elif feature == 'Traffic Impact':
                    traffic_config = feature_ranges['Traffic Impact']
                    input_values[feature] = st.slider(
                        "🚦 Traffic Impact",
                        min_value=traffic_config['min'],
                        max_value=traffic_config['max'],
                        value=traffic_config['default'],
                        step=traffic_config['step'],
                        help="Current traffic conditions (1-10 scale)"
                    )
                    
                elif feature == 'Order Hour':
                    input_values[feature] = st.selectbox(
                        "🕐 Order Hour",
                        options=feature_ranges['Order Hour'],
                        index=12,  # Default to 12:00
                        format_func=lambda x: f"{x:02d}:00 ({['Midnight','Night','Night','Night','Night','Night','Morning','Morning','Morning','Morning','Morning','Morning','Noon','Afternoon','Afternoon','Afternoon','Afternoon','Afternoon','Evening','Evening','Evening','Evening','Evening','Night'][x]})",
                        help="Hour of the day (24-hour format)"
                    )
        
        with col2:
            for feature in features_col2:
                if feature == 'Pizza Type':
                    pizza_types = feature_ranges['Pizza Type']
                    selected_type = st.selectbox(
                        "🍕 Pizza Type",
                        options=pizza_types['options'],
                        index=3,  # Default to Gluten-Free
                        format_func=lambda x: f"{x}. {pizza_types['labels'][x]}",
                        help="Select the type of pizza",
                        key="pizza_type_col2"
                    )
                    input_values[feature] = selected_type
                    
                elif feature == 'Distance (km)':
                    distance_config = feature_ranges['Distance (km)']
                    input_values[feature] = st.slider(
                        "📍 Distance (km)",
                        min_value=distance_config['min'],
                        max_value=distance_config['max'],
                        value=distance_config['default'],
                        step=distance_config['step'],
                        help="Distance from restaurant to delivery location (2-6 km)",
                        key="distance_col2"
                    )
                    
                elif feature == 'Is Weekend':
                    input_values[feature] = st.selectbox(
                        "📅 Is Weekend?", 
                        feature_ranges['Is Weekend'], 
                        format_func=lambda x: "🟢 Yes" if x else "🔴 No",
                        help="Weekend orders may take longer",
                        key="weekend_col2"
                    )
                    
                elif feature == 'Topping Density':
                    topping_config = feature_ranges['Topping Density']
                    input_values[feature] = st.slider(
                        "🧀 Topping Density",
                        min_value=topping_config['min'],
                        max_value=topping_config['max'],
                        value=topping_config['default'],
                        step=topping_config['step'],
                        help="Density of toppings on pizza (0.3-1.3 scale)",
                        key="topping_col2"
                    )
                    
                elif feature == 'Order Month':
                    input_values[feature] = st.selectbox(
                        "📆 Order Month",
                        options=feature_ranges['Order Month'],
                        index=5,  # Default to June
                        format_func=lambda x: f"{x} - {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1]}",
                        help="Month of the year",
                        key="month_col2"
                    )
                    
                elif feature == 'Pizza Complexity':
                    complexity_config = feature_ranges['Pizza Complexity']
                    input_values[feature] = st.slider(
                        "🎛️ Pizza Complexity",
                        min_value=complexity_config['min'],
                        max_value=complexity_config['max'],
                        value=complexity_config['default'],
                        step=complexity_config['step'],
                        help="How complex the pizza is to prepare (1-20 scale)",
                        key="complexity_col2"
                    )
                    
                elif feature == 'Traffic Impact':
                    traffic_config = feature_ranges['Traffic Impact']
                    input_values[feature] = st.slider(
                        "🚦 Traffic Impact",
                        min_value=traffic_config['min'],
                        max_value=traffic_config['max'],
                        value=traffic_config['default'],
                        step=traffic_config['step'],
                        help="Current traffic conditions (1-10 scale)",
                        key="traffic_col2"
                    )
                    
                elif feature == 'Order Hour':
                    input_values[feature] = st.selectbox(
                        "🕐 Order Hour",
                        options=feature_ranges['Order Hour'],
                        index=12,  # Default to 12:00
                        format_func=lambda x: f"{x:02d}:00 ({['Midnight','Night','Night','Night','Night','Night','Morning','Morning','Morning','Morning','Morning','Morning','Noon','Afternoon','Afternoon','Afternoon','Afternoon','Afternoon','Evening','Evening','Evening','Evening','Evening','Night'][x]})",
                        help="Hour of the day (24-hour format)",
                        key="hour_col2"
                    )
        
        # Prediction button and results
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 Predict Delivery Time", type="primary", use_container_width=True):
                # Prepare input data - SESUAI URUTAN FEATURES MODEL
                input_data = [input_values[feature] for feature in metadata['features']]
                
                # Debug: Show features and input mapping
                st.write("🔍 **Debug Information:**")
                st.write(f"**Model expects {len(metadata['features'])} features:**")
                feature_mapping = list(zip(metadata['features'], input_data))
                for feature, value in feature_mapping:
                    st.write(f"• {feature}: {value}")
                
                # Make prediction
                prediction = make_prediction(
                    model, scaler, metadata['features'], input_data, metadata['scaled_data']
                )
                
                # Only proceed if prediction is successful
                if prediction is not None:
                    # Display prediction
                    category, status = get_delivery_category(prediction)
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>🕐 Predicted Delivery Time</h2>
                        <h1>{prediction:.1f} minutes</h1>
                        <h3>{category}</h3>
                        <p>Confidence: {metadata['performance']['r2_score']:.1%} R² Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional insights
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if prediction <= 20:
                            st.success("🚀 Excellent! Fast delivery expected")
                        elif prediction <= 30:
                            st.info("⏱️ Normal delivery timeframe")
                        else:
                            st.warning("🐌 Longer delivery time expected")
                    
                    with col2:
                        # Calculate percentile based on dataset stats
                        if prediction <= dataset_stats['target_mean'] - dataset_stats['target_std']:
                            percentile = "Top 15% (Very Fast)"
                        elif prediction <= dataset_stats['target_mean']:
                            percentile = "Top 50% (Faster than average)"
                        elif prediction <= dataset_stats['target_mean'] + dataset_stats['target_std']:
                            percentile = "Bottom 50% (Slower than average)"
                        else:
                            percentile = "Bottom 15% (Very Slow)"
                        
                        st.info(f"📊 Performance: {percentile}")
                    
                    with col3:
                        diff = prediction - dataset_stats['target_mean']
                        if diff > 0:
                            st.metric("vs Average", f"+{diff:.1f} min", delta=f"{diff:.1f}")
                        else:
                            st.metric("vs Average", f"{diff:.1f} min", delta=f"{diff:.1f}")
                    
                    # Feature impact analysis
                    st.markdown("#### 🔍 Key Factors Analysis")
                    
                    impact_factors = []
                    
                    # Check impact factors berdasarkan features yang ada
                    if 'Distance (km)' in input_values and input_values['Distance (km)'] > 8:
                        impact_factors.append("🔴 High distance increases delivery time")
                    if 'Traffic Impact' in input_values and input_values['Traffic Impact'] > 7:
                        impact_factors.append("🔴 Heavy traffic will delay delivery")
                    if 'Pizza Complexity' in input_values and input_values['Pizza Complexity'] > 7:
                        impact_factors.append("🟡 Complex pizza takes longer to prepare")
                    if 'Is Weekend' in input_values and input_values['Is Weekend']:
                        impact_factors.append("🟡 Weekend orders may take slightly longer")
                    if 'Order Hour' in input_values and input_values['Order Hour'] in [11, 12, 13, 18, 19, 20]:
                        impact_factors.append("🟡 Peak hour - higher demand")
                    
                    if not impact_factors:
                        impact_factors.append("🟢 Optimal conditions for fast delivery!")
                    
                    for factor in impact_factors:
                        st.markdown(f"• {factor}")
                else:
                    st.error("❌ Prediction failed. Please check the debug information above.")
    
    # =====================================================================
    # TAB 2: MODEL PERFORMANCE
    # =====================================================================
    with tab2:
        st.markdown('<div class="sub-header">📊 Model Performance Analysis</div>', unsafe_allow_html=True)
        
        # Performance overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="performance-excellent">
                <h4>🏆 Best Performing Model</h4>
                <h3>{metadata['model_name']}</h3>
                <p><strong>R² Score:</strong> {metadata['performance']['r2_score']:.4f}</p>
                <p><strong>RMSE:</strong> {metadata['performance']['rmse']:.2f} minutes</p>
                <p><strong>MAE:</strong> {metadata['performance']['mae']:.2f} minutes</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="performance-good">
                <h4>📈 Model Insights</h4>
                <p><strong>Hyperparameter Tuned:</strong> {'Yes' if metadata.get('tuned', False) else 'No'}</p>
                <p><strong>Uses Scaled Features:</strong> {'Yes' if metadata['scaled_data'] else 'No'}</p>
                <p><strong>Feature Count:</strong> {len(metadata['features'])}</p>
                <p><strong>Prediction Accuracy:</strong> {metadata['performance']['r2_score']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Model comparison table
        st.markdown("#### 📋 All Models Comparison")
        
        # Format the dataframe for better display
        display_df = results_df.copy()
        display_df = display_df.round(4)
        
        # Add ranking
        display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
        
        # Highlight best model
        def highlight_best(row):
            if row['Rank'] == 1:
                return ['background-color: #fffacd'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            display_df.style.apply(highlight_best, axis=1),
            use_container_width=True,
            hide_index=True
        )
        
        # Visualizations
        if 'model_comparison' in images:
            st.markdown("#### 📊 Model Performance Visualization")
            st.image(images['model_comparison'], caption="Model Performance Comparison", use_column_width=True)
        
        # Performance metrics explanation
        with st.expander("📚 Understanding the Metrics"):
            st.markdown("""
            **R² Score (Coefficient of Determination):**
            - Measures how well the model explains the variance in delivery times
            - Range: 0 to 1 (higher is better)
            - Our best model: {:.1%} of delivery time variance explained
            
            **RMSE (Root Mean Square Error):**
            - Average prediction error in minutes
            - Lower values indicate better accuracy
            - Our best model: ±{:.1f} minutes average error
            
            **MAE (Mean Absolute Error):**
            - Average absolute difference between predicted and actual times
            - More interpretable than RMSE
            - Our best model: {:.1f} minutes average absolute error
            """.format(
                metadata['performance']['r2_score'],
                metadata['performance']['rmse'],
                metadata['performance']['mae']
            ))
    
    # =====================================================================
    # TAB 3: DATA ANALYSIS
    # =====================================================================
    with tab3:
        st.markdown('<div class="sub-header">📈 Data Analysis & Insights</div>', unsafe_allow_html=True)
        
        # Dataset overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Total Samples</h4>
                <h3>{dataset_stats['total_samples']:,}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📋 Features Used</h4>
                <h3>{dataset_stats['feature_count']}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>⏱️ Avg Delivery</h4>
                <h3>{dataset_stats['target_mean']:.1f} min</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📏 Std Deviation</h4>
                <h3>{dataset_stats['target_std']:.1f} min</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature importance and correlations
        st.markdown("#### 🔗 Feature Correlations with Delivery Time")
        
        # Create correlation dataframe
        corr_data = []
        for feature in metadata['features']:
            corr_info = correlations[feature]
            corr_data.append({
                'Feature': feature,
                'Correlation': corr_info['correlation'],
                'Absolute Correlation': corr_info['abs_correlation'],
                'Significance': 'High' if corr_info['p_value'] < 0.001 else 'Medium' if corr_info['p_value'] < 0.01 else 'Low'
            })
        
        corr_df = pd.DataFrame(corr_data)
        
        # Display as interactive chart
        fig = px.bar(
            corr_df, 
            x='Absolute Correlation', 
            y='Feature', 
            orientation='h',
            color='Correlation',
            color_continuous_scale='RdBu_r',
            title="Feature Importance (Correlation with Delivery Time)",
            hover_data=['Correlation', 'Significance']
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation details table
        st.dataframe(corr_df.round(4), use_container_width=True, hide_index=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            if 'correlation_heatmap' in images:
                st.markdown("#### 🌡️ Feature Correlation Heatmap")
                st.image(images['correlation_heatmap'], caption="Feature Correlation Matrix", use_column_width=True)
        
        with col2:
            if 'target_analysis' in images:
                st.markdown("#### 📊 Target Variable Analysis")
                st.image(images['target_analysis'], caption="Delivery Time Distribution & Categories", use_column_width=True)
        
        # Business insights
        st.markdown("#### 💡 Key Business Insights")
        
        # Find top 3 most correlated features
        top_features = corr_df.nlargest(3, 'Absolute Correlation')
        
        insights = []
        for _, row in top_features.iterrows():
            feature = row['Feature']
            correlation = row['Correlation']
            if correlation > 0:
                insights.append(f"🔴 **{feature}** strongly increases delivery time (correlation: {correlation:.3f})")
            else:
                insights.append(f"🟢 **{feature}** helps reduce delivery time (correlation: {correlation:.3f})")
        
        for insight in insights:
            st.markdown(insight)
        
        # Additional insights
        st.markdown("""
        **🎯 Optimization Recommendations:**
        - Focus on optimizing the top correlated factors
        - Monitor delivery performance during peak hours
        - Consider dynamic pricing based on distance and traffic
        - Implement route optimization for longer distances
        """)
    
    # =====================================================================
    # TAB 4: ABOUT
    # =====================================================================
    with tab4:
        st.markdown('<div class="sub-header">ℹ️ About This Application</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🍕 Pizza Delivery Time Prediction System
        
        This application uses advanced machine learning algorithms to predict pizza delivery times 
        based on various factors such as distance, traffic conditions, pizza complexity, and more.
        
        #### 🤖 How It Works
        
        1. **Data Collection**: Historical delivery data with 8 key features
        2. **Feature Selection**: Top 6 most correlated features identified using Spearman correlation
        3. **Model Training**: Multiple regression algorithms tested and compared
        4. **Model Selection**: Best performing model selected based on R² score
        5. **Hyperparameter Tuning**: Optimal parameters found using grid search
        6. **Deployment**: Model deployed as interactive web application
        
        #### 📊 Technical Details
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **🔧 Model Information:**
            - **Algorithm**: {metadata['model_name']}
            - **Features**: {len(metadata['features'])} selected features
            - **Scaling**: {'Applied' if metadata['scaled_data'] else 'Not required'}
            - **Tuning**: {'Hyperparameter optimized' if metadata.get('tuned', False) else 'Default parameters'}
            
            **📈 Performance Metrics:**
            - **R² Score**: {metadata['performance']['r2_score']:.4f}
            - **RMSE**: {metadata['performance']['rmse']:.2f} minutes
            - **MAE**: {metadata['performance']['mae']:.2f} minutes
            """)
        
        with col2:
            st.markdown(f"""
            **📋 Selected Features:**
            """)
            for i, feature in enumerate(metadata['features'], 1):
                correlation = correlations[feature]['correlation']
                st.markdown(f"{i}. **{feature}** (ρ = {correlation:.3f})")
        
        st.markdown("""
        #### 🎯 Use Cases
        
        - **Restaurant Operations**: Provide accurate delivery estimates to customers
        - **Resource Planning**: Optimize delivery staff allocation
        - **Customer Service**: Set realistic expectations and improve satisfaction
        - **Business Intelligence**: Identify factors that impact delivery performance
        
        #### 🔮 Future Enhancements
        
        - Real-time traffic integration
        - Weather condition factors
        - Historical customer order patterns
        - Multi-restaurant delivery optimization
        - Mobile app integration
        
        #### 📞 Support
        
        For technical support or feature requests, please contact the development team.
        """)
        
        # Model file information
        with st.expander("📁 Model Files Information"):
            st.markdown("""
            The following files are used by this application:
            
            - `best_model.pkl` - Trained machine learning model
            - `scaler.pkl` - Feature scaling transformer
            - `model_metadata.pkl` - Model configuration and performance data
            - `feature_correlations.pkl` - Feature correlation analysis results
            - `dataset_stats.pkl` - Dataset statistics and summary
            - `model_comparison_results.csv` - All models performance comparison
            - Visualization images (PNG files) for charts and analysis
            """)

if __name__ == "__main__":
    main()
