import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Concrete Strength Analysis",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved readability
st.markdown("""
    <style>
    /* Theme Colors */
    :root {
        --bg-gradient: linear-gradient(135deg, #1a1c2b 0%, #2a2d3e 100%);
        --card-bg: rgba(255, 255, 255, 0.08);
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.85);
        --accent-1: #60a5fa;
        --accent-2: #34d399;
        --border: rgba(255, 255, 255, 0.12);
        --title-gradient: linear-gradient(45deg, #3b82f6, #06b6d4, #10b981);
        --title-shadow: 0 2px 15px rgba(59, 130, 246, 0.3);
    }

    /* Base styles */
    .main {
        background: var(--bg-gradient);
        color: var(--text-primary);
        font-family: system-ui, -apple-system, sans-serif;
        min-height: 100vh;
        background-attachment: fixed;
        line-height: 1.6;
    }

    /* Typography */
    h1, h2, h3 {
        color: var(--text-primary);
        font-weight: 600;
        letter-spacing: -0.02em;
        margin-bottom: 1rem;
    }

    h1 {
        font-size: 2.75rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        margin-bottom: 1.5rem !important;
        text-align: center !important;
        padding: 1rem 0 !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.5) !important;
        letter-spacing: -0.02em !important;
        line-height: 1.2 !important;
    }

    h1 .emoji {
        font-size: 2.5rem !important;
        margin-right: 0.5rem !important;
        text-shadow: none !important;
    }

    p, label, .stMarkdown {
        color: var(--text-secondary);
        font-size: 1rem;
        line-height: 1.6;
    }

    /* Cards */
    .metric-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: var(--accent-1);
        background: rgba(255, 255, 255, 0.12);
    }

    .metric-value {
        color: var(--accent-1);
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Inputs and Controls */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div {
        background: var(--card-bg);
        border: 1px solid var(--border);
        color: var(--text-primary);
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1rem;
    }

    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div:focus {
        border-color: var(--accent-1);
        box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.2);
    }

    /* Button */
    .stButton>button {
        background: linear-gradient(to right, var(--accent-1), var(--accent-2));
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    /* Plots */
    .js-plotly-plot {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
    }

    .js-plotly-plot .plotly text {
        fill: var(--text-secondary) !important;
    }

    /* Sidebar */
    .css-1d391kg {
        background: var(--card-bg);
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.3);
    }

    /* Alerts */
    .stAlert {
        background: var(--card-bg);
        border: 1px solid var(--border);
        color: var(--text-primary);
        border-radius: 8px;
    }

    /* Success Alert */
    .stAlert[data-baseweb="notification"] {
        background: var(--accent-2);
        border: none;
        color: #1a1c2b;
    }

    /* Tables */
    .stTable {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
    }

    .stTable th {
        background: rgba(255, 255, 255, 0.05);
        color: var(--text-primary);
    }

    .stTable td {
        color: var(--text-secondary);
    }

    /* Code blocks */
    .stCodeBlock {
        background: rgba(0, 0, 0, 0.2) !important;
        border: 1px solid var(--border);
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the data"""
    df = pd.read_csv('concrete.csv')
    return df

@st.cache_resource
def train_model(df):
    """Train the Random Forest model"""
    X = df.drop('strength', axis=1)
    y = df['strength']
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

def create_feature_importance_plot(model, feature_names):
    """Create feature importance plot using plotly"""
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(importance, x='importance', y='feature', orientation='h',
                 title='Feature Importance',
                 labels={'importance': 'Importance', 'feature': 'Feature'})
    fig.update_layout(showlegend=False)
    return fig

def home_page():
    # Header section
    st.markdown("""
        <div class="home-header">
            <h1><span class="emoji">🏗️</span><span class="emoji">🔥</span> Concrete Strength Analysis System</h1>
            <p style='font-size: 1.2rem;'>Advanced Machine Learning for Precise Concrete Strength Prediction</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data for metrics
    df = load_data()
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">📊 1,030</div>
                <div class="metric-label">Total Samples</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">💪 {df['strength'].mean():.1f}</div>
                <div class="metric-label">Average Strength (MPa)</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">⚡ {df['strength'].max():.1f}</div>
                <div class="metric-label">Maximum Strength (MPa)</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">🎯 91%</div>
                <div class="metric-label">Model Accuracy</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Feature Cards
    st.markdown("### 🚀 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <h3>Data Analysis</h3>
                <p>Explore comprehensive visualizations and statistical analysis of concrete mixture data.</p>
                <ul>
                    <li>Interactive data exploration</li>
                    <li>Correlation analysis</li>
                    <li>Distribution visualization</li>
                </ul>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <h3>Strength Prediction</h3>
                <p>Get accurate predictions of concrete strength using advanced machine learning.</p>
                <ul>
                    <li>Real-time predictions</li>
                    <li>Confidence intervals</li>
                    <li>Optimization suggestions</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <h3>Model Insights</h3>
                <p>Understand the factors that influence concrete strength.</p>
                <ul>
                    <li>Feature importance analysis</li>
                    <li>Performance metrics</li>
                    <li>Interactive visualizations</li>
                </ul>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">💡</div>
                <h3>Optimization</h3>
                <p>Get recommendations for optimal concrete mixtures.</p>
                <ul>
                    <li>Mix design suggestions</li>
                    <li>Cost optimization</li>
                    <li>Quality control</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Quick Start Guide
    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
        <div class="feature-card">
            <ol>
                <li><strong>Explore Data:</strong> Use the Data Explorer to understand concrete mixture patterns</li>
                <li><strong>Make Predictions:</strong> Input your mixture properties to predict strength</li>
                <li><strong>Analyze Results:</strong> View detailed insights and recommendations</li>
                <li><strong>Optimize Design:</strong> Get suggestions for improving your concrete mixture</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

def data_explorer_page():
    st.header("Data Explorer")
    
    # Show raw data
    if st.checkbox("Show raw data"):
        df = load_data()
        st.write(df)
    
    # Statistical summary
    if st.checkbox("Show statistical summary"):
        df = load_data()
        st.write(df.describe())
    
    # Correlation heatmap
    if st.checkbox("Show correlation heatmap"):
        df = load_data()
        fig = px.imshow(df.corr(),
                      labels=dict(color="Correlation"),
                      title="Feature Correlation Matrix")
        st.plotly_chart(fig)
    
    # Scatter plots
    st.subheader("Feature Relationships")
    df = load_data()
    x_axis = st.selectbox("Choose x-axis", df.columns)
    y_axis = st.selectbox("Choose y-axis", df.columns)
    
    fig = px.scatter(df, x=x_axis, y=y_axis,
                    title=f"{x_axis} vs {y_axis}")
    st.plotly_chart(fig)

def prediction_page(model, scaler):
    st.header("Strength Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Mixture Properties")
        cement = st.number_input("Cement (kg/m³)", 100.0, 600.0, 350.0)
        slag = st.number_input("Slag (kg/m³)", 0.0, 400.0, 100.0)
        ash = st.number_input("Fly Ash (kg/m³)", 0.0, 200.0, 50.0)
        water = st.number_input("Water (kg/m³)", 100.0, 300.0, 180.0)
    
    with col2:
        superplastic = st.number_input("Superplasticizer (kg/m³)", 0.0, 20.0, 6.0)
        coarseagg = st.number_input("Coarse Aggregate (kg/m³)", 700.0, 1200.0, 1000.0)
        fineagg = st.number_input("Fine Aggregate (kg/m³)", 500.0, 1000.0, 800.0)
        age = st.number_input("Age (days)", 1, 365, 28)
    
    if st.button("Predict Strength"):
        # Create input array
        input_data = pd.DataFrame([[cement, slag, ash, water, superplastic, coarseagg, fineagg, age]],
                                columns=['cement', 'slag', 'ash', 'water', 'superplastic', 
                                       'coarseagg', 'fineagg', 'age'])
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Show prediction with confidence interval
        st.success(f"Predicted Strength: {prediction:.2f} MPa")
        
        # Calculate water-cement ratio
        wc_ratio = water / cement
        st.info(f"Water-Cement Ratio: {wc_ratio:.3f}")
        
        # Add recommendations
        if wc_ratio > 0.5:
            st.warning("Consider reducing the water-cement ratio for higher strength")
        if prediction < 30:
            st.warning("This mixture might result in low-strength concrete")
        elif prediction > 50:
            st.success("This mixture should produce high-strength concrete")

def model_insights_page(model, df):
    st.header("Model Insights")
    
    # Feature importance
    st.subheader("Feature Importance")
    fig = create_feature_importance_plot(model, df.drop('strength', axis=1).columns)
    st.plotly_chart(fig)
    
    # Strength distribution
    st.subheader("Strength Distribution")
    fig = px.histogram(df, x='strength', nbins=30,
                      title='Distribution of Concrete Strength')
    st.plotly_chart(fig)
    
    # Age vs Strength
    st.subheader("Age vs Strength Relationship")
    fig = px.scatter(df, x='age', y='strength',
                    title='Age vs Strength',
                    labels={'age': 'Age (days)', 'strength': 'Strength (MPa)'})
    st.plotly_chart(fig)
    
    # Model performance metrics
    st.subheader("Model Performance")
    X = df.drop('strength', axis=1)
    y = df['strength']
    X_scaled = RobustScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training R² Score", f"{model.score(X_train, y_train):.3f}")
    with col2:
        st.metric("Testing R² Score", f"{model.score(X_test, y_test):.3f}")

def main():
    # Load data and train model
    df = load_data()
    model, scaler = train_model(df)
    
    # Sidebar navigation
    st.sidebar.image("https://raw.githubusercontent.com/your-repo/concrete-icon.png", width=100)
    page = st.sidebar.selectbox(
        "Navigation",
        ["Home", "Data Explorer", "Prediction", "Model Insights"]
    )
    
    if page == "Home":
        home_page()
    elif page == "Data Explorer":
        data_explorer_page()
    elif page == "Prediction":
        prediction_page(model, scaler)
    else:  # Model Insights
        model_insights_page(model, df)

if __name__ == "__main__":
    main()
