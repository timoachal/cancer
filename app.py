import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Risk Assessment",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF69B4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4A90E2;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #f44336 0%, #da190b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .normal-range {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        margin: 0.2rem 0;
        font-size: 0.9rem;
    }
    
    .abnormal-range {
        background: linear-gradient(135deg, #e17055 0%, #d63031 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        margin: 0.2rem 0;
        font-size: 0.9rem;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_data
def load_model_and_data():
    with open('breast_cancer_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    with open('normal_ranges.pkl', 'rb') as f:
        normal_ranges = pickle.load(f)
    
    return model_data, normal_ranges

# Load data
model_data, normal_ranges = load_model_and_data()
model = model_data['model']
scaler = model_data['scaler']
feature_names = model_data['feature_names']
model_name = model_data['model_name']
accuracy = model_data['accuracy']

# Helper functions
def check_normal_range(feature, value, normal_ranges):
    """Check if a value is within normal range"""
    if feature in normal_ranges:
        q25 = normal_ranges[feature]['q25']
        q75 = normal_ranges[feature]['q75']
        return q25 <= value <= q75
    return True

def get_risk_level(probability):
    """Get risk level based on probability"""
    if probability < 0.3:
        return "Low Risk", "🟢"
    elif probability < 0.7:
        return "Moderate Risk", "🟡"
    else:
        return "High Risk", "🔴"

def create_gauge_chart(probability):
    """Create a gauge chart for risk probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🎗️ Breast Cancer Risk Assessment Tool Powered By MozBioMed AI!</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Supporting Breast Cancer Detection In Mozambique With Artificial Intelligence <img src="https://flagcdn.com/w40/mz.png" width="40" style="vertical-align: middle;"></h2>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Home", "🔬 Risk Assessment","💡 Recommendations", "📚 About Breast Cancer" ]
    )
    
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔬 Risk Assessment":
        show_risk_assessment()
    elif page == "💡 Recommendations":
        show_recommendations()    
    elif page == "📚 About Breast Cancer":
        show_about_breast_cancer()
    

def show_home_page():
    logo = Image.open("Logo_MozBioMed.AI.jpg")
    st.image(logo)
    st.markdown("""
                <h4>Breast Cancer is one of the leading causes of cancer-related deaths among women globally, and Mozambique 
                is no exception. In recent years the incidence of breast cancer in Mozambique has been rising, particularly in urban areas
                such as Maputo. However due to limited health awareness, many cases are diagnosed at late stages, making treatment less
                effective.</h4>
                """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h2>🎯 Welcome to the Breast Cancer Risk Assessment Tool</h2>
            <p>This advanced machine learning application helps assess breast cancer risk based on cellular characteristics from fine needle aspirate (FNA) of breast mass.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key features
    st.markdown('<h2 class="sub-header">✨ Key Features</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔬</h3>
            <h4>AI-Powered Analysis</h4>
            <p>Advanced machine learning model with instant results.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊</h3>
            <h4>Real-time Assessment</h4>
            <p>Instant risk evaluation with detailed insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>💡</h3>
            <h4>Expert Recommendations</h4>
            <p>Personalized guidance based on results</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Medical Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice. Always consult with healthcare professionals for medical decisions.
    </div>
    """, unsafe_allow_html=True)

def show_risk_assessment():
    """Risk assessment page with input form"""
    st.markdown('<h2 class="sub-header">🔬 Breast Cancer Risk Assessment</h2>', unsafe_allow_html=True)
    
    
    # Input form
    st.markdown("### 📝 Enter Cell Characteristics")
    
    # Organize inputs by categories
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔵 Average Cell Values")
        radius_mean = st.number_input("Average Cell Radius", min_value=0.0, max_value=50.0, value=14.0, step=0.1)
        texture_mean = st.number_input("Average Cell Texture", min_value=0.0, max_value=50.0, value=19.0, step=0.1)
        perimeter_mean = st.number_input("Average Cell Perimeter", min_value=0.0, max_value=200.0, value=92.0, step=0.1)
        area_mean = st.number_input("Average Cell Area", min_value=0.0, max_value=2500.0, value=655.0, step=1.0)
        smoothness_mean = st.number_input("Average Cell Smoothness", min_value=0.0, max_value=1.0, value=0.096, step=0.001)
        compactness_mean = st.number_input("Average Cell Compactness", min_value=0.0, max_value=1.0, value=0.104, step=0.001)
        concavity_mean = st.number_input("Average Cell Concavity", min_value=0.0, max_value=1.0, value=0.089, step=0.001)
        concave_points_mean = st.number_input("Average Concave Points", min_value=0.0, max_value=1.0, value=0.048, step=0.001)
        symmetry_mean = st.number_input("Average Cell Symmetry", min_value=0.0, max_value=1.0, value=0.181, step=0.001)
        fractal_dimension_mean = st.number_input("Average Fractal Dimension ", min_value=0.0, max_value=1.0, value=0.063, step=0.001)
    
    with col2:
        st.markdown("#### 🔴 Maximum Cell Values")
        radius_worst = st.number_input("Maximum Cell Radius ", min_value=0.0, max_value=50.0, value=16.0, step=0.1)
        texture_worst = st.number_input("Maximum Cell Texture", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
        perimeter_worst = st.number_input("Maximum Cell Perimeter", min_value=0.0, max_value=300.0, value=107.0, step=0.1)
        area_worst = st.number_input("Maximum Cell Area", min_value=0.0, max_value=4000.0, value=880.0, step=1.0)
        smoothness_worst = st.number_input("Maximum Cell Smoothness", min_value=0.0, max_value=1.0, value=0.132, step=0.001)
        compactness_worst = st.number_input("Maximum Cell Compactness", min_value=0.0, max_value=1.0, value=0.254, step=0.001)
        concavity_worst = st.number_input("Maximum Cell Concavity", min_value=0.0, max_value=1.0, value=0.273, step=0.001)
        concave_points_worst = st.number_input("Maximum Cell Concave Points ", min_value=0.0, max_value=1.0, value=0.114, step=0.001)
        symmetry_worst = st.number_input("Maximum Cell Symmetry", min_value=0.0, max_value=1.0, value=0.290, step=0.001)
        fractal_dimension_worst = st.number_input("Maximum Cell Fractal Dimension", min_value=0.0, max_value=1.0, value=0.084, step=0.001)
    
    # Collect input data
    input_data = [
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
        radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
        compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
    ]
    
    # Prediction button
    if st.button("🔍 Analyze Risk", type="primary"):
        # Scale the input data
        input_scaled = scaler.transform([input_data])
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]  # Probability of malignant
        
        # Display results
        st.markdown("---")
        st.markdown('<h3 class="sub-header">📊 Analysis Results</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Risk level
            risk_level, risk_icon = get_risk_level(probability)
            
            if probability < 0.3:
                st.markdown(f"""
                <div class="risk-low">
                    <h2>{risk_icon} {risk_level}</h2>
                    <h3>Probability: {probability:.1%}</h3>
                    <p>The analysis suggests a low probability of malignancy.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-high">
                    <h2>{risk_icon} {risk_level}</h2>
                    <h3>Probability: {probability:.1%}</h3>
                    <p>The analysis suggests elevated risk. Please consult a healthcare professional.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Gauge chart
        st.plotly_chart(create_gauge_chart(probability), use_container_width=True)

    st.markdown("""
    <h2 class="sub-header">You can view recommendations on the recommendations page, based on your risk status</h2>
    """, unsafe_allow_html=True)     
        
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Medical Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice. Always consult with healthcare professionals for medical decisions.
    </div>
    """, unsafe_allow_html=True)    

    

def show_about_breast_cancer():
    """Educational content about breast cancer"""
    st.markdown('<h2 class="sub-header">📚 About Breast Cancer</h2>', unsafe_allow_html=True)
    
    # Overview
    st.markdown("""
    <div class="info-box">
        <h3>🎗️ What is Breast Cancer?</h3>
        <p>Breast cancer is a disease in which cells in breast tissue grow uncontrollably. It is the second most common cancer in women, but it can also occur in men. Early detection and treatment significantly improve outcomes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistics
    col1, col2 = st.columns(2)
    
    
    with col1:
        st.metric("5-Year Survival Rate", "99%", "When caught early")
    
    with col2:
        st.metric("Average Age at Diagnosis", "62 years", "Median age")
    
    # Types of breast cancer
    st.markdown("### 🔬 Types of Breast Cancer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Invasive Ductal Carcinoma (IDC)
        - Most common type (80% of cases)
        - Starts in milk ducts
        - Can spread to nearby tissue
        
        #### Invasive Lobular Carcinoma (ILC)
        - Second most common (10-15% of cases)
        - Starts in milk-producing glands
        - Harder to detect on mammograms
        """)
    
    with col2:
        st.markdown("""
        #### Ductal Carcinoma In Situ (DCIS)
        - Non-invasive breast cancer
        - Confined to milk ducts
        - High cure rate with treatment
        
        #### Triple-Negative Breast Cancer
        - Lacks three common receptors
        - More aggressive but responds to chemotherapy
        - More common in younger women
        """)
    
    # Risk factors
    st.markdown("### ⚠️ Risk Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Non-Modifiable Risk Factors
        - **Age**: Risk increases with age
        - **Gender**: Women are at higher risk
        - **Family History**: Genetic predisposition
        - **Personal History**: Previous breast cancer
        - **Genetic Mutations**: BRCA1, BRCA2
        - **Race/Ethnicity**: Varies by population
        """)
    
    with col2:
        st.markdown("""
        #### Modifiable Risk Factors
        - **Lifestyle**: Diet, exercise, alcohol
        - **Reproductive History**: Age at first pregnancy
        - **Hormone Therapy**: Long-term use
        - **Radiation Exposure**: Previous treatments
        - **Weight**: Obesity after menopause
        - **Smoking**: Increases risk
        """)
    
    # Symptoms
    st.markdown("### 🔍 Signs and Symptoms")
    
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Warning Signs to Watch For:</h4>
        <ul>
            <li><strong>Lump or thickening</strong> in breast or underarm</li>
            <li><strong>Change in size or shape</strong> of breast</li>
            <li><strong>Dimpling or puckering</strong> of breast skin</li>
            <li><strong>Nipple discharge</strong> (other than breast milk)</li>
            <li><strong>Nipple turning inward</strong> or changes in appearance</li>
            <li><strong>Redness or scaling</strong> of nipple or breast skin</li>
            <li><strong>Swelling</strong> in part or all of breast</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Screening guidelines
    st.markdown("### 🏥 Screening Guidelines")
    
    screening_data = {
        'Age Group': ['20-39 years', '40-49 years', '50-74 years', '75+ years'],
        'Recommendation': [
            'Clinical breast exam every 3 years',
            'Annual mammogram (discuss with doctor)',
            'Annual mammogram',
            'Discuss with healthcare provider'
        ],
        'Additional Notes': [
            'Self-awareness important',
            'Consider family history',
            'Standard screening age',
            'Based on health status'
        ]
    }
    
    screening_df = pd.DataFrame(screening_data)
    st.table(screening_df)
    
    # Prevention
    st.markdown("### 🛡️ Prevention Strategies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Lifestyle Modifications
        - **Regular Exercise**: 150 minutes/week moderate activity
        - **Healthy Diet**: Rich in fruits, vegetables, whole grains
        - **Maintain Healthy Weight**: Especially after menopause
        - **Limit Alcohol**: No more than 1 drink per day
        - **Avoid Smoking**: Increases cancer risk
        """)
    
    with col2:
        st.markdown("""
        #### Medical Interventions
        - **Regular Screening**: Follow guidelines for your age
        - **Genetic Counseling**: If family history present
        - **Prophylactic Surgery**: For high-risk individuals
        - **Chemoprevention**: Medications for high-risk cases
        - **Hormone Therapy**: Discuss risks and benefits
        """)

def show_recommendations():
    """Personalized recommendations based on risk assessment"""
    st.markdown('<h2 class="sub-header">💡 Personalized Recommendations</h2>', unsafe_allow_html=True)
    
    # Risk-based recommendations
    st.markdown("### 🎯 Risk-Based Recommendations")
    
    # Simulate different risk levels
    risk_levels = ["Low Risk (0-30%)", "Moderate Risk (30-70%)", "High Risk (70%+)"]
    
    for risk_level in risk_levels:
        with st.expander(f"📊 {risk_level}"):
            if "Low Risk" in risk_level:
                st.markdown("""
                <div class="risk-low">
                    <h4>🟢 Low Risk Recommendations</h4>
                    <ul>
                        <li><strong>Continue Regular Screening:</strong> Follow standard mammography guidelines</li>
                        <li><strong>Maintain Healthy Lifestyle:</strong> Regular exercise, balanced diet</li>
                        <li><strong>Self-Awareness:</strong> Perform monthly self-examinations</li>
                        <li><strong>Annual Check-ups:</strong> Regular visits with healthcare provider</li>
                        <li><strong>Stay Informed:</strong> Keep up with latest screening recommendations</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            elif "Moderate Risk" in risk_level:
                st.markdown("""
                <div class="warning-box">
                    <h4>🟡 Moderate Risk Recommendations</h4>
                    <ul>
                        <li><strong>Enhanced Screening:</strong> Consider more frequent mammograms</li>
                        <li><strong>Additional Imaging:</strong> Discuss MRI or ultrasound with doctor</li>
                        <li><strong>Genetic Counseling:</strong> Consider genetic testing if family history</li>
                        <li><strong>Lifestyle Modifications:</strong> Focus on risk reduction strategies</li>
                        <li><strong>Close Monitoring:</strong> More frequent clinical breast exams</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            else:
                st.markdown("""
                <div class="risk-high">
                    <h4>🔴 High Risk Recommendations</h4>
                    <ul>
                        <li><strong>Immediate Consultation:</strong> See oncologist or breast specialist</li>
                        <li><strong>Comprehensive Imaging:</strong> MRI, ultrasound, possible biopsy</li>
                        <li><strong>Genetic Testing:</strong> BRCA1/BRCA2 and other gene panels</li>
                        <li><strong>Consider Chemoprevention:</strong> Discuss medications with doctor</li>
                        <li><strong>Intensive Surveillance:</strong> Every 6 months screening</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    # General health recommendations
    st.markdown("### 🌟 General Health Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🏃‍♀️ Physical Activity
        - **Aerobic Exercise**: 150 minutes moderate or 75 minutes vigorous per week
        - **Strength Training**: 2+ days per week
        - **Daily Movement**: Reduce sedentary time
        - **Activities**: Walking, swimming, cycling, dancing
        
        #### 🥗 Nutrition
        - **Fruits & Vegetables**: 5-9 servings daily
        - **Whole Grains**: Choose over refined grains
        - **Lean Proteins**: Fish, poultry, legumes, nuts
        - **Healthy Fats**: Olive oil, avocados, nuts
        - **Limit**: Processed foods, red meat, sugar
        """)
    
    with col2:
        st.markdown("""
        #### 😴 Lifestyle Factors
        - **Sleep**: 7-9 hours of quality sleep nightly
        - **Stress Management**: Meditation, yoga, counseling
        - **Alcohol**: Limit to 1 drink per day or less
        - **Smoking**: Quit smoking and avoid secondhand smoke
        - **Weight**: Maintain healthy BMI (18.5-24.9)
        
        #### 🏥 Medical Care
        - **Regular Check-ups**: Annual physical exams
        - **Screening**: Follow age-appropriate guidelines
        - **Vaccinations**: Stay up to date
        - **Medications**: Take as prescribed
        - **Communication**: Open dialogue with healthcare team
        """)
    
    # Emergency signs
    st.markdown("### 🚨 When to Seek Immediate Medical Attention")
    
    st.markdown("""
    <div class="risk-high">
        <h4>⚠️ Contact Your Healthcare Provider Immediately If You Notice:</h4>
        <ul>
            <li><strong>New Lump:</strong> Any new lump in breast or underarm area</li>
            <li><strong>Skin Changes:</strong> Dimpling, puckering, or orange-peel texture</li>
            <li><strong>Nipple Changes:</strong> Discharge, inversion, or scaling</li>
            <li><strong>Breast Changes:</strong> Sudden size or shape changes</li>
            <li><strong>Pain:</strong> Persistent, unexplained breast pain</li>
            <li><strong>Swelling:</strong> Unexplained swelling in breast or lymph nodes</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    

if __name__ == "__main__":
    main()

