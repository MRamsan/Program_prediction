"""
Enhanced Academic Program Predictor - Streamlit Application
Supports Random Forest and Gradient Boosting models
Flexible input handling (1, 2, or all 3 inputs)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Enhanced Program Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .metric-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .info-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        background: #e0e7ff;
        color: #4338ca;
        font-weight: 600;
        margin: 0.25rem;
    }
    .prob-bar {
        background-color: #e5e7eb;
        border-radius: 10px;
        height: 40px;
        margin-top: 10px;
        overflow: hidden;
        position: relative;
    }
    .prob-fill {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        transition: width 0.5s ease;
    }
    .input-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .status-provided {
        background: #d1fae5;
        color: #065f46;
        border-left: 3px solid #10b981;
    }
    .status-default {
        background: #fef3c7;
        color: #92400e;
        border-left: 3px solid #f59e0b;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# JOB CLASSIFICATION
# ============================================================================
JOB_GROUPS = {
    'ai_data_ml': [
        'data scientist', 'associate data scientist', 'business analyst',
        'data analyst', 'data analytics', 'machine learning',
        'ml engineer', 'ai', 'cloud transformation'
    ],
    'cyber_security': [
        'security analyst', 'security engineer', 'cyber security',
        'application security', 'information security',
        'threat intelligence', 'network security'
    ],
    'software_engineering': [
        'software engineer', 'software design engineer',
        'system development engineer', 'developer', 'python',
        'full stack', 'qa engineer', 'engineer', 'digital engineer',
        'digital media analyst', 'geospatial analyst'
    ],
    'business_management': [
        'manager', 'management trainee', 'consultant-functional',
        'operations associate', 'academic associate', 'teacher',
        'ecologist', 'analyst', 'marketing', 'business development', 
        'sales trainee', 'sales development', 'business development trainee'
    ],
    'others': []
}

def classify_job(title):
    """Classify job title into predefined groups"""
    if not title:
        return 'others'
    
    title = str(title).lower().strip()
    for group, keywords in JOB_GROUPS.items():
        for keyword in keywords:
            if keyword.lower() in title:
                return group
    return 'others'

# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained model and preprocessing artifacts"""
    try:
        with open('program_prediction_model.pkl', 'rb') as f:
            model_artifacts = pickle.load(f)
        return model_artifacts
    except FileNotFoundError:
        st.error("⚠️ Model file not found: program_prediction_model.pkl")
        st.info("Please ensure program_prediction_model.pkl is in the app directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def make_prediction(company, job_role, package, model_artifacts):
    """
    Make program prediction based on available inputs.
    Can work with 1, 2, or all 3 inputs provided.
    """
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    encoders = model_artifacts['encoders']
    
    # Track which inputs were provided
    inputs_status = {
        'company': False,
        'job_role': False,
        'package': False
    }
    
    # Initialize feature dictionary
    features = {}
    
    # Handle Company
    if company and company.strip():
        inputs_status['company'] = True
        company = company.strip()
        try:
            company_encoded = encoders['company'].transform([company])[0]
            features['company_encoded'] = company_encoded
        except ValueError:
            # Unknown company - use first class as default
            company_encoded = encoders['company'].transform([encoders['company'].classes_[0]])[0]
            features['company_encoded'] = company_encoded
    else:
        # Use median encoding
        features['company_encoded'] = len(encoders['company'].classes_) // 2
    
    # Handle Job Role
    if job_role and job_role.strip():
        inputs_status['job_role'] = True
        job_group = classify_job(job_role)
        try:
            job_encoded = encoders['job'].transform([job_group])[0]
            features['job_group_encoded'] = job_encoded
            features['job_group_name'] = job_group
        except ValueError:
            job_encoded = 0
            features['job_group_encoded'] = job_encoded
            features['job_group_name'] = 'others'
    else:
        features['job_group_encoded'] = 0
        features['job_group_name'] = 'default'
    
    # Handle Package
    if package is not None:
        inputs_status['package'] = True
        # Discretize package using the saved bins
        package_group = pd.cut([package], bins=encoders['bins'], 
                              labels=encoders['labels'], include_lowest=True)[0]
        features['package_group'] = int(package_group)
        features['package_value'] = package
    else:
        features['package_group'] = 2  # Median group
        features['package_value'] = None
    
    # Create feature array
    feature_array = np.array([[
        features['company_encoded'],
        features['job_group_encoded'],
        features['package_group']
    ]])
    
    # Scale features
    feature_scaled = scaler.transform(feature_array)
    
    # Make prediction
    prediction_encoded = model.predict(feature_scaled)[0]
    prediction_proba = model.predict_proba(feature_scaled)[0]
    
    # Decode prediction
    prediction = encoders['program'].inverse_transform([prediction_encoded])[0]
    
    # Create probability dictionary
    prob_dict = {}
    for i, prob in enumerate(prediction_proba):
        program = encoders['program'].classes_[i]
        prob_dict[program] = prob
    
    return {
        'predicted_program': prediction,
        'confidence': prediction_proba[prediction_encoded],
        'probabilities': prob_dict,
        'job_group': features.get('job_group_name', 'default'),
        'inputs_status': inputs_status,
        'package_group': features['package_group'],
        'inputs_count': sum(inputs_status.values())
    }

# ============================================================================
# UI COMPONENTS
# ============================================================================
def display_header():
    """Display application header"""
    st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        border-radius: 15px; margin-bottom: 2rem;'>
            <h1 style='color: white; margin: 0; font-size: 2.5rem;'>🎓 Enhanced Academic Program Predictor</h1>
            <p style='color: white; font-size: 1.2rem; margin-top: 0.5rem;'>
                Flexible AI System - Works with 1, 2, or All 3 Inputs
            </p>
        </div>
    """, unsafe_allow_html=True)

def display_sidebar(model_artifacts):
    """Display sidebar with information"""
    with st.sidebar:
        st.markdown("### 🔍 About This System")
        st.info("""
            This enhanced prediction system can make accurate recommendations 
            even with partial information. Provide any combination of:
            - Company name
            - Job role
            - Package
        """)
        
        st.markdown("---")
        
        st.markdown("### 📊 Model Information")
        model_name = model_artifacts.get('model_name', 'Ensemble Model')
        st.write(f"**Algorithm:** {model_name}")
        st.write(f"**Programs:** {len(model_artifacts['encoders']['program'].classes_)}")
        st.write(f"**Companies:** {len(model_artifacts['encoders']['company'].classes_)}")
        st.write(f"**Job Groups:** {len(model_artifacts['encoders']['job'].classes_)}")
        
        st.markdown("---")
        
        st.markdown("### 📚 Available Programs")
        programs = sorted(model_artifacts['encoders']['program'].classes_)
        for prog in programs:
            st.write(f"• **{prog.upper()}**")
        
        st.markdown("---")
        
        st.markdown("### 🎯 Job Categories")
        for group in JOB_GROUPS.keys():
            if group != 'others':
                st.write(f"• {group.replace('_', ' ').title()}")
        
        st.markdown("---")
        
        st.markdown("### 💰 Package Ranges")
        st.write("• **0-4 LPA:** Entry Level")
        st.write("• **4-7 LPA:** Junior")
        st.write("• **7-9 LPA:** Mid Level")
        st.write("• **9-14 LPA:** Senior")
        st.write("• **14+ LPA:** Expert")

def display_input_status(inputs_status):
    """Display which inputs were provided"""
    st.markdown("### 📋 Input Status")
    
    cols = st.columns(3)
    
    status_info = [
        ('company', '🏢 Company', inputs_status['company']),
        ('job_role', '💼 Job Role', inputs_status['job_role']),
        ('package', '💰 Package', inputs_status['package'])
    ]
    
    for i, (key, label, provided) in enumerate(status_info):
        with cols[i]:
            if provided:
                st.markdown(f"""
                    <div class='input-status status-provided'>
                        <strong>{label}</strong><br>
                        ✓ Provided
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='input-status status-default'>
                        <strong>{label}</strong><br>
                        ○ Using Default
                    </div>
                """, unsafe_allow_html=True)

def display_results(result):
    """Display prediction results"""
    # Input status
    display_input_status(result['inputs_status'])
    
    st.markdown("---")
    
    # Main prediction card
    st.markdown(f"""
        <div class='prediction-card'>
            <h2 style='color: white; margin: 0; font-size: 1.5rem;'>Recommended Program</h2>
            <h1 style='color: white; font-size: 3rem; margin: 1rem 0;'>
                {result['predicted_program'].upper()}
            </h1>
            <p style='color: white; font-size: 1.2rem; margin: 0;'>
                Confidence: {result['confidence']:.1%}
            </p>
            <p style='color: white; font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;'>
                Based on {result['inputs_count']} input(s) provided
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence_color = "#22c55e" if result['confidence'] > 0.7 else "#f59e0b" if result['confidence'] > 0.5 else "#ef4444"
        confidence_label = "High" if result['confidence'] > 0.7 else "Moderate" if result['confidence'] > 0.5 else "Fair"
        st.markdown(f"""
            <div class='metric-card'>
                <h3 style='color: {confidence_color}; margin: 0;'>{result['confidence']:.1%}</h3>
                <p style='color: #6b7280; margin: 0.5rem 0 0 0;'>Confidence Level</p>
                <p style='color: {confidence_color}; margin: 0.25rem 0 0 0; font-size: 0.9rem;'>{confidence_label}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        job_display = result['job_group'].replace('_', ' ').title() if result['job_group'] != 'default' else 'Not Specified'
        st.markdown(f"""
            <div class='metric-card'>
                <h3 style='color: #3b82f6; margin: 0;'>{job_display}</h3>
                <p style='color: #6b7280; margin: 0.5rem 0 0 0;'>Job Category</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        rank_2 = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)[1]
        st.markdown(f"""
            <div class='metric-card'>
                <h3 style='color: #8b5cf6; margin: 0;'>{rank_2[0].upper()}</h3>
                <p style='color: #6b7280; margin: 0.5rem 0 0 0;'>2nd Choice</p>
                <p style='color: #8b5cf6; margin: 0.25rem 0 0 0; font-size: 0.9rem;'>{rank_2[1]:.1%}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Probability breakdown
    st.markdown("---")
    st.markdown("### 📊 All Program Probabilities")
    
    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
    
    for i, (program, prob) in enumerate(sorted_probs):
        icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📌"
        
        col_prog, col_bar = st.columns([1, 3])
        
        with col_prog:
            st.markdown(f"**{icon} {program.upper()}**")
        
        with col_bar:
            st.markdown(f"""
                <div class='prob-bar'>
                    <div class='prob-fill' style='width: {prob*100}%;'>
                        {prob:.1%}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # Interpretation
    st.markdown("---")
    st.markdown("### 💡 Interpretation & Recommendations")
    
    inputs_count = result['inputs_count']
    confidence = result['confidence']
    
    # Confidence interpretation
    if confidence > 0.75:
        confidence_msg = "**High Confidence Prediction** ✅"
        confidence_detail = f"The model is highly confident that **{result['predicted_program'].upper()}** is the best fit for your profile."
    elif confidence > 0.55:
        confidence_msg = "**Moderate Confidence Prediction** ℹ️"
        confidence_detail = f"The model suggests **{result['predicted_program'].upper()}** as the top choice, but **{sorted_probs[1][0].upper()}** ({sorted_probs[1][1]:.1%}) is also worth considering."
    else:
        confidence_msg = "**Fair Confidence Prediction** ⚠️"
        confidence_detail = "The prediction shows fair confidence. Multiple programs may be suitable for your profile."
    
    # Input completeness message
    if inputs_count == 3:
        input_msg = "**Complete Information** 🎯: All three inputs provided - most accurate prediction possible."
    elif inputs_count == 2:
        input_msg = "**Partial Information** 📝: Two inputs provided - good prediction accuracy with some defaults applied."
    else:
        input_msg = "**Limited Information** 📋: Only one input provided - prediction uses more default values. Providing more information will improve accuracy."
    
    # Display messages
    if confidence > 0.7:
        st.success(f"{confidence_msg}\n\n{confidence_detail}\n\n{input_msg}")
    elif confidence > 0.5:
        st.info(f"{confidence_msg}\n\n{confidence_detail}\n\n{input_msg}")
    else:
        st.warning(f"{confidence_msg}\n\n{confidence_detail}\n\n{input_msg}\n\nTop recommendations:\n1. **{sorted_probs[0][0].upper()}** ({sorted_probs[0][1]:.1%})\n2. **{sorted_probs[1][0].upper()}** ({sorted_probs[1][1]:.1%})")
    
    # Improvement suggestions
    if inputs_count < 3:
        st.markdown("### 🎯 Improve Prediction Accuracy")
        missing = []
        if not result['inputs_status']['company']:
            missing.append("Company name")
        if not result['inputs_status']['job_role']:
            missing.append("Job role")
        if not result['inputs_status']['package']:
            missing.append("Package information")
        
        st.info(f"💡 **Tip:** Provide {', '.join(missing)} for more accurate predictions!")

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application logic"""
    
    # Display header
    display_header()
    
    # Load model
    model_artifacts = load_model()
    
    if model_artifacts is None:
        st.stop()
    
    # Display sidebar
    display_sidebar(model_artifacts)
    
    # Main input section
    st.markdown("## 📝 Enter Your Information")
    
    # Info message
    st.info("💡 **Flexible Input:** You can provide any combination of inputs (1, 2, or all 3). The system will make predictions even with partial information!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏢 Company (Optional)")
        
        # Get list of known companies
        companies = [''] + sorted(list(model_artifacts['encoders']['company'].classes_))
        
        company_mode = st.radio(
            "Select input method:",
            ["Choose from list", "Enter manually"],
            horizontal=True,
            key="company_mode"
        )
        
        if company_mode == "Choose from list":
            company = st.selectbox(
                "Select company",
                companies,
                help="Choose from known companies or leave blank"
            )
        else:
            company = st.text_input(
                "Enter company name:",
                placeholder="e.g., IBM, Google, Microsoft",
                help="Type any company name"
            )
        
        st.markdown("---")
        
        st.markdown("#### 💼 Job Role (Optional)")
        
        job_examples = [
            '',
            'Data Scientist',
            'Software Engineer',
            'Cyber Security Analyst',
            'Business Analyst',
            'Full Stack Developer',
            'ML Engineer',
            'Security Engineer',
            'Management Trainee'
        ]
        
        job_mode = st.radio(
            "Select input method:",
            ["Choose from examples", "Enter manually"],
            horizontal=True,
            key="job_mode"
        )
        
        if job_mode == "Choose from examples":
            job_role = st.selectbox(
                "Select job role",
                job_examples,
                help="Select from common examples or leave blank"
            )
        else:
            job_role = st.text_input(
                "Enter job role:",
                placeholder="e.g., Python Developer, Marketing Manager",
                help="Type any job role"
            )
    
    with col2:
        st.markdown("#### 💰 Package Information (Optional)")
        
        package_mode = st.radio(
            "Input method:",
            ["Slider", "Text Input", "Skip"],
            horizontal=True,
            key="package_mode"
        )
        
        package = None
        
        if package_mode == "Slider":
            package = st.slider(
                "Annual Package (LPA)",
                min_value=0.0,
                max_value=20.0,
                value=6.0,
                step=0.5,
                help="Select your salary package"
            )
        elif package_mode == "Text Input":
            package_text = st.text_input(
                "Enter package (LPA):",
                placeholder="e.g., 8.5",
                help="Enter package in lakhs per annum"
            )
            if package_text:
                try:
                    package = float(package_text)
                except ValueError:
                    st.error("Please enter a valid number")
                    package = None
        
        if package is not None:
            st.metric("Selected Package", f"₹{package} LPA")
            
            # Show package range
            if package <= 4:
                st.info("📊 Range: Entry Level (0-4 LPA)")
            elif package <= 7:
                st.info("📊 Range: Junior Level (4-7 LPA)")
            elif package <= 9:
                st.info("📊 Range: Mid Level (7-9 LPA)")
            elif package <= 14:
                st.info("📊 Range: Senior Level (9-14 LPA)")
            else:
                st.info("📊 Range: Expert Level (14+ LPA)")
        else:
            st.info("⏭️ Package input skipped - will use default value")
    
    # Prediction button
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn2:
        predict_button = st.button("🚀 Predict Program", use_container_width=True)
    
    # Make prediction
    if predict_button:
        # Count provided inputs
        provided_count = sum([
            bool(company and company.strip()),
            bool(job_role and job_role.strip()),
            package is not None
        ])
        
        if provided_count == 0:
            st.warning("⚠️ Please provide at least one input for meaningful predictions!")
        else:
            with st.spinner("🔮 Analyzing your profile..."):
                result = make_prediction(company, job_role, package, model_artifacts)
            
            st.success(f"✅ Prediction Complete! (Based on {provided_count}/3 inputs)")
            display_results(result)
    
    # Example predictions section
    with st.expander("📚 Example Predictions - Click to Explore"):
        st.markdown("### Example Scenarios")
        
        examples = [
            {
                'title': "Complete Profile",
                'company': "IBM",
                'job': "Data Scientist",
                'package': 8.5
            },
            {
                'title': "Company + Package Only",
                'company': "Federal Bank",
                'job': None,
                'package': 12.0
            },
            {
                'title': "Job Role Only",
                'company': None,
                'job': "Cyber Security Engineer",
                'package': None
            },
            {
                'title': "Package Only",
                'company': None,
                'job': None,
                'package': 5.5
            }
        ]
        
        for example in examples:
            with st.container():
                st.markdown(f"**{example['title']}**")
                col_ex1, col_ex2, col_ex3, col_ex4 = st.columns([2, 2, 2, 1])
                
                with col_ex1:
                    st.text(f"Company: {example['company'] or 'N/A'}")
                with col_ex2:
                    st.text(f"Job: {example['job'] or 'N/A'}")
                with col_ex3:
                    st.text(f"Package: {example['package'] or 'N/A'} LPA")
                with col_ex4:
                    if st.button("Try", key=f"ex_{example['title']}"):
                        result = make_prediction(
                            example['company'],
                            example['job'],
                            example['package'],
                            model_artifacts
                        )
                        st.write(f"**Prediction:** {result['predicted_program'].upper()} ({result['confidence']:.1%})")
                
                st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #6b7280; padding: 1rem;'>
            <p>🎓 Enhanced Academic Program Predictor | Powered by Machine Learning</p>
            <p style='font-size: 0.875rem;'>Built with Streamlit • Random Forest / Gradient Boosting</p>
            <p style='font-size: 0.8rem;'>💡 Works with partial information • Flexible input handling</p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
