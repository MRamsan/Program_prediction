import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Program Recommendation System",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        font-size: 1.1rem;
        border-radius: 10px;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_or_train_model():
    """Load existing model or train a new one"""
    
    # Try to load pre-trained model
    if os.path.exists('model.pkl') and os.path.exists('encoders.pkl') and os.path.exists('scaler.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, encoders, scaler
    
    # If model doesn't exist, train it
    st.warning("Model not found. Please ensure you have the dataset and run training first.")
    return None, None, None

def train_model(df):
    """Train the model with the provided dataset"""
    
    # Data preprocessing
    from imblearn.over_sampling import RandomOverSampler
    
    # Balance company
    X = df.drop("company", axis=1)
    y = df["company"]
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    df = pd.concat([X_resampled, y_resampled], axis=1)
    
    # Balance program
    X = df.drop("program", axis=1)
    y = df["program"]
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    df = pd.concat([X_resampled, y_resampled], axis=1)
    
    # Job grouping
    job_groups = {
        'software_engineering': [
            'Software Engineer', 'Software Design Engineer', 'System Development Engineer',
            'Full stack Developer', 'Python Developer', 'QA Engineer', 'Digital Engineer'
        ],
        'data_science': [
            'Data Scientist', 'Associate Data Scientist', 'Data Analyst', 'Data Analytics',
            'AI', 'AI-Validation AI/ML Engineer', 'ML Engineer', 'AIand Machine Learning Algorithms.',
            'Cloud transformation& AI'
        ],
        'cyber_security': [
            'Associate Security Engineer', 'Security Analyst EY', 'Application security Engineer',
            'Cyber Security Engineer', 'Cyber Security', 'Network Security Engineering',
            'Information Security', 'Intelligence Analyst, Cyber Threat Intelligence'
        ],
        'marketing_sales': [
            'Marketing', 'Marketing Executive Corporate Relations Internship',
            'Sales Development Freshers', 'Sales Trainee', 'Business Development',
            'Business Development Trainee'
        ],
        'business_management': [
            'Manager', 'Management Trainee', 'Consultant-Functional', 'Business Analyst',
            'Operations Associate', 'Academic Associate (Teacher)', 'Ecologist'
        ],
        'internship': ['Internship with Placement'],
        'others': ['Geospatial Analyst', 'Digital Media Analyst', 'Engineer']
    }
    
    def classify_job(title):
        title = str(title).lower().strip()
        for group, keywords in job_groups.items():
            for keyword in keywords:
                if keyword.lower() in title:
                    return group
        return 'others'
    
    df['job_group'] = df['job'].apply(classify_job)
    
    # Package grouping
    df['package'] = pd.to_numeric(df['package'], errors='coerce')
    bins = [0, 5, 10, 15]
    labels = [0, 1, 2]
    df['package_group'] = pd.cut(df['package'], bins=bins, labels=labels, include_lowest=True)
    df['package_group'] = df['package_group'].astype(int)
    
    # Convert to lowercase
    def to_lower(title):
        if isinstance(title, str): 
            return title.lower().strip()
        return title
    
    for i in df.select_dtypes(include=['object']).columns:
        df[i] = df[i].apply(to_lower)
    
    # Label encoding
    encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    # Prepare features
    columns = ['company', 'job_group', 'package_group']
    X = df[columns]
    y = df['program']
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_scaled, y)
    
    # Train model
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=len(y.unique()),
        learning_rate=0.1,
        max_depth=5,
        n_estimators=300,
        eval_metric="mlogloss"
    )
    
    model.fit(X_train_res, y_train_res)
    
    # Save model and preprocessors
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return model, encoders, scaler

def main():
    st.title("🎓 Program Recommendation System")
    st.markdown("### Find the best program based on company, job role, and package")
    
    # Load model
    model, encoders, scaler = load_or_train_model()
    
    if model is None:
        st.error("Model not loaded. Please upload a dataset to train the model.")
        
        uploaded_file = st.file_uploader("Upload your CSV dataset", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("Dataset loaded successfully!")
            
            if st.button("Train Model"):
                with st.spinner("Training model... This may take a few minutes."):
                    model, encoders, scaler = train_model(df)
                    st.success("Model trained successfully!")
                    st.rerun()
        return
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        # Company input
        company_options = list(encoders['company'].classes_)
        company = st.selectbox(
            "Select Company",
            options=company_options,
            help="Choose the company offering the job"
        )
        
        # Job group input
        job_group_options = [
            'data_science',
            'software_engineering',
            'marketing_sales',
            'internship',
            'cyber_security',
            'others',
            'business_management'
        ]
        job_group = st.selectbox(
            "Select Job Category",
            options=job_group_options,
            help="Choose the job category"
        )
    
    with col2:
        # Package input
        package = st.number_input(
            "Enter Package (in LPA)",
            min_value=0.0,
            max_value=15.0,
            value=6.0,
            step=0.5,
            help="Enter the salary package in Lakhs Per Annum"
        )
        
        # Package grouping
        if package <= 5:
            package_group = 0
        elif package <= 10:
            package_group = 1
        else:
            package_group = 2
        
        st.info(f"Package Range: {'0-5 LPA' if package_group == 0 else '5-10 LPA' if package_group == 1 else '10-15 LPA'}")
    
    # Predict button
    if st.button("🔮 Predict Program", use_container_width=True):
        try:
            # Encode inputs
            company_encoded = encoders['company'].transform([company.lower().strip()])[0]
            job_group_encoded = encoders['job_group'].transform([job_group.lower().strip()])[0]
            
            # Create input array
            input_data = np.array([[company_encoded, job_group_encoded, package_group]])
            
            # Scale input
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Decode prediction
            program = encoders['program'].inverse_transform([prediction])[0]
            
            # Display result
            st.markdown("---")
            st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
            st.success(f"### 🎯 Recommended Program: **{program.upper()}**")
            
            # Show confidence
            confidence = prediction_proba[prediction] * 100
            st.metric("Confidence", f"{confidence:.1f}%")
            
            # Show all probabilities
            st.markdown("#### All Program Probabilities:")
            proba_df = pd.DataFrame({
                'Program': encoders['program'].inverse_transform(range(len(prediction_proba))),
                'Probability': prediction_proba * 100
            }).sort_values('Probability', ascending=False)
            
            proba_df['Probability'] = proba_df['Probability'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(proba_df, hide_index=True, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please make sure all inputs are valid and the model is properly trained.")

if __name__ == "__main__":
    main()
