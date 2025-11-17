import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Job Role Prediction App (RandomForest + XGBoost)")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # STEP 1: Oversample company
    # -------------------------------
    X = df.drop("company", axis=1)
    y = df["company"]

    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    df = pd.concat([X_resampled, y_resampled], axis=1)

    # -------------------------------
    # STEP 2: Oversample program
    # -------------------------------
    X = df.drop("program", axis=1)
    y = df["program"]

    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    df = pd.concat([X_resampled, y_resampled], axis=1)

    # -------------------------------
    # STEP 3: Job Group Classification
    # -------------------------------
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

    df["job_group"] = df["job"].apply(classify_job)

    # -------------------------------
    # STEP 4: Package Grouping
    # -------------------------------
    df["package"] = pd.to_numeric(df["package"], errors="coerce")

    bins = [0, 5, 10, 15]
    labels = [0, 1, 2]
    df["package_group"] = pd.cut(df["package"], bins=bins, labels=labels, include_lowest=True)
    df["package_group"] = df["package_group"].astype(int)

    # -------------------------------
    # STEP 5: Convert text to lower and encode
    # -------------------------------
    def to_lower(x):
        return str(x).lower().strip() if isinstance(x, str) else x

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(to_lower)

    le = LabelEncoder()
    for col in df.select_dypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    # -------------------------------
    # STEP 6: Select Features
    # -------------------------------
    features = ["company", "job_group", "package_group"]
    X = df[features]
    y = df["program"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # -------------------------------
    # Train-Test Split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------------
    # Apply SMOTE
    # -------------------------------
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # -------------------------------
    # Train RandomForest
    # -------------------------------
    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
    rf_model.fit(X_train_res, y_train_res)
    rf_pred = rf_model.predict(X_test)

    st.write("### 🎯 RandomForest Classification Report")
    st.text(classification_report(y_test, rf_pred))

    # -------------------------------
    # Train XGBoost
    # -------------------------------
    xgb_model = XGBClassifier(
        objective="multi:softmax",
        num_class=len(y.unique()),
        learning_rate=0.1,
        max_depth=5,
        n_estimators=300,
        eval_metric="mlogloss"
    )
    xgb_model.fit(X_train_res, y_train_res)
    xgb_pred = xgb_model.predict(X_test)

    st.write("### 🚀 XGBoost Classification Report")
    st.text(classification_report(y_test, xgb_pred))

    # -------------------------------
    # Confusion Matrix
    # -------------------------------
    st.write("## 📉 Confusion Matrix (XGBoost)")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, xgb_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # -------------------------------
    # Prediction Section
    # -------------------------------
    st.write("## 🔮 Predict New Job Program")

    company = st.number_input("Company (Encoded Value)", min_value=0)
    job_group = st.number_input("Job Group (Encoded Value)", min_value=0)
    package_group = st.selectbox("Package Group", [0, 1, 2])

    if st.button("Predict Program"):
        new_data = scaler.transform([[company, job_group, package_group]])
        pred = xgb_model.predict(new_data)
        st.success(f"Predicted Program: {pred[0]}")
