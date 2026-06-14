# ============================================================
# Customer Churn Intelligence Platform
# AI-Powered Customer Retention & Predictive Analytics
# Developed by Arun Sharma
# ============================================================

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ============================================================
# CONSTANTS
# ============================================================

THRESHOLD = 0.5905

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

MODELS_DIR = PROJECT_ROOT / "Models"
ASSETS_DIR = BASE_DIR / "assets"
PNG_DIR = ASSETS_DIR / "png"

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Intelligence Platform",
    page_icon=str(PNG_DIR / "logo.png"),
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")

    return model, feature_names

if not MODELS_DIR.exists():
    st.error("Models folder not found.")
    st.stop()

model, feature_names = load_artifacts()


# ============================================================
# ICON HELPER
# ============================================================
import base64

def get_base64_image(path):
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception:
        return ""

def icon_html(icon_name, width=40, margin_right=15):
    path = PNG_DIR / icon_name
    b64 = get_base64_image(path)
    if b64:
        return f'<img src="data:image/png;base64,{b64}" width="{width}" style="margin-right: {margin_right}px; vertical-align: middle;">'
    return ""

def icon(icon_name, width=40):
    path = PNG_DIR / icon_name

    try:
        if path.exists():
            st.image(str(path), width=width)
    except:
        pass


# ============================================================
# AURORA LUXE CSS
# ============================================================

st.markdown(
    """
<style>

/* ==========================================================
DYNAMIC THEME (Light & Dark Mode Support)
========================================================== */

/* 
  Removed hardcoded background colors for .stApp and sidebar
  to rely on Streamlit's native theme. Added subtle borders 
  and smooth transitions instead.
*/

section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128, 128, 128, 0.1);
    transition: background-color 0.3s ease;
}

/* ==========================================================
TYPOGRAPHY
========================================================== */

html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", sans-serif;
}

h1, h2, h3, h4 {
    /* Utilizing Streamlit's native text color variable */
    color: var(--text-color);
}

/* ==========================================================
CARDS (Hero, Feature, Metric, Result)
========================================================== */

/* Using var(--secondary-background-color) allows adaptive background */
.hero-card, .feature-card, .metric-card, .result-card {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.hero-card {
    border-radius: 20px;
    padding: 30px;
    background: linear-gradient(135deg, var(--secondary-background-color), rgba(99, 102, 241, 0.05));
    border-left: 4px solid #6366F1;
}

.feature-card {
    padding: 18px;
    margin-bottom: 15px;
}

.hero-card:hover, .feature-card:hover, .metric-card:hover, .result-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
}

/* ==========================================================
METRICS CONTAINER (Streamlit Native)
========================================================== */

div[data-testid="metric-container"] {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 16px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
}

/* ==========================================================
BUTTONS
========================================================== */

.stButton > button {
    background: linear-gradient(135deg, #6366F1, #818CF8);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-weight: 600;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(99, 102, 241, 0.2);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #4F46E5, #6366F1);
    color: white;
    box-shadow: 0 6px 14px rgba(99, 102, 241, 0.4);
    transform: translateY(-1px);
}

.stButton > button p {
    color: white !important;
}

/* ==========================================================
FOOTER
========================================================== */

.footer {
    text-align: center;
    color: var(--text-color);
    opacity: 0.6;
    padding-top: 20px;
    padding-bottom: 20px;
}

/* ==========================================================
INPUTS
========================================================== */

div[data-baseweb="select"] {
    background-color: var(--secondary-background-color);
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(icon_html("logo.png", 60, 0), unsafe_allow_html=True)

    st.markdown(
        """
## Customer Churn
### Intelligence Platform
"""
    )

    st.caption(
        "AI-Powered Customer Retention & Predictive Analytics"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Predict Churn",
            "About",
        ]
    )

    st.divider()

    st.markdown("### Model Summary")

    st.metric(
        "Best Model",
        "Random Forest",
    )

    st.metric(
        "ROC-AUC",
        "83.60%",
    )

    st.metric(
        "Recall",
        "78.61%",
    )

    st.metric(
        "Threshold",
        "59.05%",
    )
# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    st.markdown(
        f"""
        <div class="hero-card" style="display: flex; align-items: center; margin-bottom: 25px;">
            {icon_html("brain-circuit.png", 75, 25)}
            <div>
                <h1 style="margin: 0; padding-bottom: 5px;">Customer Churn Intelligence Platform</h1>
                <p style="margin: 0; padding-bottom: 5px; color: var(--text-color); opacity: 0.8; font-size: 1.1em;">AI-Powered Customer Retention & Predictive Analytics</p>
                <p style="margin: 0; font-size: 0.85em; opacity: 0.6;">Built by Arun Sharma</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # ========================================================
    # KPI SECTION
    # ========================================================

    k1, k2, k3, k4 = st.columns(4)

    with k1:

        st.metric(
            "ROC-AUC",
            "83.60%"
        )

    with k2:

        st.metric(
            "Recall",
            "78.61%"
        )

    with k3:

        st.metric(
            "Features",
            "30"
        )

    with k4:

        st.metric(
            "Threshold",
            "59.05%"
        )

    st.write("")

    # ========================================================
    # PROJECT OBJECTIVE
    # ========================================================

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            {icon_html("target.png", 32, 12)}
            <h3 style="margin: 0; padding: 0;">Project Objective</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="feature-card">

This application predicts whether a telecom customer is likely to churn using Machine Learning.

The platform helps businesses identify customers at risk of leaving and supports proactive retention strategies before churn occurs.

The objective is to transform predictive analytics into actionable business intelligence.

</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    # ========================================================
    # KEY FEATURES
    # ========================================================

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            {icon_html("rocket.png", 32, 12)}
            <h3 style="margin: 0; padding: 0;">Key Features</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    features = [
        "Predict customer churn risk in real time.",
        "Display churn probability using a trained Random Forest model.",
        "Categorize customers into actionable risk levels.",
        "Identify potential churn drivers.",
        "Visualize risk through an interactive gauge chart.",
        "Integrated with Power BI dashboards.",
        "Built using Python, Streamlit, and Scikit-Learn.",
    ]

    for feature in features:

        st.markdown(
            f"""
<div class="feature-card">
• {feature}
</div>
""",
            unsafe_allow_html=True,
        )

    st.write("")

    # ========================================================
    # TECHNOLOGY STACK
    # ========================================================

    st.subheader("Technology Stack")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:

        st.info(
            """
Python

• Pandas
• NumPy
• Scikit-Learn
"""
        )

    with tech2:

        st.info(
            """
Machine Learning

• Random Forest
• Joblib
• Plotly
"""
        )

    with tech3:

        st.info(
            """
Visualization

• Streamlit
• Power BI
• DAX
"""
        )

    st.write("")

    # ========================================================
    # BUSINESS VALUE
    # ========================================================

    st.subheader("Why This Matters")

    st.markdown(
        """
<div class="feature-card">

Customer retention is significantly more cost-effective than customer acquisition.

By identifying customers likely to churn, organizations can:

• Reduce revenue loss.

• Prioritize retention campaigns.

• Improve customer satisfaction.

• Optimize marketing spend.

• Enable data-driven decision making.

</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    # ========================================================
    # PROJECT SNAPSHOT
    # ========================================================

    snap1, snap2, snap3 = st.columns(3)

    with snap1:

        st.metric(
            "Dataset",
            "IBM Telco"
        )

    with snap2:

        st.metric(
            "Algorithm",
            "Random Forest"
        )

    with snap3:

        st.metric(
            "Deployment",
            "Streamlit"
        )

# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "About":

    st.markdown(
        f"""
        <div class="hero-card" style="display: flex; align-items: center; margin-bottom: 25px;">
            {icon_html("brain-circuit.png", 65, 20)}
            <div>
                <h1 style="margin: 0; padding-bottom: 5px;">About the Project</h1>
                <p style="margin: 0; color: var(--text-color); opacity: 0.8; font-size: 1.1em;">End-to-End Customer Churn Intelligence Solution</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # ========================================================
    # PROJECT OVERVIEW
    # ========================================================

    st.subheader("Project Overview")

    st.markdown(
        """
<div class="feature-card">

The Customer Churn Intelligence Platform is an end-to-end predictive analytics solution designed to identify customers at risk of churn and transform machine learning outputs into actionable business intelligence.

The project combines:

• Exploratory Data Analysis

• Machine Learning

• Streamlit Deployment

• Interactive Power BI Dashboards

• Predictive Analytics

• Customer Retention Intelligence

</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    st.subheader("Dataset")

    st.markdown(
        """
<div class="feature-card">

Dataset Used:

IBM Telco Customer Churn Dataset

The dataset contains customer demographic information, subscription details, billing information, and churn outcomes.

It is widely used for customer retention and predictive analytics research.

</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    # ========================================================
    # TECHNOLOGIES USED
    # ========================================================

    st.subheader("Technologies Used")

    t1, t2, t3 = st.columns(3)

    with t1:

        st.success(
            """
Programming

• Python
• Pandas
• NumPy
"""
        )

    with t2:

        st.success(
            """
Machine Learning

• Scikit-Learn
• Random Forest
• Joblib
"""
        )

    with t3:

        st.success(
            """
Visualization

• Streamlit
• Plotly
• Power BI
"""
        )

    st.write("")

    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.subheader("Model Performance")

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "Accuracy",
            "75.20%"
        )

        st.metric(
            "Precision",
            "52.22%"
        )

    with m2:

        st.metric(
            "Recall",
            "78.61%"
        )

        st.metric(
            "F1 Score",
            "62.75%"
        )

    with m3:

        st.metric(
            "ROC-AUC",
            "83.60%"
        )

        st.metric(
            "Threshold",
            "59.05%"
        )

    st.write("")

    # ========================================================
    # CROSS VALIDATION
    # ========================================================

    st.subheader("Cross Validation")

    cv1, cv2 = st.columns(2)

    with cv1:

        st.metric(
            "Mean ROC-AUC",
            "84.40%"
        )

    with cv2:

        st.metric(
            "Standard Deviation",
            "1.29%"
        )

    st.write("")

    # ========================================================
    # PROJECT LINKS
    # ========================================================

    st.subheader("Project Links")

    st.info(
        """
GitHub Repository:
https://github.com/arun-69tech/Customer-Churn-Prediction

Live Streamlit App:
https://telco-churn-prediction-model.streamlit.app/

"""
    )

    st.write("")

    # ========================================================
    # DEVELOPER INFORMATION
    # ========================================================

    st.subheader("Developed By")

    st.markdown(
        """
<div class="feature-card">

Arun Sharma

Data Analyst | Machine Learning Enthusiast

Customer Churn Intelligence Platform

Focused on building intelligent, business-driven analytical solutions that transform data into decisions.

</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    # ========================================================
    # CERTIFICATION
    # ========================================================

    st.subheader("Project Highlights")

    st.success(
        """
✓ End-to-End Machine Learning Project

✓ Production Ready Streamlit Deployment

✓ Integrated Business Intelligence Dashboard

✓ Interactive Risk Visualization

✓ Recruiter Portfolio Ready
"""
    )

# ============================================================
# PREDICT CHURN PAGE
# ============================================================

elif page == "Predict Churn":

    st.markdown(
        f"""
        <div class="hero-card" style="display: flex; align-items: center; margin-bottom: 25px;">
            {icon_html("brain-circuit.png", 65, 20)}
            <div>
                <h1 style="margin: 0; padding-bottom: 5px;">Predict Customer Churn</h1>
                <p style="margin: 0; color: var(--text-color); opacity: 0.8; font-size: 1.1em;">Predict customer churn risk using AI-powered analytics.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # ========================================================
    # CUSTOMER INFORMATION
    # ========================================================

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        partner = st.selectbox(
            "Partner",
            ["No", "Yes"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["No", "Yes"]
        )

        tenure = st.slider(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

    with col2:

        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

        estimated_total = round(
            monthly * tenure,
            2
        )

        st.metric(
            "Estimated Total Charges",
            f"₹ {estimated_total:,.2f}"
        )

        total = estimated_total

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["No", "Yes"]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check"
            ]
        )

    st.write("")

    # ========================================================
    # SERVICE INFORMATION
    # ========================================================

    st.subheader("Service Information")

    col3, col4 = st.columns(2)

    with col3:

        phone_service = st.selectbox(
            "Phone Service",
            ["No", "Yes"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No",
                "Yes",
                "No phone service"
            ]
        )

        internet = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        online_security = st.selectbox(
            "Online Security",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    with col4:

        online_backup = st.selectbox(
            "Online Backup",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        device_protection = st.selectbox(
            "Device Protection",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    st.write("")

    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    btn1, btn2, btn3 = st.columns([1, 2, 1])

    with btn2:

        predict_button = st.button(
            "Predict Churn",
            use_container_width=True
        )

# ============================================================
# PREDICTION ENGINE
# ============================================================

if page == "Predict Churn" and predict_button:

    with st.spinner("Generating prediction..."):

        # Initialize all features
        data = {
            feature: 0
            for feature in feature_names
        }

        # Numerical Features
        data["SeniorCitizen"] = (
            1 if senior == "Yes" else 0
        )

        data["tenure"] = tenure
        data["MonthlyCharges"] = monthly
        data["TotalCharges"] = total

        # Customer Information
        if gender == "Male":
            data["gender_Male"] = 1

        if partner == "Yes":
            data["Partner_Yes"] = 1

        if dependents == "Yes":
            data["Dependents_Yes"] = 1

        # Phone Service
        if phone_service == "Yes":
            data["PhoneService_Yes"] = 1

        if multiple_lines == "Yes":
            data["MultipleLines_Yes"] = 1

        elif multiple_lines == "No phone service":
            data[
                "MultipleLines_No phone service"
            ] = 1

        # Internet Service
        if internet == "Fiber optic":
            data[
                "InternetService_Fiber optic"
            ] = 1

        elif internet == "No":
            data[
                "InternetService_No"
            ] = 1

        # Online Security
        if online_security == "Yes":
            data[
                "OnlineSecurity_Yes"
            ] = 1

        elif online_security == "No internet service":
            data[
                "OnlineSecurity_No internet service"
            ] = 1

        # Online Backup
        if online_backup == "Yes":
            data[
                "OnlineBackup_Yes"
            ] = 1

        elif online_backup == "No internet service":
            data[
                "OnlineBackup_No internet service"
            ] = 1

        # Device Protection
        if device_protection == "Yes":
            data[
                "DeviceProtection_Yes"
            ] = 1

        elif device_protection == "No internet service":
            data[
                "DeviceProtection_No internet service"
            ] = 1

        # Tech Support
        if tech_support == "Yes":
            data[
                "TechSupport_Yes"
            ] = 1

        elif tech_support == "No internet service":
            data[
                "TechSupport_No internet service"
            ] = 1

        # Streaming TV
        if streaming_tv == "Yes":
            data[
                "StreamingTV_Yes"
            ] = 1

        elif streaming_tv == "No internet service":
            data[
                "StreamingTV_No internet service"
            ] = 1

        # Streaming Movies
        if streaming_movies == "Yes":
            data[
                "StreamingMovies_Yes"
            ] = 1

        elif streaming_movies == "No internet service":
            data[
                "StreamingMovies_No internet service"
            ] = 1

        # Contract
        if contract == "One year":
            data[
                "Contract_One year"
            ] = 1

        elif contract == "Two year":
            data[
                "Contract_Two year"
            ] = 1

        # Paperless Billing
        if paperless == "Yes":
            data[
                "PaperlessBilling_Yes"
            ] = 1

        # Payment Method
        if payment == "Credit card (automatic)":
            data[
                "PaymentMethod_Credit card (automatic)"
            ] = 1

        elif payment == "Electronic check":
            data[
                "PaymentMethod_Electronic check"
            ] = 1

        elif payment == "Mailed check":
            data[
                "PaymentMethod_Mailed check"
            ] = 1

        input_df = pd.DataFrame([data])

        input_df = input_df[
            feature_names
        ]

        # ========================================================
        # PREDICTION
        # ========================================================

        probability = model.predict_proba(input_df)[0][1]

        prediction = probability >= THRESHOLD

        probability_percent = probability * 100

        st.write("")
        st.markdown("---")

        result_col1, result_col2 = st.columns([1, 12])

        with result_col1:
            icon("sparkles.png", 40)

        with result_col2:
            st.header("Prediction Result")

        st.write("")

        # ========================================================
        # RESULT CARDS
        # ========================================================

        card1, card2, card3 = st.columns(3)

        with card1:

            if prediction:

                icon("triangle-alert.png", 45)

                st.error(
                    "Likely to Churn"
                )

            else:

                icon("badge-check.png", 45)

                st.success(
                    "Likely to Stay"
                )

        with card2:

            st.metric(
                "Churn Probability",
                f"{probability_percent:.2f}%"
            )

        with card3:

            if probability >= 0.80:

                icon("flame.png", 45)

                st.error(
                    "HIGH RISK"
                )

                risk_level = "High"

            elif probability >= THRESHOLD:

                icon("alert-circle.png", 45)

                st.warning(
                    "MEDIUM RISK"
                )

                risk_level = "Medium"

            else:

                icon("badge-check.png", 45)

                st.success(
                    "LOW RISK"
                )

                risk_level = "Low"

        st.write("")

        # ========================================================
        # RISK METER
        # ========================================================

        meter1, meter2 = st.columns([1, 12])

        with meter1:
            icon("target.png", 35)

        with meter2:
            st.subheader("Risk Meter")

        gauge = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=probability_percent,

                number={
                    "suffix": "%"
                },

                title={
                    "text": "Churn Risk"
                },

                gauge={

                    "axis": {
                        "range": [0, 100]
                    },

                    "steps": [

                        {
                            "range": [
                                0,
                                THRESHOLD * 100
                            ],
                            "color": "#10B981"
                        },

                        {
                            "range": [
                                THRESHOLD * 100,
                                80
                            ],
                            "color": "#FBBF24"
                        },

                        {
                            "range": [
                                80,
                                100
                            ],
                            "color": "#FB7185"
                        },
                    ],

                    "threshold": {

                        "line": {
                            "color": "#EF4444",
                            "width": 4
                        },

                        "value": THRESHOLD * 100,
                    },
                },
            )
        )

        gauge.update_layout(
            height=350,
            margin=dict(
                l=20,
                r=20,
                t=40,
                b=20
            )
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        st.write("")

        # ========================================================
        # POTENTIAL CHURN DRIVERS
        # ========================================================

        driver1, driver2 = st.columns([1, 12])

        with driver1:
            icon("lightbulb.png", 35)

        with driver2:
            st.subheader(
                "Potential Churn Drivers"
            )

        drivers = []

        if contract == "Month-to-month":

            drivers.append(
                "Month-to-month contract"
            )

        if monthly >= 80:

            drivers.append(
                "High monthly charges"
            )

        if tenure <= 12:

            drivers.append(
                "Low customer tenure"
            )

        if tech_support == "No":

            drivers.append(
                "No Tech Support"
            )

        if online_security == "No":

            drivers.append(
                "No Online Security"
            )

        if internet == "Fiber optic":

            drivers.append(
                "Fiber optic subscription"
            )

        if len(drivers) == 0:

            st.success(
                "No major churn drivers detected based on selected inputs."
            )

        else:

            for driver in drivers:

                st.info(
                    f"• {driver}"
                )

        st.write("")

        # ========================================================
        # MODEL INFORMATION
        # ========================================================

        with st.expander(
            "Model Information"
        ):

            st.write(
                "**Best Model:** Random Forest"
            )

            st.write(
                "**Accuracy:** 75.20%"
            )

            st.write(
                "**Precision:** 52.22%"
            )

            st.write(
                "**Recall:** 78.61%"
            )

            st.write(
                "**F1 Score:** 62.75%"
            )

            st.write(
                "**ROC-AUC:** 83.60%"
            )

            st.write(
                "**Cross Validation Mean ROC-AUC:** 84.40%"
            )

            st.write(
                "**Cross Validation Std:** 1.29%"
            )

            st.write(
                "**Deployment Threshold:** 59.05%"
            )

            st.write(
                "**Number of Features:** 30"
            )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.info(
    """
⚠️ Disclaimer:
This prediction is intended to support business decision-making and should not be used as the sole basis for customer retention actions.
"""
)

st.markdown(
    """
    <div class='footer'>

        Customer Churn Intelligence Platform
        Designed & Developed by Arun Sharma
        Data Analyst | Machine Learning Enthusiast
        
        Technologies Used:
        Python • Scikit-Learn • Random Forest • Streamlit • Power BI
        © 2026 Arun Sharma. All Rights Reserved.

    </div>
    """,
    unsafe_allow_html=True,
)