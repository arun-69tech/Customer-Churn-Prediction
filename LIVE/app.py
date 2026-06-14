from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CONSTANTS
# ==========================================================

THRESHOLD = 0.5905

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = PROJECT_ROOT / "Models"

# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    return model, feature_names

model, feature_names = load_artifacts()

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.hero-box {
    background-color: #0E1117;
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
    border: 1px solid #31333F;
}

.metric-card {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #31333F;
    text-align: center;
}

.footer {
    text-align: center;
    color: gray;
    padding-top: 20px;
    padding-bottom: 20px;
}

.section-title {
    padding-top: 10px;
    padding-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "🏠 Home",
        "🔮 Predict Churn",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
**Best Model:** Random Forest

ROC-AUC: 83.60%

Recall: 78.61%

Threshold: 59.05%
"""
)

# ==========================================================
# HOME PAGE
# ==========================================================

if page == "🏠 Home":

    st.markdown("""
    <div class='hero-box'>
        <h1>📊 Customer Churn Prediction System</h1>

Predict whether a telecom customer is likely to churn using Machine Learning.

Technologies Used:
- Python
- Scikit-Learn
- Random Forest
- Streamlit
- Power BI

Built by Arun Sharma
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("ROC-AUC", "83.60%")

    with col2:
        st.metric("Recall", "78.61%")

    with col3:
        st.metric("Features", "30")

    with col4:
        st.metric("Threshold", "59.05%")

    st.markdown("---")

    st.subheader("🎯 Project Objective")

    st.write("""
This application predicts whether a telecom customer is likely to churn.
The model helps businesses identify high-risk customers and take proactive
retention actions before customers leave.
""")

    st.subheader("🚀 Key Features")

    st.write("""
- Predict customer churn risk.
- Display churn probability.
- Categorize customers into risk levels.
- Powered by a trained Random Forest model.
- Integrated with Power BI dashboards.
""")

# ==========================================================
# ABOUT PAGE
# ==========================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.write("""
### Dataset
IBM Telco Customer Churn Dataset

### Objective
Predict customer churn and support customer retention strategies.

### Technologies Used

- Python
- Pandas
- Scikit-Learn
- Random Forest
- Streamlit
- Power BI

### Model Performance

- Accuracy: 75.20%
- Precision: 52.22%
- Recall: 78.61%
- F1 Score: 62.75%
- ROC-AUC: 83.60%

### Cross Validation

- Mean ROC-AUC: 84.40%
- Standard Deviation: 1.29%

### Built By

**Arun Sharma**
""")

# ==========================================================
# PREDICTION PAGE
# ==========================================================

elif page == "🔮 Predict Churn":

    st.title("🔮 Predict Customer Churn")

    st.write(
        "Enter customer information below and click Predict."
    )

    st.markdown("## Customer Information")

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

        estimated_total = round(monthly * tenure, 2)

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

    st.markdown("## Services")

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

    predict_button = st.button(
        "🚀 Predict Churn",
        use_container_width=True
    )

# ==========================================================
# PREDICTION ENGINE
# ==========================================================

if page == "🔮 Predict Churn" and predict_button:

    with st.spinner("🔮 Generating prediction..."):

        # Initialize all features with 0
        data = {feature: 0 for feature in feature_names}

        # Numerical Features
        data["SeniorCitizen"] = 1 if senior == "Yes" else 0
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

        # Phone Services
        if phone_service == "Yes":
            data["PhoneService_Yes"] = 1

        if multiple_lines == "Yes":
            data["MultipleLines_Yes"] = 1
        elif multiple_lines == "No phone service":
            data["MultipleLines_No phone service"] = 1

        # Internet Service
        if internet == "Fiber optic":
            data["InternetService_Fiber optic"] = 1
        elif internet == "No":
            data["InternetService_No"] = 1

        # Online Security
        if online_security == "Yes":
            data["OnlineSecurity_Yes"] = 1
        elif online_security == "No internet service":
            data["OnlineSecurity_No internet service"] = 1

        # Online Backup
        if online_backup == "Yes":
            data["OnlineBackup_Yes"] = 1
        elif online_backup == "No internet service":
            data["OnlineBackup_No internet service"] = 1

        # Device Protection
        if device_protection == "Yes":
            data["DeviceProtection_Yes"] = 1
        elif device_protection == "No internet service":
            data["DeviceProtection_No internet service"] = 1

        # Tech Support
        if tech_support == "Yes":
            data["TechSupport_Yes"] = 1
        elif tech_support == "No internet service":
            data["TechSupport_No internet service"] = 1

        # Streaming TV
        if streaming_tv == "Yes":
            data["StreamingTV_Yes"] = 1
        elif streaming_tv == "No internet service":
            data["StreamingTV_No internet service"] = 1

        # Streaming Movies
        if streaming_movies == "Yes":
            data["StreamingMovies_Yes"] = 1
        elif streaming_movies == "No internet service":
            data["StreamingMovies_No internet service"] = 1

        # Contract
        if contract == "One year":
            data["Contract_One year"] = 1
        elif contract == "Two year":
            data["Contract_Two year"] = 1

        # Paperless Billing
        if paperless == "Yes":
            data["PaperlessBilling_Yes"] = 1

        # Payment Method
        if payment == "Credit card (automatic)":
            data["PaymentMethod_Credit card (automatic)"] = 1
        elif payment == "Electronic check":
            data["PaymentMethod_Electronic check"] = 1
        elif payment == "Mailed check":
            data["PaymentMethod_Mailed check"] = 1

        # Create DataFrame
        input_df = pd.DataFrame([data])

        # Ensure correct feature order
        input_df = input_df[feature_names]

        # ==========================================================
        # PREDICTION
        # ==========================================================

        probability = model.predict_proba(input_df)[0][1]

        prediction = probability >= THRESHOLD

        probability_percent = probability * 100

        st.markdown("---")

        st.header("📊 Prediction Result")

        # ==========================================================
        # RESULT CARDS
        # ==========================================================

        col1, col2, col3 = st.columns(3)

        with col1:

            if prediction:
                st.error("⚠️ Likely to Churn")
            else:
                st.success("✅ Likely to Stay")

        with col2:

            st.metric(
                "Churn Probability",
                f"{probability_percent:.2f}%"
            )

        with col3:

            if probability >= 0.80:
                st.error("🔴 HIGH RISK")
                risk_level = "High"

            elif probability >= THRESHOLD:
                st.warning("🟡 MEDIUM RISK")
                risk_level = "Medium"

            else:
                st.success("🟢 LOW RISK")
                risk_level = "Low"

        # ==========================================================
        # GAUGE CHART
        # ==========================================================

        st.subheader("🎯 Risk Meter")

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability_percent,
                number={"suffix": "%"},
                title={"text": "Churn Risk"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, THRESHOLD * 100], "color": "lightgreen"},
                        {"range": [THRESHOLD * 100, 80], "color": "gold"},
                        {"range": [80, 100], "color": "tomato"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "value": THRESHOLD * 100,
                    },
                },
            )
        )

        gauge.update_layout(height=350)

        st.plotly_chart(gauge, use_container_width=True)

        # ==========================================================
        # TOP CHURN DRIVERS
        # ==========================================================

        st.subheader("💡 Potential Churn Drivers")

        drivers = []

        if contract == "Month-to-month":
            drivers.append("• Month-to-month contract")

        if monthly >= 80:
            drivers.append("• High monthly charges")

        if tenure <= 12:
            drivers.append("• Low customer tenure")

        if tech_support == "No":
            drivers.append("• No Tech Support")

        if online_security == "No":
            drivers.append("• No Online Security")

        if internet == "Fiber optic":
            drivers.append("• Fiber optic subscription")

        if len(drivers) == 0:
            st.success(
                "No major churn drivers detected based on selected inputs."
            )
        else:
            for driver in drivers:
                st.write(driver)

        # ==========================================================
        # MODEL INFORMATION
        # ==========================================================

        with st.expander("📈 Model Information"):

            st.write("**Best Model:** Random Forest")
            st.write("**Accuracy:** 75.20%")
            st.write("**Precision:** 52.22%")
            st.write("**Recall:** 78.61%")
            st.write("**F1 Score:** 62.75%")
            st.write("**ROC-AUC:** 83.60%")
            st.write("**Cross Validation Mean ROC-AUC:** 84.40%")
            st.write("**Cross Validation Std:** 1.29%")
            st.write("**Deployment Threshold:** 59.05%")
            st.write("**Number of Features:** 30")

# ==========================================================
# FOOTER
# ==========================================================
st.info(
    """
⚠️ Disclaimer:
This prediction is intended to support business decision-making and should
not be used as the sole basis for customer retention actions.
"""
)
st.markdown("---")

st.markdown(
    """
    <div class='footer'>
        <h4>📊 Customer Churn Prediction System</h4>

        Built by Arun Sharma

        <br><br>

        Dataset: IBM Telco Customer Churn Dataset

        <br>

        Technologies Used:
        Python • Scikit-Learn • Random Forest • Streamlit • Power BI

        <br><br>

        © 2026 Arun Sharma. All Rights Reserved.
    </div>
    """,
    unsafe_allow_html=True
)