import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ===================== LOAD MODEL =====================
model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Credit Risk System",
    layout="wide"
)

# ===================== MINIMAL STYLE =====================
st.markdown("""
<style>
html, body {
    background-color: #0e1117;
    color: #e6e6e6;
    font-size: 16px;
}

.stApp {
    background-color: #0e1117;
    color: #e6e6e6;
}

h1, h2, h3 {
    color: #e6e6e6;
    font-weight: 600;
}

label {
    color: #cfcfcf !important;
    font-size: 14px !important;
}

.stButton > button {
    background-color: #1f1f1f;
    color: white;
    border: 1px solid #333;
    padding: 10px;
    border-radius: 6px;
}

.stButton > button:hover {
    background-color: #2a2a2a;
}
</style>
""", unsafe_allow_html=True)

# ===================== TITLE =====================
st.title("🏦 Credit Risk Assessment System")
st.write("Enter applicant details to evaluate credit risk using ML model")

st.markdown("---")

# ===================== INPUT FORM =====================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Financial Information")

    account_balance = st.selectbox(
        "Account Balance",
        [1, 2, 3, 4],
        help="1 = No account, 2 = Low, 3 = Medium, 4 = High balance"
    )

    credit_amount = st.selectbox(
        "Credit Amount (€)",
        [1000, 2000, 5000, 10000, 15000, 20000],
        help="Select loan amount (fixed allowed values)"
    )

    savings = st.selectbox(
        "Savings / Stocks",
        [1, 2, 3, 4],
        help="1 = Low savings, 4 = Very high savings"
    )

    duration = st.selectbox(
        "Credit Duration (months)",
        [6, 12, 24, 36, 48, 60, 72],
        help="Loan duration in months"
    )

    purpose = st.selectbox(
        "Loan Purpose",
        [0, 1, 2, 3, 4],
        help="Encoded category: e.g. car, education, business etc."
    )

with col2:
    st.subheader("Personal Information")

    age = st.selectbox(
        "Age",
        list(range(18, 76)),
        help="Applicant age (18–75)"
    )

    employment = st.selectbox(
        "Employment Level",
        [1, 2, 3, 4, 5],
        help="1 = unemployed, 5 = stable long-term job"
    )

    apartment = st.selectbox(
        "Housing Type",
        [1, 2, 3],
        help="1 = free, 2 = rent, 3 = own house"
    )

    dependents = st.selectbox(
        "Number of Dependents",
        [0, 1, 2, 3, 4, 5],
        help="Family dependents"
    )

    foreign_worker = st.selectbox(
        "Foreign Worker",
        [0, 1],
        help="0 = No, 1 = Yes"
    )

# ===================== PREDICTION =====================

st.markdown("---")

if st.button("🔍 Evaluate Credit Risk"):

    input_data = pd.DataFrame([{
        "Account_Balance": account_balance,
        "Duration_of_Credit_monthly": duration,
        "Payment_Status_of_Previous_Credit": 0,
        "Purpose": purpose,
        "Credit_Amount": credit_amount,
        "Value_Savings_Stocks": savings,
        "Length_of_current_employment": employment,
        "Instalment_per_cent": 0,
        "Sex_Marital_Status": 0,
        "Guarantors": 0,
        "Duration_in_Current_address": 0,
        "Most_valuable_available_asset": 0,
        "Age_years": age,
        "Concurrent_Credits": 0,
        "Type_of_apartment": apartment,
        "No_of_Credits_at_this_Bank": 0,
        "Occupation": 0,
        "No_of_dependents": dependents,
        "Telephone": 0,
        "Foreign_Worker": foreign_worker
    }])

    # scale numeric features
    num_cols = [
        "Credit_Amount",
        "Age_years",
        "Duration_of_Credit_monthly"
    ]

    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # prediction
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # ===================== RESULT =====================
    st.markdown("## 📊 Result")

    if pred == 1:
        st.success("✔ GOOD CREDIT RISK")
    else:
        st.error("✖ BAD CREDIT RISK")

    st.write(f"Probability of Good Risk: **{prob:.2%}**")

    st.progress(float(prob))

    # ===================== CHARTS =====================
    st.markdown("## 📈 Analysis Dashboard")

    colA, colB = st.columns(2)

    with colA:
        fig, ax = plt.subplots()
        ax.bar(
            ["Bad Risk", "Good Risk"],
            [1 - prob, prob],
            color=["#444444", "#dddddd"]
        )
        ax.set_ylim(0, 1)
        ax.set_title("Risk Probability Distribution")
        st.pyplot(fig)

    with colB:
        fig, ax = plt.subplots()
        ax.pie(
            [prob, 1 - prob],
            labels=["Good", "Bad"],
            autopct="%1.1f%%",
            colors=["#dddddd", "#444444"]
        )
        ax.set_title("Risk Share")
        st.pyplot(fig)

    # ===================== SIMPLE INSIGHT =====================
    st.markdown("### 📌 Insight")

    if prob > 0.7:
        st.info("Low risk applicant — likely to repay loan.")
    elif prob > 0.4:
        st.warning("Medium risk applicant — review recommended.")
    else:
        st.error("High risk applicant — loan not recommended.")