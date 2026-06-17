import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Credit Risk AI",
    layout="wide"
)

# ---------------- CLEAN DARK DESIGN ----------------
st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #0b0f19;
    color: #e6edf3;
    font-size: 18px;
}

.stApp {
    background-color: #0b0f19;
    color: #e6edf3;
}

/* Titles */
h1, h2, h3 {
    color: #4cc9f0;
    font-weight: 700;
}

/* Input labels */
label {
    color: #c9d1d9 !important;
    font-size: 16px !important;
}

/* Input boxes */
.stNumberInput input, .stSelectbox div {
    background-color: #161b22 !important;
    color: #e6edf3 !important;
    border-radius: 8px;
    font-size: 16px !important;
}

/* Button */
.stButton > button {
    background-color: #4cc9f0;
    color: black;
    font-size: 18px;
    padding: 10px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

st.title("🏦 Credit Risk Assessment System")

st.write("AI-powered system for evaluating loan risk")

st.markdown("---")

# ---------------- INPUTS (SAFE + VALIDATED) ----------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 Financial Information")

    account_balance = st.selectbox(
        "Account Balance (bank status)",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1 - No account",
            2: "2 - Low balance",
            3: "3 - Medium balance",
            4: "4 - High balance"
        }[x],
        help="Customer's bank account status"
    )

    credit_amount = st.number_input(
        "Credit Amount (€)",
        min_value=100,
        max_value=20000,
        step=100,
        help="Loan amount requested (100 - 20000)"
    )

    savings = st.selectbox(
        "Savings / Stocks",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1 - Low savings",
            2: "2 - Medium savings",
            3: "3 - High savings",
            4: "4 - Very high savings"
        }[x],
        help="Customer savings level"
    )

    duration = st.slider(
        "Credit Duration (months)",
        min_value=6,
        max_value=72,
        step=6,
        help="Loan duration (6 - 72 months)"
    )

with col2:
    st.subheader("👤 Personal Information")

    age = st.slider(
        "Age",
        min_value=18,
        max_value=75,
        help="Customer age"
    )

    employment = st.selectbox(
        "Employment Length",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: {
            1: "< 1 year",
            2: "1-4 years",
            3: "4-7 years",
            4: "7+ years",
            5: "Stable employment"
        }[x],
        help="Work experience level"
    )

    apartment = st.selectbox(
        "Housing Type",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "Free housing",
            2: "Rent",
            3: "Own house"
        }[x],
        help="Living situation"
    )

    foreign_worker = st.selectbox(
        "Foreign Worker",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Whether customer is foreign worker"
    )

    dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=5,
        help="Family dependents (0-5)"
    )

# ---------------- PREDICTION ----------------

st.markdown("---")

if st.button("🔮 Predict Risk", use_container_width=True):

    input_data = pd.DataFrame([{
        "Account_Balance": account_balance,
        "Duration_of_Credit_monthly": duration,
        "Payment_Status_of_Previous_Credit": 0,
        "Purpose": 0,
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

    continuous_cols = [
        "Credit_Amount",
        "Age_years",
        "Duration_of_Credit_monthly"
    ]

    input_data[continuous_cols] = scaler.transform(input_data[continuous_cols])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.markdown("## 📊 Result")

    colA, colB = st.columns(2)

    with colA:
        if prediction == 1:
            st.success("✅ GOOD CREDIT RISK")
        else:
            st.error("❌ BAD CREDIT RISK")

    with colB:
        st.metric("Good Risk Probability", f"{prob:.2%}")

    st.progress(float(prob))

    # ---------------- SIMPLE VISUAL ----------------
    st.markdown("### 📈 Risk Breakdown")

    fig, ax = plt.subplots()
    ax.bar(["Bad Risk", "Good Risk"], [1-prob, prob], color=["#ff4b4b", "#4caf50"])
    ax.set_ylim(0, 1)
    st.pyplot(fig)