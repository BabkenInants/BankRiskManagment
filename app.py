import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ===================== LOAD MODEL =====================
model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Credit Risk Assessment",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===================== STYLE =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #080c14; }

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1400px;
}

.header-banner {
    background: linear-gradient(135deg, #0d1b2e 0%, #0f2744 50%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.header-icon { font-size: 2.8rem; line-height: 1; }
.header-title {
    font-size: 1.9rem; font-weight: 700; color: #e8f0fe;
    letter-spacing: -0.02em; margin: 0;
}
.header-sub {
    font-size: 0.875rem; color: #6b8bb5;
    margin: 0.25rem 0 0 0; font-weight: 400;
}

.section-card {
    background: #0d1520;
    border: 1px solid #1a2d45;
    border-radius: 12px;
    padding: 1.5rem 1.5rem 0.75rem 1.5rem;
    margin-bottom: 1rem;
}
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3d7cc9;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #1a2d45;
}

div[data-baseweb="select"] > div {
    background-color: #0a1220 !important;
    border-color: #1e3a5f !important;
    border-radius: 8px !important;
    color: #c8d8f0 !important;
}
div[data-baseweb="popover"] { background-color: #0d1520 !important; border: 1px solid #1e3a5f !important; }
li[role="option"] { background-color: #0d1520 !important; color: #c8d8f0 !important; }
li[role="option"]:hover { background-color: #1a2d45 !important; }

label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color: #7a9ec4 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}

input[type="number"] {
    background-color: #0a1220 !important;
    color: #c8d8f0 !important;
    border-color: #1e3a5f !important;
}

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1a4a8a, #2563b0) !important;
    color: #ffffff !important;
    border: none !important;
    padding: 0.75rem 2.5rem !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(37,99,176,0.3) !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #1e56a0, #2d74d0) !important;
    box-shadow: 0 6px 20px rgba(37,99,176,0.5) !important;
}

.result-good {
    background: linear-gradient(135deg, #0a2a1a, #0d3320);
    border: 1px solid #1a6640;
    border-left: 4px solid #22c55e;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.result-bad {
    background: linear-gradient(135deg, #2a0a0a, #330d0d);
    border: 1px solid #66201a;
    border-left: 4px solid #ef4444;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.result-verdict { font-size: 0.68rem; color: #6b8bb5; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; margin: 0; }
.result-title { font-size: 1.4rem; font-weight: 700; margin: 0.3rem 0; }
.result-good .result-title { color: #22c55e; }
.result-bad .result-title { color: #ef4444; }
.result-prob { font-family: 'JetBrains Mono', monospace; font-size: 2.6rem; font-weight: 500; margin: 0.25rem 0 0 0; }
.result-good .result-prob { color: #4ade80; }
.result-bad .result-prob { color: #f87171; }
.result-prob-label { font-size: 0.68rem; color: #6b8bb5; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 500; }

.metric-row { display: flex; gap: 0.75rem; margin: 0.75rem 0; }
.metric-chip {
    background: #0d1520; border: 1px solid #1a2d45; border-radius: 8px;
    padding: 0.6rem 1rem; flex: 1;
}
.mc-label { font-size: 0.62rem; color: #4d6a8a; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }
.mc-value { font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; color: #c8d8f0; font-weight: 500; margin-top: 0.1rem; }

.insight-box {
    background: #0d1520; border: 1px solid #1a2d45; border-radius: 10px;
    padding: 0.9rem 1.1rem; margin-top: 0.75rem;
    font-size: 0.85rem; color: #8ba8cc; line-height: 1.6;
}

.styled-divider { border: none; border-top: 1px solid #1a2d45; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ===================== HEADER =====================
st.markdown("""
<div class="header-banner">
    <div class="header-icon">🏦</div>
    <div>
        <p class="header-title">Credit Risk Assessment</p>
        <p class="header-sub">ML-powered loan eligibility evaluation · German Credit Dataset</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ===================== 3-COLUMN FORM =====================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-card"><div class="section-label">💳 Loan Details</div>', unsafe_allow_html=True)

    account_balance = st.selectbox("Account Balance", [1, 2, 3, 4],
        format_func=lambda x: {
            1: "No account / Overdrawn",
            2: "Low balance  (< 200 €)",
            3: "Medium balance  (≥ 200 €)",
            4: "No prior credit / Paid back"
        }[x],
        help="The current status of the applicant's checking account. Applicants with no account or a negative balance are considered higher risk.")

    credit_amount = st.number_input("Credit Amount (€)", min_value=250, max_value=20000, value=2500, step=250,
        help="The total loan amount requested in euros. Higher amounts relative to income and savings increase the risk score.")

    duration = st.slider("Duration (months)", min_value=6, max_value=72, value=24, step=6,
        help="The repayment period of the loan in months. Longer durations increase exposure and generally correlate with higher risk.")

    purpose = st.selectbox("Loan Purpose", [0, 1, 2, 3, 4, 5, 6, 8, 9, 10],
        format_func=lambda x: {
            0: "Car (new)", 1: "Car (used)", 2: "Furniture / Equipment",
            3: "Radio / Television", 4: "Domestic Appliances",
            5: "Repairs", 6: "Education", 8: "Retraining",
            9: "Business", 10: "Other"
        }[x],
        help="The stated reason for taking the loan. Business and education loans tend to carry different risk profiles than consumer goods purchases.")

    instalment_pct = st.selectbox("Instalment Rate (% of income)", [1, 2, 3, 4],
        format_func=lambda x: {1: "< 20%", 2: "20–25%", 3: "25–35%", 4: "≥ 35%"}[x],
        help="Monthly loan repayment as a percentage of the applicant's disposable income. Higher percentages indicate the applicant is more financially stretched.")

    payment_status = st.selectbox("Previous Credit Status", [0, 1, 2, 3, 4],
        format_func=lambda x: {
            0: "Paid duly — no issues",
            1: "All credits paid back duly",
            2: "Existing credits paid duly",
            3: "Delay in paying off",
            4: "Critical account"
        }[x],
        help="How the applicant has handled past credit obligations. A history of delays or critical accounts is a strong negative signal for the model.")

    savings = st.selectbox("Savings / Stocks", [1, 2, 3, 4, 5],
        format_func=lambda x: {
            1: "< 100 €", 2: "100–500 €", 3: "500–1000 €",
            4: "≥ 1000 €", 5: "Unknown / No savings"
        }[x],
        help="The total value of the applicant's savings accounts and stock holdings. Higher savings act as a financial buffer and reduce predicted risk.")

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card"><div class="section-label">👤 Personal Profile</div>', unsafe_allow_html=True)

    age = st.slider("Age", min_value=18, max_value=75, value=35,
        help="The applicant's age in years. Age is used as a proxy for financial stability and career stage.")

    sex_marital = st.selectbox("Sex / Marital Status", [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Male — Divorced / Separated",
            2: "Female — Non-single  /  Male Single",
            3: "Male — Married / Widowed",
            4: "Female — Single"
        }[x],
        help="Combined sex and marital status category as defined in the original German Credit Dataset encoding.")

    employment = st.selectbox("Length of Employment", [1, 2, 3, 4, 5],
        format_func=lambda x: {
            1: "Unemployed", 2: "< 1 year",
            3: "1–4 years", 4: "4–7 years", 5: "≥ 7 years"
        }[x],
        help="How long the applicant has been at their current job. Longer employment duration signals income stability and reduces predicted risk.")

    occupation = st.selectbox("Occupation", [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Unskilled — Non-resident",
            2: "Unskilled — Resident",
            3: "Skilled / Official",
            4: "Management / Highly Qualified"
        }[x],
        help="The applicant's occupational category. Higher skill levels generally correlate with more stable income and lower default risk.")

    dependents = st.selectbox("Number of Dependents", [1, 2],
        format_func=lambda x: "3 or more" if x == 1 else "0 to 2",
        help="The number of people financially dependent on the applicant (e.g. children, non-working spouse). More dependents reduce disposable income.")

    telephone = st.selectbox("Telephone", [1, 2],
        format_func=lambda x: "Not registered" if x == 1 else "Yes — Registered",
        help="Whether the applicant has a registered telephone number. Used in the original dataset as a minor indicator of social stability.")

    foreign_worker = st.selectbox("Foreign Worker", [1, 2],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Whether the applicant is classified as a foreign worker. Included as a demographic feature in the original dataset.")

    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-card"><div class="section-label">🏠 Assets & Housing</div>', unsafe_allow_html=True)

    apartment = st.selectbox("Housing Type", [1, 2, 3],
        format_func=lambda x: {
            1: "Free (with parents / employer)",
            2: "Renting",
            3: "Own property"
        }[x],
        help="The applicant's current living situation. Owning property is considered a positive asset indicator; free housing may suggest limited financial independence.")

    most_valuable_asset = st.selectbox("Most Valuable Asset", [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Real estate",
            2: "Life insurance / Savings",
            3: "Car / Other",
            4: "Unknown / None"
        }[x],
        help="The most significant asset the applicant owns. Assets serve as implicit collateral and improve the risk profile.")

    duration_address = st.selectbox("Years at Current Address", [1, 2, 3, 4],
        format_func=lambda x: {
            1: "< 1 year", 2: "1–4 years",
            3: "4–7 years", 4: "≥ 7 years"
        }[x],
        help="How long the applicant has lived at their current address. Residential stability is a minor positive indicator of reliability.")

    guarantors = st.selectbox("Guarantors", [1, 2, 3],
        format_func=lambda x: {1: "None", 2: "Co-applicant", 3: "Guarantor"}[x],
        help="Whether a third party co-signs or guarantees the loan. The presence of a guarantor reduces the lender's risk significantly.")

    concurrent_credits = st.selectbox("Concurrent Credits", [1, 2, 3],
        format_func=lambda x: {
            1: "At other banks",
            2: "At department stores",
            3: "None"
        }[x],
        help="Whether the applicant currently has active credit lines elsewhere. Multiple concurrent credits increase the total debt burden.")

    no_credits_bank = st.selectbox("Credits at This Bank", [1, 2, 3, 4],
        format_func=lambda x: {
            1: "One", 2: "Two or three",
            3: "Four or five", 4: "Six or more"
        }[x],
        help="The number of existing credit accounts the applicant holds at this bank. A high number may indicate over-reliance on credit.")

    st.markdown('</div>', unsafe_allow_html=True)

# ===================== BUTTON =====================
st.markdown("""
<style>
div[data-testid="stButton"] {
    display: flex;
    justify-content: center;
}
div[data-testid="stButton"] > button {
    width: auto !important;
    padding: 0.75rem 3rem !important;
}
</style>
""", unsafe_allow_html=True)
evaluate = st.button("🔍  Evaluate Credit Risk")

# ===================== PREDICTION =====================
if evaluate:
    input_data = pd.DataFrame([{
        "Account_Balance": account_balance,
        "Duration_of_Credit_monthly": duration,
        "Payment_Status_of_Previous_Credit": payment_status,
        "Purpose": purpose,
        "Credit_Amount": credit_amount,
        "Value_Savings_Stocks": savings,
        "Length_of_current_employment": employment,
        "Instalment_per_cent": instalment_pct,
        "Sex_Marital_Status": sex_marital,
        "Guarantors": guarantors,
        "Duration_in_Current_address": duration_address,
        "Most_valuable_available_asset": most_valuable_asset,
        "Age_years": age,
        "Concurrent_Credits": concurrent_credits,
        "Type_of_apartment": apartment,
        "No_of_Credits_at_this_Bank": no_credits_bank,
        "Occupation": occupation,
        "No_of_dependents": dependents,
        "Telephone": telephone,
        "Foreign_Worker": foreign_worker
    }])

    columns = joblib.load("columns.pkl")
    input_data = input_data[columns]

    num_cols_scale = ["Credit_Amount", "Age_years", "Duration_of_Credit_monthly"]
    input_data[num_cols_scale] = scaler.transform(input_data[num_cols_scale])

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    bad_prob = 1 - prob

    st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)

    res_col, chart_col = st.columns([1, 1.6])

    with res_col:
        if pred == 1:
            st.markdown(f"""
            <div class="result-good">
                <p class="result-verdict">Assessment Result</p>
                <p class="result-title">✔ Good Credit Risk</p>
                <p class="result-prob">{prob:.1%}</p>
                <p class="result-prob-label">Probability of Repayment</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-bad">
                <p class="result-verdict">Assessment Result</p>
                <p class="result-title">✖ Bad Credit Risk</p>
                <p class="result-prob">{bad_prob:.1%}</p>
                <p class="result-prob-label">Probability of Default</p>
            </div>
            """, unsafe_allow_html=True)

        risk_level = "Low" if prob > 0.7 else ("Medium" if prob > 0.4 else "High")
        risk_color = "#22c55e" if prob > 0.7 else ("#f59e0b" if prob > 0.4 else "#ef4444")

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-chip">
                <div class="mc-label">Good Risk</div>
                <div class="mc-value">{prob:.1%}</div>
            </div>
            <div class="metric-chip">
                <div class="mc-label">Bad Risk</div>
                <div class="mc-value">{bad_prob:.1%}</div>
            </div>
        </div>
        <div class="metric-row">
            <div class="metric-chip">
                <div class="mc-label">Risk Level</div>
                <div class="mc-value" style="color:{risk_color}">{risk_level}</div>
            </div>
            <div class="metric-chip">
                <div class="mc-label">Loan Amount</div>
                <div class="mc-value">{credit_amount:,} €</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if prob > 0.7:
            insight = "📗 Strong repayment profile. Applicant is statistically likely to service this loan without issue."
        elif prob > 0.4:
            insight = "📙 Borderline profile. Manual review recommended. Consider requesting additional collateral."
        else:
            insight = "📕 High default risk. Key risk factors may include low savings, poor account status, or long loan duration."

        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

    with chart_col:
        plt.rcParams.update({
            "figure.facecolor": "#0d1520",
            "axes.facecolor": "#0d1520",
            "axes.edgecolor": "#1a2d45",
            "axes.labelcolor": "#7a9ec4",
            "xtick.color": "#7a9ec4",
            "ytick.color": "#7a9ec4",
            "text.color": "#c8d8f0",
            "grid.color": "#1a2d45",
        })

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        fig.patch.set_facecolor("#0d1520")

        # Bar chart
        ax = axes[0]
        bars = ax.bar(
            ["Bad Risk", "Good Risk"], [bad_prob, prob],
            color=["#ef4444", "#22c55e"], width=0.5, zorder=3
        )
        ax.set_ylim(0, 1.15)
        ax.set_title("Risk Breakdown", fontsize=11, pad=10, color="#c8d8f0", fontweight="600")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", zorder=0, alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, [bad_prob, prob]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03,
                    f"{val:.1%}", ha="center", va="bottom",
                    fontsize=12, fontweight="600", color="#c8d8f0")

        # Donut chart
        ax2 = axes[1]
        wedges, texts, autotexts = ax2.pie(
            [prob, bad_prob],
            labels=["Good", "Bad"],
            autopct="%1.1f%%",
            colors=["#22c55e", "#ef4444"],
            startangle=90,
            pctdistance=0.75,
            wedgeprops={"width": 0.55, "edgecolor": "#0d1520", "linewidth": 2}
        )
        for t in texts:
            t.set_color("#7a9ec4")
            t.set_fontsize(10)
        for at in autotexts:
            at.set_color("#0d1520")
            at.set_fontsize(10)
            at.set_fontweight("700")
        ax2.set_title("Risk Distribution", fontsize=11, pad=10, color="#c8d8f0", fontweight="600")
        ax2.text(0, 0, f"{prob:.0%}\nGood", ha="center", va="center",
                 fontsize=13, fontweight="700", color="#c8d8f0", linespacing=1.4)

        plt.tight_layout(pad=1.5)
        st.pyplot(fig)
        plt.close()