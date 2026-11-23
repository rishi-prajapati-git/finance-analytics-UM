"""
Page 05: Real-Time Machine Learning Customer Churn Predictor & What-If Simulator
"""

import sys
from pathlib import Path
import streamlit as st
import plotly.express as px
import pandas as pd

# Path setup
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference import ChurnPredictor, SAMPLE_PERSONAS
from app.ui_utils import apply_custom_styles, render_hero_banner, render_gauge_chart

st.set_page_config(
    page_title="AI Churn Predictor | European Banking",
    page_icon="🤖",
    layout="wide"
)

apply_custom_styles()

render_hero_banner(
    title="🤖 Real-Time Churn Predictor & Simulator",
    subtitle="Predict account exit probability using trained XGBoost ML Pipeline and test counterfactual retention levers.",
    badge_text="Production Inference Engine"
)

@st.cache_resource
def get_predictor():
    return ChurnPredictor()

try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Error loading model pipeline: {e}. Please ensure models are trained.")
    st.stop()

# ── Persona Presets ─────────────────────────────────────────────────────────────
st.markdown("### 👤 Select Customer Persona or Customize Attributes")
selected_persona_name = st.selectbox(
    "Choose a pre-configured customer scenario to pre-fill inputs:",
    options=list(SAMPLE_PERSONAS.keys())
)

preset_values = SAMPLE_PERSONAS.get(selected_persona_name)
if preset_values is None:
    preset_values = {
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 48,
        "CreditScore": 650,
        "Tenure": 4,
        "Balance": 115000.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 95000.0
    }

# ── Input Form ──────────────────────────────────────────────────────────────────
with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🌍 Demographics")
        geo_options = ["France", "Germany", "Spain"]
        geography = st.selectbox(
            "Country",
            options=geo_options,
            index=geo_options.index(preset_values["Geography"])
        )
        gender_options = ["Female", "Male"]
        gender = st.selectbox(
            "Gender",
            options=gender_options,
            index=gender_options.index(preset_values["Gender"])
        )
        age = st.slider(
            "Customer Age",
            min_value=18,
            max_value=90,
            value=int(preset_values["Age"])
        )

    with col2:
        st.markdown("#### 💳 Financial Portfolio")
        credit_score = st.slider(
            "Credit Score",
            min_value=350,
            max_value=850,
            value=int(preset_values["CreditScore"])
        )
        balance = st.number_input(
            "Account Balance (€)",
            min_value=0.0,
            max_value=300000.0,
            value=float(preset_values["Balance"]),
            step=1000.0
        )
        salary = st.number_input(
            "Estimated Annual Salary (€)",
            min_value=10000.0,
            max_value=250000.0,
            value=float(preset_values["EstimatedSalary"]),
            step=2500.0
        )

    with col3:
        st.markdown("#### 🤝 Relationship & Engagement")
        tenure = st.slider(
            "Tenure (Years with Bank)",
            min_value=0,
            max_value=10,
            value=int(preset_values["Tenure"])
        )
        products = st.selectbox(
            "Number of Products",
            options=[1, 2, 3, 4],
            index=[1, 2, 3, 4].index(int(preset_values["NumOfProducts"]))
        )
        card_choice = st.radio(
            "Credit Card Holder?",
            options=["Yes (1)", "No (0)"],
            index=0 if preset_values["HasCrCard"] == 1 else 1,
            horizontal=True
        )
        active_choice = st.radio(
            "Active Member Status?",
            options=["Active (1)", "Inactive (0)"],
            index=0 if preset_values["IsActiveMember"] == 1 else 1,
            horizontal=True
        )

has_card = 1 if "Yes" in card_choice else 0
is_active = 1 if "Active" in active_choice else 0

customer_payload = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": products,
    "HasCrCard": has_card,
    "IsActiveMember": is_active,
    "EstimatedSalary": salary
}

st.markdown("---")

# ── Inference Execution ────────────────────────────────────────────────────────
prediction = predictor.predict(customer_payload)
churn_pct = prediction["churn_probability_pct"]

left_col, right_col = st.columns([1, 1.2])

with left_col:
    st.plotly_chart(render_gauge_chart(prediction["churn_probability"]), use_container_width=True)
    
    st.markdown(f"""
    <div style="text-align: center; margin-top: -15px; margin-bottom: 20px;">
        <span class="{ 'badge-critical' if churn_pct >= 60 else ('badge-warning' if churn_pct >= 30 else 'badge-success') }">
            {prediction['badge']}
        </span>
    </div>
    """, unsafe_allow_html=True)

with right_col:
    st.subheader("🔍 Key Risk Determinants")
    if prediction["risk_drivers"]:
        for driver in prediction["risk_drivers"]:
            st.markdown(f"⚠️ **{driver}**")
    else:
        st.markdown("✅ No elevated statistical risk factors detected.")

    st.subheader("📋 Retention Action Playbook")
    for action in prediction["recommended_actions"]:
        st.markdown(f"👉 {action}")

# ── What-If Counterfactual Sandbox ─────────────────────────────────────────────
st.markdown("---")
st.subheader("🧪 What-If Counterfactual Sandbox")
st.markdown(
    "Explore how taking proactive relationship actions (e.g. activating membership or cross-selling a second product) "
    "would quantitatively reduce this customer's predicted attrition probability."
)

col_w1, col_w2, col_w3 = st.columns(3)

with col_w1:
    sim_active = st.checkbox("Toggle to Active Member", value=(is_active == 0))
with col_w2:
    sim_products = st.selectbox("Simulate New Product Count", options=[1, 2, 3, 4], index=1)
with col_w3:
    sim_balance_change = st.select_slider(
        "Balance Adjustment Factor",
        options=["50% Drop", "Same Balance", "20% Growth"],
        value="Same Balance"
    )

multiplier = 0.5 if sim_balance_change == "50% Drop" else (1.2 if sim_balance_change == "20% Growth" else 1.0)
sim_new_balance = balance * multiplier

variations = {
    "IsActiveMember": 1 if sim_active else 0,
    "NumOfProducts": sim_products,
    "Balance": sim_new_balance
}

what_if_res = predictor.simulate_what_if(customer_payload, variations)

col_res1, col_res2, col_res3 = st.columns(3)
col_res1.metric("Original Churn Risk", f"{what_if_res['base_pct']}%")
col_res2.metric("Counterfactual Risk", f"{what_if_res['new_pct']}%")
col_res3.metric(
    "Net Risk Delta",
    f"{what_if_res['delta_pct']:+.1f}%",
    delta=f"{what_if_res['delta_pct']:+.1f}%",
    delta_color="inverse"
)

if what_if_res["reduction_achieved"]:
    st.success(f"🎯 **Success**: The simulated strategy yields a **{abs(what_if_res['delta_pct'])}% reduction** in attrition probability!")
else:
    st.warning("⚠️ The simulated scenario maintains or elevates customer churn risk.")
