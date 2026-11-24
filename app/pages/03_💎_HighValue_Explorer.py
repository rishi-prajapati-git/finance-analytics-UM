"""
Page 03: High-Value Customer Risk & Capital Exposure
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_dataset
from app.ui_utils import apply_custom_styles, render_hero_banner

st.set_page_config(page_title="High-Value Explorer", page_icon="💎", layout="wide")
apply_custom_styles()

render_hero_banner(
    title="💎 High-Value Customer Exposure & Wealth Risk",
    subtitle="Isolate premium banking accounts with elevated balances and quantify total liquidity at risk.",
    badge_text="Portfolio Wealth Risk"
)

@st.cache_data
def get_data():
    return load_dataset()

df = get_data()

with st.sidebar:
    st.markdown("### 🎛️ High-Value Parameters")
    balance_pct = st.slider(
        "Balance Threshold (Percentile)",
        min_value=50,
        max_value=95,
        value=75,
        step=5,
        help="Accounts at or above this percentile are classified as High-Value."
    )
    selected_countries = st.multiselect(
        "Geography",
        options=sorted(df['Geography'].unique().tolist()),
        default=sorted(df['Geography'].unique().tolist())
    )

threshold = df['Balance'].quantile(balance_pct / 100)
hv_df = df[
    (df['Balance'] >= threshold) &
    (df['Geography'].isin(selected_countries))
]

# ── Metric Row ────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

total_hv = len(hv_df)
churned_hv = int(hv_df['Exited'].sum())
hv_churn_rate = (hv_df['Exited'].mean() * 100) if total_hv > 0 else 0
balance_at_risk = hv_df[hv_df['Exited'] == 1]['Balance'].sum()

col1.metric("High-Value Cohort", f"{total_hv:,}", f"Balance ≥ €{threshold:,.0f}")
col2.metric("Exited Accounts", f"{churned_hv:,}")
col3.metric("Cohort Churn Rate", f"{hv_churn_rate:.1f}%", delta=f"{hv_churn_rate - 20.37:+.1f}% vs base", delta_color="inverse")
col4.metric("Capital at Risk", f"€{balance_at_risk / 1e6:.2f}M", "Total exited balance", delta_color="inverse")

st.markdown("---")

# ── Charts ────────────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Balance vs Estimated Salary (High-Value)")
    fig = px.scatter(
        hv_df,
        x='Balance',
        y='EstimatedSalary',
        color=hv_df['Exited'].map({0: 'Retained', 1: 'Churned'}),
        color_discrete_map={'Retained': '#10B981', 'Churned': '#E11D48'},
        opacity=0.75,
        hover_data=['Geography', 'Age', 'NumOfProducts']
    )
    fig.update_layout(
        paper_bgcolor='rgba(255,255,255,0)',
        plot_bgcolor='rgba(255,241,242,0.4)',
        legend_title_text="Status",
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.subheader("Regional High-Value Churn Rate")
    geo_hv = hv_df.groupby('Geography')['Exited'].agg(['mean', 'count']).reset_index()
    geo_hv['Churn Rate %'] = (geo_hv['mean'] * 100).round(1)
    
    fig2 = px.bar(
        geo_hv, x='Geography', y='Churn Rate %',
        color='Geography', text='Churn Rate %',
        color_discrete_map={'Germany': '#BE123C', 'France': '#F43F5E', 'Spain': '#FB7185'}
    )
    fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig2.update_layout(
        showlegend=False,
        paper_bgcolor='rgba(255,255,255,0)',
        plot_bgcolor='rgba(255,241,242,0.4)',
        yaxis_title="Churn Rate (%)",
        xaxis_title="",
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Table ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🚨 Priority High-Value Churned Customer Accounts")

churner_table = hv_df[hv_df['Exited'] == 1][[
    'Geography', 'Gender', 'Age', 'CreditScore',
    'Tenure', 'Balance', 'NumOfProducts',
    'IsActiveMember', 'EstimatedSalary'
]].sort_values('Balance', ascending=False).reset_index(drop=True)

churner_table['Formatted_Balance'] = churner_table['Balance'].apply(lambda x: f"€{x:,.2f}")
churner_table['Formatted_Salary'] = churner_table['EstimatedSalary'].apply(lambda x: f"€{x:,.2f}")
churner_table['Active'] = churner_table['IsActiveMember'].map({0: 'No', 1: 'Yes'})

disp = churner_table[['Geography', 'Gender', 'Age', 'CreditScore', 'Tenure', 'Formatted_Balance', 'NumOfProducts', 'Active', 'Formatted_Salary']]
st.dataframe(disp, use_container_width=True, height=300)
