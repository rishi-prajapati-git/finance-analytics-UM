import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="High-Value Customer Explorer", layout="wide")

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "churn_final_scored.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.title("High-Value Customer Explorer")
st.markdown("**Identify and analyze customers with the highest account balances**")
st.markdown("---")

# Sidebar
st.sidebar.title("Filters")
balance_pct = st.sidebar.slider(
    "High-Value Threshold (Balance Percentile)",
    min_value = 50,
    max_value = 95,
    value     = 75,
    step      = 5,
    help      = "Customers above this balance percentile are considered high-value"
)
selected_countries = st.sidebar.multiselect(
    "Countries",
    options = df['Geography'].unique().tolist(),
    default = df['Geography'].unique().tolist()
)

threshold = df['Balance'].quantile(balance_pct / 100)
hv_df     = df[
    (df['Balance'] >= threshold) &
    (df['Geography'].isin(selected_countries))
]

# ── Metric Cards ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("High-Value Customers",  f"{len(hv_df):,}")
col2.metric("Churned (High-Value)",  f"{hv_df['Exited'].sum():,}")
col3.metric("Churn Rate",            f"{hv_df['Exited'].mean() * 100:.1f}%")
col4.metric("Balance at Risk",
            f"€{hv_df[hv_df['Exited']==1]['Balance'].sum():,.0f}")

st.markdown(f"*Showing customers with balance ≥ €{threshold:,.0f} "
            f"(top {100 - balance_pct}% by balance)*")
st.markdown("---")

# ── Charts ────────────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Balance vs Salary — High-Value Customers")
    fig = px.scatter(
        hv_df,
        x     = 'Balance',
        y     = 'EstimatedSalary',
        color = hv_df['Exited'].map({0: 'Retained', 1: 'Churned'}),
        color_discrete_map = {'Retained': '#185FA5', 'Churned': '#D85A30'},
        opacity = 0.6,
        hover_data = ['Geography', 'Age', 'NumOfProducts']
    )
    fig.update_layout(legend_title_text='Status')
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.subheader("High-Value Churn Rate by Country")
    geo_hv = hv_df.groupby('Geography')['Exited'].agg(['sum','count','mean'])\
                  .reset_index()
    geo_hv.columns = ['Country', 'Churned', 'Total', 'Churn Rate']
    geo_hv['Churn Rate %'] = (geo_hv['Churn Rate'] * 100).round(1)
    fig2 = px.bar(geo_hv, x='Country', y='Churn Rate %',
                  color='Country', text='Churn Rate %',
                  color_discrete_map={
                      'Germany': '#D85A30', 'France': '#185FA5', 'Spain': '#1D9E75'
                  })
    fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# ── High-Value Churner Table ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("High-Value Churned Customer Records")

churner_table = hv_df[hv_df['Exited'] == 1][[
    'Geography', 'Gender', 'Age', 'CreditScore',
    'Tenure', 'Balance', 'NumOfProducts',
    'IsActiveMember', 'EstimatedSalary'
]].sort_values('Balance', ascending=False).reset_index(drop=True)

churner_table['Balance']         = churner_table['Balance'].apply(lambda x: f"€{x:,.0f}")
churner_table['EstimatedSalary'] = churner_table['EstimatedSalary'].apply(lambda x: f"€{x:,.0f}")
churner_table['IsActiveMember']  = churner_table['IsActiveMember'].map({0: 'No', 1: 'Yes'})

st.dataframe(churner_table, use_container_width=True, height=350)
st.caption(f"Showing {len(churner_table)} high-value customers who churned, "
           f"sorted by balance (highest first)")