"""
Main Executive Portal: European Banking Customer Segmentation & Churn Analytics
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_dataset
from app.ui_utils import apply_custom_styles, render_hero_banner

st.set_page_config(
    page_title="Executive Churn Analytics | European Banking",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_styles()

@st.cache_data
def get_data():
    return load_dataset("churn_final_scored.csv")

df = get_data()

# ── Header Banner ─────────────────────────────────────────────────────────────
render_hero_banner(
    title="🏦 Executive Churn & Segmentation Intelligence",
    subtitle="European Central Bank Retail Banking Analytics • 10,000 Customer Accounts",
    badge_text="Live Production Dashboard"
)

# ── Sidebar Filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Global Filters")
    st.markdown("---")

    selected_countries = st.multiselect(
        "Geography",
        options=sorted(df['Geography'].unique().tolist()),
        default=sorted(df['Geography'].unique().tolist()),
        help="Filter dashboard metrics by customer country of residence."
    )

    selected_gender = st.multiselect(
        "Gender",
        options=sorted(df['Gender'].unique().tolist()),
        default=sorted(df['Gender'].unique().tolist())
    )

    age_range = st.slider(
        "Age Range",
        min_value=int(df['Age'].min()),
        max_value=int(df['Age'].max()),
        value=(int(df['Age'].min()), int(df['Age'].max()))
    )

    st.markdown("---")
    st.info(
        "💡 **Navigation Tip**:\n"
        "Explore the left sidebar for deep-dive segment analytics, "
        "interactive **AI Churn Prediction**, and **Model Benchmarks**."
    )

# Filter dataset
filtered_df = df[
    (df['Geography'].isin(selected_countries)) &
    (df['Gender'].isin(selected_gender)) &
    (df['Age'] >= age_range[0]) &
    (df['Age'] <= age_range[1])
]

if filtered_df.empty:
    st.warning("⚠️ No customer records match the selected filter criteria.")
    st.stop()

# ── KPI Cards Row ─────────────────────────────────────────────────────────────
total_cust = len(filtered_df)
churn_rate = filtered_df['Exited'].mean() * 100
active_cust = int(filtered_df['IsActiveMember'].sum())
hv_thresh = df['Balance'].quantile(0.75)
hv_churners = int(filtered_df[(filtered_df['Balance'] >= hv_thresh) & (filtered_df['Exited'] == 1)].shape[0])
capital_at_risk = filtered_df[filtered_df['Exited'] == 1]['Balance'].sum()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    kpi1.metric(
        "Total Portfolio Customers",
        f"{total_cust:,}",
        f"{total_cust / len(df) * 100:.0f}% of total base"
    )

with kpi2:
    delta_str = f"{(churn_rate - 20.37):+.1f}% vs baseline"
    kpi2.metric(
        "Overall Churn Rate",
        f"{churn_rate:.1f}%",
        delta=delta_str,
        delta_color="inverse"
    )

with kpi3:
    kpi3.metric(
        "High-Value Churners",
        f"{hv_churners:,}",
        f"Top 25% balance tier"
    )

with kpi4:
    kpi4.metric(
        "Total Capital at Risk",
        f"€{capital_at_risk / 1e6:.2f}M",
        f"{capital_at_risk / max(1.0, filtered_df['Balance'].sum()) * 100:.1f}% of balance",
        delta_color="inverse"
    )

st.markdown("---")

# ── Visual Analytics Row 1 ─────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Customer Status Breakdown")
    status_df = filtered_df['Exited'].value_counts().reset_index()
    status_df.columns = ['Status', 'Count']
    status_df['Status'] = status_df['Status'].map({0: 'Retained', 1: 'Churned'})

    fig_pie = px.pie(
        status_df,
        names='Status',
        values='Count',
        color='Status',
        color_discrete_map={'Retained': '#3B82F6', 'Churned': '#EF4444'},
        hole=0.45
    )
    fig_pie.update_traces(textinfo='percent+label', textfont_size=13)
    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_r:
    st.subheader("Regional Churn Risk Comparison")
    geo_df = filtered_df.groupby('Geography')['Exited'].agg(['mean', 'count']).reset_index()
    geo_df['Churn Rate %'] = (geo_df['mean'] * 100).round(1)
    geo_df = geo_df.sort_values('Churn Rate %', ascending=False)

    fig_geo = px.bar(
        geo_df,
        x='Geography',
        y='Churn Rate %',
        color='Geography',
        text='Churn Rate %',
        color_discrete_map={'Germany': '#EF4444', 'France': '#3B82F6', 'Spain': '#10B981'}
    )
    fig_geo.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_geo.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.4)',
        showlegend=False,
        yaxis_title="Churn Rate (%)",
        xaxis_title="",
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_geo, use_container_width=True)

st.markdown("---")

# ── Visual Analytics Row 2 ─────────────────────────────────────────────────────
col_l2, col_r2 = st.columns(2)

with col_l2:
    st.subheader("Churn Rate by Age Cohort")
    age_df = filtered_df.groupby('age_segment', observed=True)['Exited'].agg(['mean', 'count']).reset_index()
    age_df['Churn Rate %'] = (age_df['mean'] * 100).round(1)

    fig_age = px.bar(
        age_df,
        x='age_segment',
        y='Churn Rate %',
        color='Churn Rate %',
        color_continuous_scale='Reds',
        text='Churn Rate %'
    )
    fig_age.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_age.update_layout(
        coloraxis_showscale=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.4)',
        yaxis_title="Churn Rate (%)",
        xaxis_title="",
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_age, use_container_width=True)

with col_r2:
    st.subheader("Activity Status Impact (Active vs Inactive)")
    act_df = filtered_df.groupby('IsActiveMember')['Exited'].agg(['mean', 'count']).reset_index()
    act_df['Status'] = act_df['IsActiveMember'].map({0: 'Inactive Member', 1: 'Active Member'})
    act_df['Churn Rate %'] = (act_df['mean'] * 100).round(1)

    fig_act = px.bar(
        act_df,
        x='Status',
        y='Churn Rate %',
        color='Status',
        color_discrete_map={'Inactive Member': '#EF4444', 'Active Member': '#10B981'},
        text='Churn Rate %'
    )
    fig_act.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_act.update_layout(
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.4)',
        yaxis_title="Churn Rate (%)",
        xaxis_title="",
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_act, use_container_width=True)

# ── Quick Export & Drilldown ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("📥 Export Filtered Analytical Records")
csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Filtered Data (CSV)",
    data=csv_bytes,
    file_name="filtered_banking_churn_records.csv",
    mime="text/csv"
)
