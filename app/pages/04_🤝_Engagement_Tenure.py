"""
Page 04: Engagement, Activity Levels & Loyalty Dynamics
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

st.set_page_config(page_title="Engagement & Tenure", page_icon="🤝", layout="wide")
apply_custom_styles()

render_hero_banner(
    title="🤝 Engagement & Tenure Dynamics",
    subtitle="Assess how customer interaction frequency, account activity, and multi-product relationships drive retention.",
    badge_text="Engagement Analytics"
)

@st.cache_data
def get_data():
    return load_dataset()

df = get_data()

with st.sidebar:
    st.markdown("### 🎛️ Regional Scope")
    selected_countries = st.multiselect(
        "Geography",
        options=sorted(df['Geography'].unique().tolist()),
        default=sorted(df['Geography'].unique().tolist())
    )

filtered = df[df['Geography'].isin(selected_countries)]

# ── KPI Row ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
inactive_churn = filtered[filtered['IsActiveMember'] == 0]['Exited'].mean() * 100
active_churn = filtered[filtered['IsActiveMember'] == 1]['Exited'].mean() * 100
engagement_gap = inactive_churn - active_churn
retention_opps = int(filtered['retention_opportunity'].sum())

col1.metric("Inactive Member Churn", f"{inactive_churn:.1f}%")
col2.metric("Active Member Churn", f"{active_churn:.1f}%")
col3.metric("Engagement Delta Gap", f"{engagement_gap:.1f}%", delta=f"{engagement_gap:+.1f}% higher risk", delta_color="inverse")
col4.metric("Retention Opportunities", f"{retention_opps:,}", "High-balance inactive accounts")

st.markdown("---")

# ── Row 1 ─────────────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Active vs Inactive Attrition Rate")
    act_data = filtered.groupby('IsActiveMember')['Exited'].agg(['mean', 'count']).reset_index()
    act_data['Status'] = act_data['IsActiveMember'].map({0: 'Inactive Member', 1: 'Active Member'})
    act_data['Churn Rate %'] = (act_data['mean'] * 100).round(1)
    
    fig = px.bar(
        act_data, x='Status', y='Churn Rate %',
        color='Status', text='Churn Rate %',
        color_discrete_map={'Inactive Member': '#EF4444', 'Active Member': '#10B981'}
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.4)',
        yaxis_title="Churn Rate (%)",
        xaxis_title="",
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.subheader("Tenure Group Attrition Curve")
    tenure_data = filtered.groupby('tenure_group', observed=True)['Exited'].agg(['mean', 'count']).reset_index()
    tenure_data['Churn Rate %'] = (tenure_data['mean'] * 100).round(1)
    
    fig2 = px.bar(
        tenure_data, x='Tenure Group', y='Churn Rate %',
        color='Churn Rate %', color_continuous_scale='Blues',
        text='Churn Rate %'
    )
    fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig2.update_layout(
        coloraxis_showscale=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.4)',
        yaxis_title="Churn Rate (%)",
        xaxis_title="",
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2 ─────────────────────────────────────────────────────────────────────
st.markdown("---")
col_l2, col_r2 = st.columns(2)

with col_l2:
    st.subheader("Product Holdings vs Churn Rate")
    prod_data = filtered.groupby('NumOfProducts')['Exited'].agg(['mean', 'count']).reset_index()
    prod_data['Churn Rate %'] = (prod_data['mean'] * 100).round(1)
    
    fig3 = px.bar(
        prod_data, x='NumOfProducts', y='Churn Rate %',
        color='Churn Rate %', color_continuous_scale='Reds',
        text='Churn Rate %'
    )
    fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig3.update_layout(
        coloraxis_showscale=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.4)',
        xaxis=dict(tickmode='linear', dtick=1),
        yaxis_title="Churn Rate (%)",
        xaxis_title="Number of Products Held",
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_r2:
    st.subheader("High-Balance Retention Targets by Region")
    ret_data = filtered[filtered['retention_opportunity'] == 1]\
                       .groupby('Geography').size().reset_index()
    ret_data.columns = ['Geography', 'Opportunities']
    
    fig4 = px.bar(
        ret_data, x='Geography', y='Opportunities',
        color='Geography', text='Opportunities',
        color_discrete_map={'Germany': '#EF4444', 'France': '#3B82F6', 'Spain': '#10B981'}
    )
    fig4.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig4.update_layout(
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.4)',
        yaxis_title="Target Customer Count",
        xaxis_title="",
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig4, use_container_width=True)
