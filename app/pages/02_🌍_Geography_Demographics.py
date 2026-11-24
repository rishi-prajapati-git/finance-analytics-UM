"""
Page 02: Geography & Demographic Patterns
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

st.set_page_config(page_title="Geography & Demographics", page_icon="🌍", layout="wide")
apply_custom_styles()

render_hero_banner(
    title="🌍 Regional & Demographic Risk Patterns",
    subtitle="Examine country-level disparities, age segment vulnerability, and gender dynamics across European markets.",
    badge_text="Demographic Intelligence"
)

@st.cache_data
def get_data():
    return load_dataset()

df = get_data()

with st.sidebar:
    st.markdown("### 🎛️ Demographic Filters")
    selected_gender = st.multiselect(
        "Gender",
        options=sorted(df['Gender'].unique().tolist()),
        default=sorted(df['Gender'].unique().tolist())
    )
    selected_age = st.multiselect(
        "Age Cohorts",
        options=sorted(df['age_segment'].dropna().unique().tolist()),
        default=sorted(df['age_segment'].dropna().unique().tolist())
    )

filtered = df[
    (df['Gender'].isin(selected_gender)) &
    (df['age_segment'].isin(selected_age))
]

# ── Row 1 ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Regional Attrition Disparity")
    geo = filtered.groupby('Geography')['Exited'].agg(['mean', 'count']).reset_index()
    geo['Churn Rate %'] = (geo['mean'] * 100).round(1)
    
    fig = px.bar(
        geo, x='Geography', y='Churn Rate %',
        color='Geography', text='Churn Rate %',
        color_discrete_map={'Germany': '#BE123C', 'France': '#F43F5E', 'Spain': '#FB7185'}
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        showlegend=False,
        paper_bgcolor='rgba(255,255,255,0)',
        plot_bgcolor='rgba(255,241,242,0.4)',
        yaxis_title="Churn Rate (%)",
        xaxis_title="",
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Country vs Age Cohort Churn Rates")
    age_geo = filtered.groupby(['Geography', 'age_segment'], observed=True)['Exited']\
                      .mean().reset_index()
    age_geo['Churn Rate %'] = (age_geo['Exited'] * 100).round(1)
    
    fig2 = px.bar(
        age_geo, x='age_segment', y='Churn Rate %',
        color='Geography', barmode='group', text='Churn Rate %',
        color_discrete_map={'Germany': '#BE123C', 'France': '#F43F5E', 'Spain': '#FB7185'}
    )
    fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside', textfont_size=10)
    fig2.update_layout(
        paper_bgcolor='rgba(255,255,255,0)',
        plot_bgcolor='rgba(255,241,242,0.4)',
        yaxis_title="Churn Rate (%)",
        xaxis_title="Age Segment",
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2 ─────────────────────────────────────────────────────────────────────
st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    st.subheader("Gender Churn Divergence by Country")
    gender = filtered.groupby(['Geography', 'Gender'])['Exited'].mean().reset_index()
    gender['Churn Rate %'] = (gender['Exited'] * 100).round(1)
    
    fig3 = px.bar(
        gender, x='Geography', y='Churn Rate %',
        color='Gender', barmode='group', text='Churn Rate %',
        color_discrete_map={'Male': '#3B82F6', 'Female': '#E11D48'}
    )
    fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig3.update_layout(
        paper_bgcolor='rgba(255,255,255,0)',
        plot_bgcolor='rgba(255,241,242,0.4)',
        yaxis_title="Churn Rate (%)",
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Tenure Trajectory vs Attrition")
    tenure = filtered.groupby('tenure_group', observed=True)['Exited'].mean().reset_index()
    tenure.rename(columns={'tenure_group': 'Tenure Group'}, inplace=True)
    tenure['Churn Rate %'] = (tenure['Exited'] * 100).round(1)
    
    fig4 = px.line(
        tenure, x='Tenure Group', y='Churn Rate %',
        markers=True, text='Churn Rate %',
        color_discrete_sequence=['#E11D48']
    )
    fig4.update_traces(textposition='top center', texttemplate='%{text:.1f}%', line_width=3, marker_size=10)
    fig4.update_layout(
        paper_bgcolor='rgba(255,255,255,0)',
        plot_bgcolor='rgba(255,241,242,0.4)',
        yaxis_title="Churn Rate (%)",
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig4, use_container_width=True)
