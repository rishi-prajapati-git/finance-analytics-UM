"""
Page 01: Segment-wise Churn Analysis
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

st.set_page_config(page_title="Segment Churn Summary", page_icon="📊", layout="wide")
apply_custom_styles()

render_hero_banner(
    title="📊 Multi-Dimensional Segment Churn Summary",
    subtitle="Evaluate customer attrition rates and cohort volumes across demographics, balance, and tenure brackets.",
    badge_text="Cohort Analytics"
)

@st.cache_data
def get_data():
    return load_dataset()

df = get_data()

# ── Sidebar Filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Segmentation Controls")
    selected_segment = st.selectbox(
        "Segment Dimension",
        ["Age Segment", "Credit Band", "Tenure Group", "Balance Segment", "Number of Products"]
    )
    selected_countries = st.multiselect(
        "Geography Filter",
        options=sorted(df['Geography'].unique().tolist()),
        default=sorted(df['Geography'].unique().tolist())
    )

filtered = df[df['Geography'].isin(selected_countries)]

segment_map = {
    "Age Segment": "age_segment",
    "Credit Band": "credit_band",
    "Tenure Group": "tenure_group",
    "Balance Segment": "balance_segment",
    "Number of Products": "products_segment"
}

seg_col = segment_map[selected_segment]

seg_data = filtered.groupby(seg_col, observed=True)['Exited']\
                   .agg(['sum', 'count', 'mean']).reset_index()
seg_data.columns = ['Segment', 'Churned', 'Total', 'Churn Rate']
seg_data['Churn Rate %'] = (seg_data['Churn Rate'] * 100).round(1)
seg_data['Retained'] = seg_data['Total'] - seg_data['Churned']
seg_data['% of Total Churn'] = (seg_data['Churned'] / seg_data['Churned'].sum() * 100).round(1)

# ── Visual Analytics ───────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Attrition Rate by {selected_segment}")
    fig = px.bar(
        seg_data.sort_values('Churn Rate %', ascending=False),
        x='Segment',
        y='Churn Rate %',
        color='Churn Rate %',
        color_continuous_scale='Reds',
        text='Churn Rate %'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        coloraxis_showscale=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.4)',
        yaxis_title="Churn Rate (%)",
        xaxis_title="",
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader(f"Share of Total Churned Population ({selected_segment})")
    fig2 = px.pie(
        seg_data,
        names='Segment',
        values='Churned',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig2.update_traces(textinfo='percent+label', textfont_size=12)
    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.subheader(f"📋 Detailed Breakdown: {selected_segment}")
display_table = seg_data[['Segment', 'Total', 'Churned', 'Retained', 'Churn Rate %', '% of Total Churn']].copy()
st.dataframe(
    display_table.style.highlight_max(subset=['Churn Rate %'], color='#7F1D1D'),
    use_container_width=True
)
