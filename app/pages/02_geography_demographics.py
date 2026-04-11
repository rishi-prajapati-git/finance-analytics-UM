import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Geography & Demographics", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/churn_final_scored.csv")

df = load_data()

st.title("Geography & Demographics Analysis")
st.markdown("**Churn patterns across countries, age groups, and gender**")
st.markdown("---")

# Sidebar filters
st.sidebar.title("Filters")
selected_gender = st.sidebar.multiselect(
    "Gender",
    options = df['Gender'].unique().tolist(),
    default = df['Gender'].unique().tolist()
)
selected_age = st.sidebar.multiselect(
    "Age Segment",
    options = df['age_segment'].dropna().unique().tolist(),
    default = df['age_segment'].dropna().unique().tolist()
)

filtered = df[
    (df['Gender'].isin(selected_gender)) &
    (df['age_segment'].isin(selected_age))
]

# ── Row 1 ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Rate by Country")
    geo = filtered.groupby('Geography')['Exited'].mean().reset_index()
    geo.columns = ['Country', 'Churn Rate']
    geo['Churn Rate %'] = (geo['Churn Rate'] * 100).round(1)
    fig = px.bar(geo, x='Country', y='Churn Rate %',
                 color='Country', text='Churn Rate %',
                 color_discrete_map={
                     'Germany': '#D85A30', 'France': '#185FA5', 'Spain': '#1D9E75'
                 })
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Age Segment Churn Comparison")
    age_geo = filtered.groupby(['Geography', 'age_segment'], observed=True)['Exited']\
                      .mean().reset_index()
    age_geo.columns = ['Country', 'Age Segment', 'Churn Rate']
    age_geo['Churn Rate %'] = (age_geo['Churn Rate'] * 100).round(1)
    fig2 = px.bar(age_geo, x='Age Segment', y='Churn Rate %',
                  color='Country', barmode='group', text='Churn Rate %',
                  color_discrete_map={
                      'Germany': '#D85A30', 'France': '#185FA5', 'Spain': '#1D9E75'
                  })
    fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside',
                       textfont_size=9)
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2 ─────────────────────────────────────────────────────────────────────
st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    st.subheader("Gender-wise Churn Rate")
    gender = filtered.groupby(['Geography', 'Gender'])['Exited'].mean().reset_index()
    gender.columns = ['Country', 'Gender', 'Churn Rate']
    gender['Churn Rate %'] = (gender['Churn Rate'] * 100).round(1)
    fig3 = px.bar(gender, x='Country', y='Churn Rate %',
                  color='Gender', barmode='group', text='Churn Rate %',
                  color_discrete_map={'Male': '#185FA5', 'Female': '#D4537E'})
    fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside',
                       textfont_size=10)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Tenure Group vs Churn Rate")
    tenure = filtered.groupby('tenure_group', observed=True)['Exited']\
                     .mean().reset_index()
    tenure.columns = ['Tenure Group', 'Churn Rate']
    tenure['Churn Rate %'] = (tenure['Churn Rate'] * 100).round(1)
    fig4 = px.line(tenure, x='Tenure Group', y='Churn Rate %',
                   markers=True, text='Churn Rate %',
                   color_discrete_sequence=['#185FA5'])
    fig4.update_traces(textposition='top center', texttemplate='%{text:.1f}%',
                       line_width=2.5, marker_size=10)
    fig4.update_layout(yaxis_title='Churn Rate (%)')
    st.plotly_chart(fig4, use_container_width=True)