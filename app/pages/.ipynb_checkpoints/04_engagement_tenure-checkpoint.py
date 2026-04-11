import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Engagement & Tenure", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/churn_final_scored.csv")

df = load_data()

st.title("Engagement & Tenure Panel")
st.markdown("**How activity levels and customer tenure affect churn behavior**")
st.markdown("---")

# Sidebar
st.sidebar.title("Filters")
selected_countries = st.sidebar.multiselect(
    "Countries",
    options = df['Geography'].unique().tolist(),
    default = df['Geography'].unique().tolist()
)
filtered = df[df['Geography'].isin(selected_countries)]

# ── KPI Row ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
inactive_churn = filtered[filtered['IsActiveMember'] == 0]['Exited'].mean() * 100
active_churn   = filtered[filtered['IsActiveMember'] == 1]['Exited'].mean() * 100
engagement_gap = inactive_churn - active_churn
retention_opps = filtered['retention_opportunity'].sum()

col1.metric("Inactive Member Churn", f"{inactive_churn:.1f}%")
col2.metric("Active Member Churn",   f"{active_churn:.1f}%")
col3.metric("Engagement Gap",        f"{engagement_gap:.1f}%",
            delta=f"{engagement_gap:.1f}% higher risk if inactive",
            delta_color="inverse")
col4.metric("Retention Opportunities", f"{retention_opps:,}")

st.markdown("---")

# ── Row 1 ─────────────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Active vs Inactive Churn Rate")
    act_data = filtered.groupby('IsActiveMember')['Exited'].mean().reset_index()
    act_data['Status']     = act_data['IsActiveMember'].map({0:'Inactive', 1:'Active'})
    act_data['Churn Rate %'] = (act_data['Exited'] * 100).round(1)
    fig = px.bar(act_data, x='Status', y='Churn Rate %',
                 color='Status', text='Churn Rate %',
                 color_discrete_map={'Inactive':'#D85A30', 'Active':'#1D9E75'})
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.subheader("Tenure Group Churn Rate")
    tenure_data = filtered.groupby('tenure_group', observed=True)['Exited']\
                          .mean().reset_index()
    tenure_data.columns = ['Tenure Group', 'Churn Rate']
    tenure_data['Churn Rate %'] = (tenure_data['Churn Rate'] * 100).round(1)
    fig2 = px.bar(tenure_data, x='Tenure Group', y='Churn Rate %',
                  color='Churn Rate %', color_continuous_scale='Oranges',
                  text='Churn Rate %')
    fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig2.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2 ─────────────────────────────────────────────────────────────────────
st.markdown("---")
col_l2, col_r2 = st.columns(2)

with col_l2:
    st.subheader("Number of Products vs Churn Rate")
    prod_data = filtered.groupby('NumOfProducts')['Exited'].agg(['mean','count'])\
                        .reset_index()
    prod_data.columns = ['Num Products', 'Churn Rate', 'Total Customers']
    prod_data['Churn Rate %'] = (prod_data['Churn Rate'] * 100).round(1)
    fig3 = px.bar(prod_data, x='Num Products', y='Churn Rate %',
                  color='Churn Rate %', color_continuous_scale='Reds',
                  text='Churn Rate %')
    fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig3.update_layout(coloraxis_showscale=False,
                       xaxis=dict(tickmode='linear', dtick=1))
    st.plotly_chart(fig3, use_container_width=True)

with col_r2:
    st.subheader("Retention Opportunities by Country")
    ret_data = filtered[filtered['retention_opportunity'] == 1]\
                       .groupby('Geography').size().reset_index()
    ret_data.columns = ['Country', 'Opportunities']
    fig4 = px.bar(ret_data, x='Country', y='Opportunities',
                  color='Country', text='Opportunities',
                  color_discrete_map={
                      'Germany': '#D85A30', 'France': '#185FA5', 'Spain': '#1D9E75'
                  })
    fig4.update_traces(textposition='outside')
    fig4.update_layout(showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")
st.subheader("Engagement vs Churn — Detailed Breakdown")
breakdown = filtered.groupby(['Geography', 'IsActiveMember'])['Exited']\
                    .mean().reset_index()
breakdown['Member Status'] = breakdown['IsActiveMember'].map({0:'Inactive', 1:'Active'})
breakdown['Churn Rate %']  = (breakdown['Exited'] * 100).round(1)

fig5 = px.bar(breakdown, x='Geography', y='Churn Rate %',
              color='Member Status', barmode='group',
              text='Churn Rate %',
              color_discrete_map={'Inactive':'#D85A30', 'Active':'#1D9E75'})
fig5.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig5.update_layout(legend_title_text='Member Status')
st.plotly_chart(fig5, use_container_width=True)