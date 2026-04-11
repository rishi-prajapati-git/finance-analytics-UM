import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Churn Analytics | European Banking",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/churn_final_scored.csv")

df = load_data()

# Sidebar
st.sidebar.title("Filters")
st.sidebar.markdown("---")

selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=df['Geography'].unique().tolist(),
    default=df['Geography'].unique().tolist()
)

selected_gender = st.sidebar.multiselect(
    "Select Gender",
    options=df['Gender'].unique().tolist(),
    default=df['Gender'].unique().tolist()
)

filtered_df = df[
    (df['Geography'].isin(selected_countries)) &
    (df['Gender'].isin(selected_gender))
]

# Header
st.title("Customer Segmentation & Churn Pattern Analytics")
st.markdown("**European Banking | Unified Mentor Internship | ECB**")
st.markdown("---")

# KPI Cards
col1, col2, col3, col4 = st.columns(4)

total_customers = len(filtered_df)
overall_churn   = filtered_df['Exited'].mean() * 100
active_members  = int(filtered_df['IsActiveMember'].sum())
hv_threshold    = df['Balance'].quantile(0.75)
hv_churners     = int(filtered_df[
    (filtered_df['Balance'] >= hv_threshold) &
    (filtered_df['Exited'] == 1)
].shape[0])

col1.metric("Total Customers",    f"{total_customers:,}")
col2.metric("Overall Churn Rate", f"{overall_churn:.1f}%")
col3.metric("Active Members",     f"{active_members:,}")
col4.metric("High-Value Churners",f"{hv_churners:,}")

st.markdown("---")

# Row 1
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Overall Churn Distribution")
    churn_counts = filtered_df['Exited'].value_counts().reset_index()
    churn_counts.columns = ['Status', 'Count']
    churn_counts['Status'] = churn_counts['Status'].map({0: 'Retained', 1: 'Churned'})
    fig_pie = px.pie(
        churn_counts,
        names='Status',
        values='Count',
        color='Status',
        color_discrete_map={'Retained': '#185FA5', 'Churned': '#D85A30'},
        hole=0.4
    )
    fig_pie.update_traces(textinfo='percent+label', textfont_size=13)
    fig_pie.update_layout(showlegend=True, margin=dict(t=20, b=20))
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Churn Rate by Country")
    geo_churn = filtered_df.groupby('Geography')['Exited'].mean().reset_index()
    geo_churn.columns = ['Country', 'Churn Rate']
    geo_churn['Churn Rate %'] = (geo_churn['Churn Rate'] * 100).round(1)
    geo_churn = geo_churn.sort_values('Churn Rate %', ascending=False)
    fig_geo = px.bar(
        geo_churn,
        x='Country',
        y='Churn Rate %',
        color='Country',
        color_discrete_map={
            'Germany': '#D85A30',
            'France':  '#185FA5',
            'Spain':   '#1D9E75'
        },
        text='Churn Rate %'
    )
    fig_geo.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_geo.update_layout(showlegend=False, margin=dict(t=20, b=20),
                          yaxis_title='Churn Rate (%)')
    st.plotly_chart(fig_geo, use_container_width=True)

st.markdown("---")

# Row 2
col_l2, col_r2 = st.columns(2)

with col_l2:
    st.subheader("Churn Rate by Age Segment")
    age_churn = filtered_df.groupby('age_segment', observed=True)['Exited']\
                            .mean().reset_index()
    age_churn.columns = ['Age Segment', 'Churn Rate']
    age_churn['Churn Rate %'] = (age_churn['Churn Rate'] * 100).round(1)
    fig_age = px.bar(
        age_churn,
        x='Age Segment',
        y='Churn Rate %',
        color='Churn Rate %',
        color_continuous_scale='Blues',
        text='Churn Rate %'
    )
    fig_age.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_age.update_layout(margin=dict(t=20, b=20), coloraxis_showscale=False)
    st.plotly_chart(fig_age, use_container_width=True)

with col_r2:
    st.subheader("Active vs Inactive Member Churn")
    active_churn = filtered_df.groupby('IsActiveMember')['Exited']\
                               .mean().reset_index()
    active_churn['Member Status'] = active_churn['IsActiveMember'].map(
        {0: 'Inactive', 1: 'Active'}
    )
    active_churn['Churn Rate %'] = (active_churn['Exited'] * 100).round(1)
    fig_active = px.bar(
        active_churn,
        x='Member Status',
        y='Churn Rate %',
        color='Member Status',
        color_discrete_map={'Inactive': '#D85A30', 'Active': '#1D9E75'},
        text='Churn Rate %'
    )
    fig_active.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_active.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig_active, use_container_width=True)

st.markdown("---")
st.caption("Use the sidebar to filter by country and gender. All charts update dynamically.")