import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Churn Summary", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/churn_final_scored.csv")

df = load_data()

st.title("Segment-wise Churn Summary")
st.markdown("**Churn rates across all customer segments**")
st.markdown("---")

st.sidebar.title("Filters")
selected_segment = st.sidebar.selectbox(
    "View Segment",
    ["Age Segment", "Credit Band", "Tenure Group",
     "Balance Segment", "Number of Products"]
)
selected_countries = st.sidebar.multiselect(
    "Countries",
    options = df['Geography'].unique().tolist(),
    default = df['Geography'].unique().tolist()
)

filtered = df[df['Geography'].isin(selected_countries)]

segment_map = {
    "Age Segment"        : "age_segment",
    "Credit Band"        : "credit_band",
    "Tenure Group"       : "tenure_group",
    "Balance Segment"    : "balance_segment",
    "Number of Products" : "NumOfProducts"
}

seg_col = segment_map[selected_segment]

seg_data = filtered.groupby(seg_col, observed=True)['Exited']\
                   .agg(['sum','count','mean']).reset_index()
seg_data.columns = ['Segment', 'Churned', 'Total', 'Churn Rate']
seg_data['Churn Rate %']    = (seg_data['Churn Rate'] * 100).round(1)
seg_data['Retained']        = seg_data['Total'] - seg_data['Churned']
seg_data['% of All Churned'] = (seg_data['Churned'] / seg_data['Churned'].sum() * 100).round(1)

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Churn Rate by {selected_segment}")
    fig = px.bar(seg_data.sort_values('Churn Rate %', ascending=False),
                 x='Segment', y='Churn Rate %',
                 color='Churn Rate %', color_continuous_scale='RdYlGn_r',
                 text='Churn Rate %')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader(f"Churn Contribution by {selected_segment}")
    fig2 = px.pie(seg_data, names='Segment', values='Churned',
                  hole=0.35, color_discrete_sequence=px.colors.qualitative.Set2)
    fig2.update_traces(textinfo='percent+label')
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.subheader("Detailed Segment Table")
display_df = seg_data[['Segment','Total','Churned','Retained',
                        'Churn Rate %','% of All Churned']].copy()
st.dataframe(display_df, use_container_width=True)