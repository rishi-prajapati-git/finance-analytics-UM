"""
UI Components, Custom Styling, and Plotting Helpers for the Streamlit FinTech Dashboard.
Clean White & Rose-Pink Modern Aesthetic with Brightly Visible Emojis & Logos.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import re


def apply_custom_styles():
    """Inject modern, crisp White & Rose-Pink styling into Streamlit."""
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 95%;
    }
    
    /* Hero Header Card - White & Soft Rose-Pink */
    .hero-card {
        background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 45%, #FCE7F3 100%);
        border: 1px solid #FECDD3;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px -4px rgba(225, 29, 72, 0.08), 0 4px 8px -4px rgba(225, 29, 72, 0.04);
    }
    
    .hero-title-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
    }
    
    .hero-icon {
        font-size: 2.2rem;
        display: inline-block;
        -webkit-text-fill-color: initial !important;
        color: initial !important;
    }
    
    .hero-title-text {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #BE123C 0%, #E11D48 60%, #FB7185 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        color: #475569;
        font-size: 1.05rem;
        font-weight: 500;
        margin-bottom: 0px;
    }
    
    /* Metric Cards - Crisp White with Rose-Pink left accent */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #FFE4E6;
        border-left: 5px solid #E11D48;
        border-radius: 14px;
        padding: 18px 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 16px rgba(225, 29, 72, 0.06);
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px -4px rgba(225, 29, 72, 0.15);
        border-color: #FDA4AF;
    }
    
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #9F1239;
        margin-bottom: 6px;
    }
    
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #0F172A;
    }
    
    .kpi-subtext {
        font-size: 0.82rem;
        font-weight: 500;
        color: #64748B;
        margin-top: 4px;
    }
    
    /* Modern Badges */
    .badge-critical {
        background-color: #FFE4E6;
        color: #BE123C;
        border: 1px solid #FDA4AF;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .badge-warning {
        background-color: #FEF3C7;
        color: #B45309;
        border: 1px solid #FCD34D;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .badge-success {
        background-color: #D1FAE5;
        color: #065F46;
        border: 1px solid #6EE7B7;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .badge-pink {
        background-color: #FCE7F3;
        color: #9D174D;
        border: 1px solid #FBCFE8;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Sidebar Custom Styling - Soft Pinkish White */
    [data-testid="stSidebar"] {
        background-color: #FFF1F2;
        border-right: 1px solid #FFE4E6;
    }
    
    /* Sidebar navigation links hover */
    [data-testid="stSidebarNav"] a:hover {
        background-color: #FCE7F3;
        color: #BE123C;
    }
    
    /* Plotly chart container styling */
    .stPlotlyChart {
        background: #FFFFFF;
        border: 1px solid #FFE4E6;
        border-radius: 14px;
        padding: 12px;
        box-shadow: 0 4px 16px rgba(225, 29, 72, 0.04);
    }
    
    /* Buttons in Pink Accent */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #E11D48 0%, #BE123C 100%);
        color: #FFFFFF !important;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1.2rem;
        transition: all 0.2s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(225, 29, 72, 0.35);
        color: #FFFFFF !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_hero_banner(title: str, subtitle: str, badge_text: str = "ECB Enterprise Analytics"):
    """
    Render hero header banner with separate icon container so that emojis/logos
    are ALWAYS 100% brightly visible and NEVER affected by text gradient transparency.
    """
    # Separate leading emoji if present
    match = re.match(r"^([\U00010000-\U0010ffff\u2600-\u27bf\u2300-\u23ff\u2b50\u2b55\u200d]+)\s*(.*)", title)
    if match:
        icon = match.group(1)
        text_title = match.group(2)
    else:
        # Check first token
        tokens = title.strip().split(' ', 1)
        if len(tokens) == 2 and any(ord(c) > 127 for c in tokens[0]):
            icon = tokens[0]
            text_title = tokens[1]
        else:
            icon = ""
            text_title = title

    html = f"""
    <div class="hero-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
            <div>
                <div class="hero-title-container">
                    {f'<span class="hero-icon">{icon}</span>' if icon else ''}
                    <span class="hero-title-text">{text_title}</span>
                </div>
                <div class="hero-subtitle">{subtitle}</div>
            </div>
            <span class="badge-pink">{badge_text}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, subtext: str = "", border_color: str = "#E11D48"):
    """Render a single styled White & Pink KPI metric card."""
    html = f"""
    <div class="kpi-card" style="border-left: 5px solid {border_color};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {f'<div class="kpi-subtext">{subtext}</div>' if subtext else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_gauge_chart(probability: float) -> go.Figure:
    """Create a semi-circular gauge chart indicating churn probability on white/pink theme."""
    prob_pct = round(probability * 100, 1)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        number={'suffix': '%', 'font': {'size': 44, 'color': '#0F172A', 'family': 'Inter'}},
        title={'text': "Predicted Churn Probability", 'font': {'size': 18, 'color': '#9F1239', 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': "#E11D48", 'thickness': 0.28},
            'bgcolor': "#FFFFFF",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.35)"},
                {'range': [30, 60], 'color': "rgba(245, 158, 11, 0.35)"},
                {'range': [60, 100], 'color': "rgba(225, 29, 72, 0.35)"}
            ],
            'threshold': {
                'line': {'color': "#BE123C", 'width': 3},
                'thickness': 0.75,
                'value': 45.0
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(255,255,255,0)',
        plot_bgcolor='rgba(255,255,255,0)',
        height=260,
        margin=dict(l=20, r=20, t=40, b=10)
    )
    return fig
