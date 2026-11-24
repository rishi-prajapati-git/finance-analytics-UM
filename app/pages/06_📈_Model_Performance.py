"""
Page 06: Model Benchmarks, ROC Curves, Explainability & Cost ROI Analysis
"""

import sys
import json
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.ui_utils import apply_custom_styles, render_hero_banner

st.set_page_config(
    page_title="Model Benchmarks & Explainability",
    page_icon="📈",
    layout="wide"
)

apply_custom_styles()

render_hero_banner(
    title="📈 Machine Learning Benchmarks & Explainability",
    subtitle="Comparative evaluation of classification models, ROC curves, feature importance, and financial ROI formulation.",
    badge_text="Audited ML Architecture"
)

MODELS_DIR = ROOT_DIR / "models"

@st.cache_data
def load_model_artifacts():
    with open(MODELS_DIR / "model_comparison.json", "r", encoding="utf-8") as f:
        comparison = json.load(f)
    with open(MODELS_DIR / "model_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(MODELS_DIR / "feature_importance.json", "r", encoding="utf-8") as f:
        feat_imp = json.load(f)
    with open(MODELS_DIR / "roc_pr_curves.json", "r", encoding="utf-8") as f:
        curves = json.load(f)
    return comparison, metrics, feat_imp, curves

try:
    comparison, metrics, feat_imp, curves = load_model_artifacts()
except Exception as e:
    st.error(f"Could not load ML artifacts: {e}. Please run training pipeline first.")
    st.stop()

# ── Row 1: Model Scoreboard ───────────────────────────────────────────────────
st.subheader("🏆 Model Benchmark Scoreboard (5-Fold Stratified CV)")

records = []
for model_name, stats in comparison.items():
    records.append({
        "Model": model_name,
        "ROC-AUC": stats["ROC_AUC"],
        "CV ROC-AUC (Mean)": stats["CV_ROC_AUC_Mean"],
        "F1-Score": stats["F1_Score"],
        "Recall": stats["Recall"],
        "Precision": stats["Precision"],
        "Accuracy": stats["Accuracy"]
    })

score_df = pd.DataFrame(records).sort_values("ROC-AUC", ascending=False)
st.dataframe(
    score_df.style.highlight_max(subset=["ROC-AUC", "CV ROC-AUC (Mean)", "F1-Score", "Recall"], color="#1E3A8A"),
    use_container_width=True
)

st.markdown("---")

# ── Row 2: ROC and Precision-Recall Curves ──────────────────────────────────────
col_c1, col_c2 = st.columns(2)

with col_c1:
    st.subheader("ROC Curves Comparison")
    fig_roc = go.Figure()
    palette = {
        "Logistic Regression": "#94A3B8",
        "Random Forest": "#3B82F6",
        "Gradient Boosting": "#10B981",
        "XGBoost Classifier": "#EF4444"
    }

    for name, c_data in curves.items():
        auc_score = comparison.get(name, {}).get("ROC_AUC", 0.0)
        fig_roc.add_trace(go.Scatter(
            x=c_data["fpr"],
            y=c_data["tpr"],
            mode="lines",
            name=f"{name} (AUC: {auc_score:.3f})",
            line=dict(color=palette.get(name, "#38BDF8"), width=2.5)
        ))

    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random Classifier",
        line=dict(color="#64748B", dash="dash")
    ))

    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate (Recall)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.4)',
        legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.98),
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_roc, use_container_width=True)

with col_c2:
    st.subheader("Precision-Recall Curves")
    fig_pr = go.Figure()

    for name, c_data in curves.items():
        fig_pr.add_trace(go.Scatter(
            x=c_data["recall_curve"],
            y=c_data["precision_curve"],
            mode="lines",
            name=name,
            line=dict(color=palette.get(name, "#38BDF8"), width=2.5)
        ))

    fig_pr.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.4)',
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_pr, use_container_width=True)

st.markdown("---")

# ── Row 3: Feature Importance & Financial Cost Simulation ──────────────────────
col_f1, col_f2 = st.columns(2)

with col_f1:
    st.subheader("Global Feature Importance (XGBoost)")
    df_feat = pd.DataFrame(feat_imp).head(10).sort_values("importance", ascending=True)
    fig_feat = px.bar(
        df_feat,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Viridis",
        text="importance"
    )
    fig_feat.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_feat.update_layout(
        coloraxis_showscale=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.4)',
        yaxis_title="",
        xaxis_title="Normalized Relative Importance",
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_feat, use_container_width=True)

with col_f2:
    st.subheader("💰 Banking Cost-Benefit ROI Model")
    st.markdown(
        "Evaluate total savings achieved by deploying machine learning over an unguided retention program:"
    )

    avg_ltv = st.number_input("Average Customer Lifetime Value (€ LTV)", value=2500, step=250)
    campaign_cost = st.number_input("Retention Offer / Contact Cost per Customer (€)", value=150, step=25)
    success_rate = st.slider("Retention Campaign Win Rate (%)", min_value=15, max_value=60, value=35) / 100.0

    # Test set statistics (2,000 customers)
    total_test = 2000
    actual_churners = int(total_test * metrics.get("churn_rate_test", 0.2037))
    xg_recall = comparison["XGBoost Classifier"]["Recall"]
    detected_churners = int(actual_churners * xg_recall)
    retained_customers = int(detected_churners * success_rate)

    gross_saved = retained_customers * avg_ltv
    total_campaign_outlay = detected_churners * campaign_cost
    net_profit = gross_saved - total_campaign_outlay

    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Retained Customers (Saved)", f"{retained_customers:,} accounts")
    col_m2.metric("Gross LTV Protected", f"€{gross_saved:,.0f}")

    col_m3, col_m4 = st.columns(2)
    col_m3.metric("Retention Budget", f"€{total_campaign_outlay:,.0f}")
    col_m4.metric("Net Financial Benefit", f"€{net_profit:,.0f}", delta="Net Profit from ML")
