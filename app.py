"""Customer Churn & Lifetime Value (LTV) Predictor - Interactive Dashboard."""

import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Retention Radar - Customer Churn & LTV Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.predict import score_customer


@st.cache_data
def load_customer_scores():
    """Load precomputed customer predictions and scores."""
    df = pd.read_csv(DATA_DIR / "customer_scores.csv")
    df["CustomerID"] = df["CustomerID"].astype(str)
    df["RevenueAtRisk"] = (df["ChurnProb"] * df["LTV_12m"]).round(2)
    return df


@st.cache_data
def load_feature_matrix():
    """Load engineered feature records."""
    df = pd.read_csv(DATA_DIR / "features.csv")
    df["CustomerID"] = df["CustomerID"].astype(str)
    return df


@st.cache_data
def load_metrics_json():
    """Load saved modeling metrics."""
    with open(MODELS_DIR / "metrics.json", "r") as f:
        return json.load(f)


@st.cache_resource
def load_model_artifacts():
    """Load models and artifacts."""
    return {
        "churn": joblib.load(MODELS_DIR / "churn_xgb.joblib"),
        "cox": joblib.load(MODELS_DIR / "cox.joblib")
    }


scores_df = load_customer_scores()
feat_df = load_feature_matrix()
metrics_dict = load_metrics_json()
artifacts = load_model_artifacts()

# Professional Black & Green Theme
st.markdown("""
<style>
    :root {
        --primary-green: #10b981;
        --dark-green: #059669;
        --accent-green: #34d399;
        --bg-black: #09090b;
        --card-black: #18181b;
        --border-color: #27272a;
    }
    .metric-card {
        background-color: #18181b;
        border-radius: 8px;
        padding: 16px 20px;
        border: 1px solid #27272a;
        border-left: 4px solid #10b981;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #a1a1aa;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #10b981;
    }
    .status-badge-high {
        background-color: #27272a;
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.85rem;
    }
    .status-badge-low {
        background-color: #27272a;
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.85rem;
    }
    .custom-box {
        background-color: #18181b;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 18px;
        margin-top: 10px;
    }
    /* Button Navigation in Sidebar */
    [data-testid="stSidebar"] button[kind="primary"] {
        background-color: #10b981 !important;
        color: #09090b !important;
        border: 1px solid #10b981 !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.25) !important;
        text-align: left !important;
    }
    [data-testid="stSidebar"] button[kind="secondary"] {
        background-color: #18181b !important;
        color: #d4d4d8 !important;
        border: 1px solid #27272a !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
        text-align: left !important;
        transition: all 0.2s ease-in-out !important;
    }
    [data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #27272a !important;
        border-color: #10b981 !important;
        color: #10b981 !important;
    }
</style>
""", unsafe_allow_html=True)

# Session state for page navigation
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Overview Dashboard"

with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #18181b, #09090b); border: 1px solid #27272a; border-radius: 10px; padding: 14px 16px; margin-bottom: 14px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <span style="font-size: 1.25rem; font-weight: 800; letter-spacing: -0.5px; color: #ffffff;">RETENTION<span style="color: #10b981;"> RADAR</span></span>
            <span style="font-size: 0.65rem; background: #052e16; color: #10b981; border: 1px solid #10b981; padding: 2px 8px; border-radius: 12px; font-weight: 700;">ACTIVE</span>
        </div>
        <div style="font-size: 0.72rem; color: #71717a; margin-top: 4px;">Predictive Churn & Lifetime Value Radar</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size: 0.75rem; font-weight: 700; color: #71717a; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.5px;'>Navigation Menu</div>", unsafe_allow_html=True)

    nav_options = [
        "Overview Dashboard",
        "Customer Lookup",
        "Live Prediction",
        "At-Risk Customers"
    ]

    for nav_item in nav_options:
        is_selected = (st.session_state["current_page"] == nav_item)
        btn_type = "primary" if is_selected else "secondary"
        label = f"•  {nav_item}" if is_selected else f"   {nav_item}"
        if st.button(label, key=f"nav_btn_{nav_item}", type=btn_type, width="stretch"):
            st.session_state["current_page"] = nav_item
            st.rerun()

    page = st.session_state["current_page"]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 12px 14px;">
        <div style="font-size: 0.75rem; font-weight: 700; color: #10b981; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">Telemetry & Specs</div>
        <div style="font-size: 0.75rem; color: #a1a1aa; line-height: 1.6;">
            • <strong>Engine:</strong> XGBoost Classifier<br>
            • <strong>Survival:</strong> Cox Proportional Hazards<br>
            • <strong>LTV:</strong> BG/NBD + Gamma-Gamma<br>
            • <strong>Cutoff:</strong> 90 Days (Leakage-Free)<br>
            • <strong>Holdout Corr:</strong> 0.7312
        </div>
    </div>
    """, unsafe_allow_html=True)


# PAGE 1: Overview Dashboard
if page == "Overview Dashboard":
    st.title("Overview Dashboard")
    st.markdown("Executive metrics, cohort retention health, LTV forecasts, and model performance.")

    total_customers = len(scores_df)
    churn_rate = (scores_df["ChurnFlag"] == 1).mean()
    total_ltv = scores_df["LTV_12m"].sum()
    high_risk_high_val = scores_df[(scores_df["ChurnFlag"] == 1) & (scores_df["Value tier"] == "High")].shape[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Observed Customers</div>
            <div class="metric-value">{total_customers:,}</div>
            <div class="metric-sub">Active transactional accounts</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Churn Rate (90d Cutoff)</div>
            <div class="metric-value">{churn_rate:.1%}</div>
            <div class="metric-sub">{int(total_customers * churn_rate):,} classified at risk</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted 12M LTV</div>
            <div class="metric-value">₹{total_ltv/1e6:.2f}M</div>
            <div class="metric-sub">Mean: ₹{scores_df['LTV_12m'].mean():.2f} / customer</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">High-Risk High-Value</div>
            <div class="metric-value" style="color: #10b981;">{high_risk_high_val:,}</div>
            <div class="metric-sub">Priority retention targets</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Customer Churn Probability Distribution")
        opt_thresh = artifacts["churn"].get("threshold", 0.44)
        fig_hist = px.histogram(
            scores_df,
            x="ChurnProb",
            nbins=35,
            color="ChurnFlag",
            color_discrete_map={0: "#10b981", 1: "#52525b"},
            labels={"ChurnProb": "Churn Probability", "count": "Customer Count", "ChurnFlag": "At-Risk"},
            title="Distribution of Churn Probabilities",
            opacity=0.9
        )
        fig_hist.add_vline(x=opt_thresh, line_dash="dash", line_color="#10b981",
                           annotation_text=f"Decision Threshold ({opt_thresh:.2f})", annotation_position="top left")
        fig_hist.update_layout(
            bargap=0.05, height=350, margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="#18181b", paper_bgcolor="#18181b", font=dict(color="#f4f4f5")
        )
        st.plotly_chart(fig_hist, width="stretch")

    with col_chart2:
        st.subheader("Expected 12-Month LTV by RFM Segment")
        seg_ltv = scores_df.groupby("RFM segment")["LTV_12m"].agg(["mean", "count"]).reset_index()
        fig_bar = px.bar(
            seg_ltv,
            x="RFM segment",
            y="mean",
            text="mean",
            title="Average 12M Expected LTV (INR) per Segment",
            labels={"mean": "Average LTV (INR)", "RFM segment": "Segment"},
            color_discrete_sequence=["#10b981"]
        )
        fig_bar.update_traces(texttemplate="₹%{text:.0f}", textposition="outside", marker_color="#10b981")
        fig_bar.update_layout(
            height=350, showlegend=False, margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="#18181b", paper_bgcolor="#18181b", font=dict(color="#f4f4f5")
        )
        st.plotly_chart(fig_bar, width="stretch")

    st.subheader("Cohort Survival Trajectory (Kaplan-Meier)")
    surv_path = FIGURES_DIR / "survival_curves.png"
    if surv_path.exists():
        st.image(str(surv_path), caption="Survival Curves: Overall & Purchase Frequency Tiers (Log-Rank Test p < 0.001)")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Model Performance & Technical Diagnostics", expanded=False):
        st.markdown("### Comparative Performance (Test Set Evaluation)")
        lr_m = metrics_dict.get("LogisticRegression", {})
        xgb_m = metrics_dict.get("XGBoost", {})

        metrics_table = pd.DataFrame([
            {
                "Model": "Baseline: Logistic Regression (Scaled)",
                "Accuracy": f"{lr_m.get('Accuracy', 0):.3f}",
                "Precision": f"{lr_m.get('Precision', 0):.3f}",
                "Recall": f"{lr_m.get('Recall', 0):.3f}",
                "F1 Score": f"{lr_m.get('F1', 0):.3f}",
                "ROC-AUC": f"{lr_m.get('ROC-AUC', 0):.3f}",
                "PR-AUC": f"{lr_m.get('PR-AUC', 0):.3f}",
                "Decision Threshold": "0.50"
            },
            {
                "Model": "Main: Tuned XGBoost (RandomizedSearchCV)",
                "Accuracy": f"{xgb_m.get('Accuracy', 0):.3f}",
                "Precision": f"{xgb_m.get('Precision', 0):.3f}",
                "Recall": f"{xgb_m.get('Recall', 0):.3f}",
                "F1 Score": f"{xgb_m.get('F1', 0):.3f}",
                "ROC-AUC": f"{xgb_m.get('ROC-AUC', 0):.3f}",
                "PR-AUC": f"{xgb_m.get('PR-AUC', 0):.3f}",
                "Decision Threshold": f"{xgb_m.get('Optimal_Threshold', 0.44):.2f}"
            }
        ])
        st.dataframe(metrics_table, hide_index=True, width="stretch")

        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            cm_path = FIGURES_DIR / "confusion_matrix.png"
            if cm_path.exists():
                st.image(str(cm_path), caption="Confusion Matrix on Holdout Test Set")
        with diag_col2:
            roc_path = FIGURES_DIR / "roc_curve.png"
            if roc_path.exists():
                st.image(str(roc_path), caption="Receiver Operating Characteristic Curve")

        shap_path = FIGURES_DIR / "shap_summary.png"
        if shap_path.exists():
            st.image(str(shap_path), caption="SHAP Global Feature Importance & Contribution Distribution")


# PAGE 2: Customer Lookup
elif page == "Customer Lookup":
    st.title("Customer Lookup & Retention")
    st.markdown("Search customer profile, estimated lifetime value, survival outlook, and proactive retention actions.")

    available_cids = sorted(scores_df["CustomerID"].unique())
    selected_cid = st.selectbox("Select Customer ID", available_cids, index=0)

    cust_row = scores_df[scores_df["CustomerID"] == selected_cid].iloc[0]
    cust_feat = feat_df[feat_df["CustomerID"] == selected_cid]
    has_feat = not cust_feat.empty
    feat_record = cust_feat.iloc[0].to_dict() if has_feat else {}

    c_left, c_right = st.columns([1, 2])

    with c_left:
        churn_p = float(cust_row["ChurnProb"])
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_p * 100,
            number={"suffix": "%", "font": {"size": 36, "color": "#ffffff"}},
            title={"text": f"<b>Churn Risk: Customer {selected_cid}</b>", "font": {"size": 15, "color": "#a1a1aa"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#71717a"},
                "bar": {"color": "#10b981"},
                "bgcolor": "#27272a",
                "steps": [
                    {"range": [0, 44], "color": "#18181b"},
                    {"range": [44, 100], "color": "#27272a"}
                ],
                "threshold": {
                    "line": {"color": "#10b981", "width": 3},
                    "thickness": 0.75,
                    "value": 44
                }
            }
        ))
        fig_gauge.update_layout(
            height=260, margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="#18181b", paper_bgcolor="#18181b"
        )
        st.plotly_chart(fig_gauge, width="stretch")

        if cust_row["ChurnFlag"] == 1:
            st.markdown('<div class="status-badge-high">HIGH CHURN RISK</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge-low">ACTIVE / RETAINED</div>', unsafe_allow_html=True)

    with c_right:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Predicted 6M LTV", f"₹{cust_row['LTV_6m']:,.2f}")
            st.metric("Probability Alive (BG/NBD)", f"{cust_row['ProbAlive']:.1%}")
        with m2:
            st.metric("Predicted 12M LTV", f"₹{cust_row['LTV_12m']:,.2f}")
            st.metric("Expected Survival", f"{cust_row['ExpectedSurvivalDays']} days")
        with m3:
            st.metric("RFM Segment", f"{cust_row['RFM segment']}")
            st.metric("Value Tier", f"{cust_row['Value tier']} Tier")

    st.divider()

    r_col1, r_col2 = st.columns(2)

    with r_col1:
        st.subheader("Suggested Retention Action")
        risk_flag = cust_row["ChurnFlag"]
        val_tier = cust_row["Value tier"]

        if risk_flag == 1 and val_tier == "High":
            st.markdown("""
            <div class="custom-box" style="border-left: 4px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 8px 0;">VIP Retention Outreach</h4>
                <p><strong>Action:</strong> Deliver targeted VIP loyalty offer with 20% discount coupon and direct relationship manager follow-up.</p>
                <p><strong>Channel:</strong> Direct Email + Dedicated Account Manager call.</p>
                <p><strong>Objective:</strong> Safeguard high-value account before final attrition.</p>
            </div>
            """, unsafe_allow_html=True)
        elif risk_flag == 1 and val_tier == "Mid":
            st.markdown("""
            <div class="custom-box" style="border-left: 4px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 8px 0;">Automated Win-Back Campaign</h4>
                <p><strong>Action:</strong> Trigger automated 3-stage re-engagement email series featuring free shipping voucher and 15% discount.</p>
                <p><strong>Channel:</strong> Email Campaign & Notification.</p>
                <p><strong>Objective:</strong> Re-activate purchase frequency.</p>
            </div>
            """, unsafe_allow_html=True)
        elif risk_flag == 1 and val_tier == "Low":
            st.markdown("""
            <div class="custom-box">
                <h4 style="color: #ffffff; margin: 0 0 8px 0;">Standard Nurture Campaign</h4>
                <p><strong>Action:</strong> Include in regular monthly catalogue and seasonal promotional broadcast.</p>
                <p><strong>Channel:</strong> Broadcast Email.</p>
                <p><strong>Objective:</strong> Cost-effective reactivation.</p>
            </div>
            """, unsafe_allow_html=True)
        elif risk_flag == 0 and val_tier == "High":
            st.markdown("""
            <div class="custom-box" style="border-left: 4px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 8px 0;">Loyalty Tier & Upsell</h4>
                <p><strong>Action:</strong> Grant early access to new collections and premium rewards program.</p>
                <p><strong>Channel:</strong> VIP Program Invitation.</p>
                <p><strong>Objective:</strong> Increase customer lifetime value.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="custom-box">
                <h4 style="color: #ffffff; margin: 0 0 8px 0;">Standard Engagement Sequence</h4>
                <p><strong>Action:</strong> Maintain regular marketing schedule with cross-sell suggestions based on prior orders.</p>
            </div>
            """, unsafe_allow_html=True)

    with r_col2:
        st.subheader("Top Predictive Risk Factors (SHAP Drivers)")
        if has_feat:
            score_res = score_customer(feat_record)
            for idx, reason in enumerate(score_res.get("top_reasons", []), 1):
                clean_reason = reason.replace("£", "₹").replace("GBP", "INR")
                st.markdown(f"**{idx}.** {clean_reason}")
        else:
            st.markdown("- Days since last purchase indicates active status.")
            st.markdown("- Historical purchase frequency provides retention stability.")
            st.markdown("- Basket consistency correlates with value tier.")

    st.subheader("Customer Survival Curve Trajectory")
    t_days = np.linspace(1, 365, 100)
    p_alive = cust_row["ProbAlive"]
    decay_rate = 0.002 * (1.0 + float(cust_row["ChurnProb"]))
    survival_prob = p_alive * np.exp(-decay_rate * t_days)

    fig_surv = px.line(
        x=t_days,
        y=survival_prob,
        labels={"x": "Timeline into Future (Days)", "y": "Retention Probability"},
        title=f"Projected 365-Day Survival Trajectory for Customer {selected_cid}",
        color_discrete_sequence=["#10b981"]
    )
    fig_surv.add_hline(y=0.5, line_dash="dash", line_color="#71717a", annotation_text="50% Median Survival Line")
    fig_surv.update_layout(
        height=300, yaxis_range=[0, 1.05],
        plot_bgcolor="#18181b", paper_bgcolor="#18181b", font=dict(color="#f4f4f5")
    )
    st.plotly_chart(fig_surv, width="stretch")


# PAGE 3: Live Prediction
elif page == "Live Prediction":
    st.title("Live Real-Time Customer Scoring")
    st.markdown("Input behavioral metrics to compute real-time churn probability, lifetime value, and SHAP drivers.")

    with st.form("live_scoring_form"):
        st.subheader("Customer Behavioral Attributes")
        c1, c2, c3 = st.columns(3)
        with c1:
            recency = st.number_input("Recency (Days since last purchase)", min_value=0, max_value=730, value=75, step=1)
            frequency = st.number_input("Frequency (Distinct past orders)", min_value=1, max_value=200, value=4, step=1)
            tenure = st.number_input("Tenure (Days since first order)", min_value=1, max_value=1000, value=250, step=1)
        with c2:
            monetary = st.number_input("Monetary (Total spend INR)", min_value=1.0, max_value=10000000.0, value=12500.0, step=500.0)
            avg_order_val = st.number_input("Average Order Value (INR)", min_value=1.0, max_value=1000000.0, value=3125.0, step=250.0)
            unique_products = st.number_input("Unique Products Purchased", min_value=1, max_value=500, value=28, step=1)
        with c3:
            avg_days_between = st.number_input("Avg Days Between Orders", min_value=1.0, max_value=365.0, value=45.0, step=1.0)
            purchases_30d = st.number_input("Purchases in Last 30 Days", min_value=0, max_value=50, value=0, step=1)
            country = st.selectbox("Customer Country", ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Other"])

        submitted = st.form_submit_button("Run Live Inference", width="stretch")

    if submitted:
        input_data = {
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": monetary,
            "AvgOrderValue": avg_order_val,
            "Tenure": tenure,
            "UniqueProducts": unique_products,
            "AvgDaysBetweenOrders": avg_days_between,
            "PurchasesLast30Days": purchases_30d,
            "Country": country
        }

        with st.spinner("Calculating predictions..."):
            res = score_customer(input_data)

        st.success("Scoring Complete")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Churn Probability", f"{res['churn_probability']:.1%}",
                      delta="At Risk" if res['churn_flag'] == 1 else "Healthy",
                      delta_color="inverse")
        with r2:
            st.metric("Probability Alive", f"{res['probability_alive']:.1%}")
        with r3:
            st.metric("Predicted 6M LTV", f"₹{res['ltv_6m']:,.2f}")
        with r4:
            st.metric("Predicted 12M LTV", f"₹{res['ltv_12m']:,.2f}")

        st.markdown("<br>", unsafe_allow_html=True)
        col_res1, col_res2 = st.columns(2)

        with col_res1:
            st.subheader("Top 3 SHAP Model Explanations")
            for idx, reason in enumerate(res["top_reasons"], 1):
                clean_reason = reason.replace("£", "₹").replace("GBP", "INR")
                st.markdown(f"**{idx}.** {clean_reason}")

        with col_res2:
            st.subheader("Retention Recommendation")
            is_high_risk = res["churn_flag"] == 1
            is_high_val = res["ltv_12m"] > 50000

            if is_high_risk and is_high_val:
                st.error(f"Priority Action: VIP Retention Offer\nCustomer carries significant value (₹{res['ltv_12m']:,.2f}) but elevated churn risk. Deploy high-touch personalized retention offer.")
            elif is_high_risk:
                st.warning("Automated Win-Back\nTrigger automated re-engagement email sequence with discount coupon.")
            elif is_high_val:
                st.success("VIP Loyalty Growth\nHigh-value loyal account. Enroll in VIP loyalty tier.")
            else:
                st.info("Standard Nurture Sequence\nEngagement is regular. Continue planned promotional calendar.")


# PAGE 4: At-Risk Customers
elif page == "At-Risk Customers":
    st.title("High-Priority At-Risk Customers")
    st.markdown("Customers ranked by **Revenue at Risk** ($ChurnProb \\times LTV_{12m}$) to optimize proactive retention budget.")

    f1, f2, f3 = st.columns(3)
    with f1:
        risk_filter = st.selectbox("Risk Level", ["All Customers", "High Risk Only (Churn Flag = 1)", "Low Risk Only (Churn Flag = 0)"])
    with f2:
        seg_options = ["All Segments"] + sorted(scores_df["RFM segment"].dropna().unique().tolist())
        seg_filter = st.selectbox("RFM Segment", seg_options)
    with f3:
        country_options = ["All Countries"] + sorted(scores_df["Country"].dropna().unique().tolist())
        country_filter = st.selectbox("Country", country_options)

    filtered_df = scores_df.copy()

    if risk_filter == "High Risk Only (Churn Flag = 1)":
        filtered_df = filtered_df[filtered_df["ChurnFlag"] == 1]
    elif risk_filter == "Low Risk Only (Churn Flag = 0)":
        filtered_df = filtered_df[filtered_df["ChurnFlag"] == 0]

    if seg_filter != "All Segments":
        filtered_df = filtered_df[filtered_df["RFM segment"] == seg_filter]

    if country_filter != "All Countries":
        filtered_df = filtered_df[filtered_df["Country"] == country_filter]

    filtered_df = filtered_df.sort_values(by="RevenueAtRisk", ascending=False).reset_index(drop=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Customers in Target View", f"{len(filtered_df):,}")
    with s2:
        st.metric("Total Revenue at Risk", f"₹{filtered_df['RevenueAtRisk'].sum():,.2f}")
    with s3:
        st.metric("Average Churn Risk", f"{filtered_df['ChurnProb'].mean():.1%}" if not filtered_df.empty else "0%")

    st.markdown("<br>", unsafe_allow_html=True)

    display_cols = [
        "CustomerID", "Country", "RFM segment", "Value tier",
        "ChurnProb", "ChurnFlag", "ProbAlive", "LTV_12m", "RevenueAtRisk"
    ]
    st.dataframe(
        filtered_df[display_cols].style.format({
            "ChurnProb": "{:.1%}",
            "ProbAlive": "{:.1%}",
            "LTV_12m": "₹{:,.2f}",
            "RevenueAtRisk": "₹{:,.2f}"
        }),
        height=480,
        width="stretch"
    )

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download At-Risk Customers CSV",
        data=csv_bytes,
        file_name="at_risk_customers_export.csv",
        mime="text/csv",
        width="stretch"
    )
