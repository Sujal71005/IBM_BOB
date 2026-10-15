"""
LoanVision AI — Main Streamlit Application
==========================================
Run: streamlit run app.py
"""

import os
import sys
from pathlib import Path

# Ensure src/ is importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_processing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COL,
    VALID_CATEGORIES,
    build_preprocessor,
    get_X_y,
    load_clean_dataset,
    split_data,
)
from src.evaluation import (
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_roc_curves,
)
from src.explainability import (
    get_global_feature_importance,
    get_local_explanation,
    get_permutation_importance,
    plot_global_importance,
    plot_local_explanation,
)
from src.model_training import (
    evaluate_all_models,
    load_best_pipeline,
    load_metrics,
    run_full_training,
    save_best_pipeline,
    select_best_model,
    train_all_models,
)
from src.prediction import predict

# ─── Page configuration ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LoanVision AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* KPI cards */
    .kpi-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,.06);
    }
    .kpi-value { font-size: 1.9rem; font-weight: 700; color: #1e3a5f; margin: 0; }
    .kpi-label { font-size: 0.78rem; color: #64748b; margin: 4px 0 0 0; text-transform: uppercase; letter-spacing: .05em; }

    /* Prediction result */
    .pred-approved {
        background: #f0fdf4; border: 2px solid #22c55e; border-radius: 12px; padding: 24px;
        text-align: center;
    }
    .pred-rejected {
        background: #fff1f2; border: 2px solid #ef4444; border-radius: 12px; padding: 24px;
        text-align: center;
    }
    .pred-title { font-size: 2rem; font-weight: 800; margin: 0; }
    .pred-prob  { font-size: 1.1rem; color: #475569; margin: 8px 0 0 0; }

    /* Section headers */
    .section-header {
        font-size: 1.4rem; font-weight: 700; color: #1e3a5f;
        border-bottom: 2px solid #3b82d4; padding-bottom: 6px; margin-bottom: 16px;
    }

    /* Disclaimer */
    .disclaimer {
        background: #fffbeb; border: 1px solid #f59e0b; border-radius: 8px;
        padding: 12px 16px; font-size: 0.85rem; color: #78350f;
    }

    /* Main titles */
    h1 { color: #1e3a5f !important; }
    h2 { color: #1e3a5f !important; }
</style>
""", unsafe_allow_html=True)

# ─── Cached data loader ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_data():
    df, val_report, clean_log = load_clean_dataset()
    return df, val_report, clean_log


@st.cache_data(show_spinner=False)
def get_train_test_split(_df):
    X, y = get_X_y(_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test


# ─── Sidebar navigation ───────────────────────────────────────────────────────────

PAGES = {
    "🏠  Executive Overview":       "overview",
    "📊  Exploratory Analysis":     "eda",
    "🤖  Model Lab":                "model_lab",
    "🎯  Loan Predictor":           "predictor",
    "🔍  Model Explainability":     "explainability",
    "📋  Data Explorer":            "data_explorer",
    "ℹ️  Project Info":             "info",
}

with st.sidebar:
    st.markdown("## 🏦 LoanVision AI")
    st.markdown("*Educational AI Prototype*")
    st.markdown("---")
    page_name = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    page = PAGES[page_name]
    st.markdown("---")
    st.markdown("""
    <div style='font-size:.75rem;color:#94a3b8;'>
    ⚠️ For educational use only.<br>
    Not a real lending decision system.
    </div>
    """, unsafe_allow_html=True)

# ─── Load data (all pages) ────────────────────────────────────────────────────────

try:
    df, val_report, clean_log = get_data()
    data_ok = True
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ════════════════════════════════════════════════════════════════════════════════
# PAGE A — Executive Overview
# ════════════════════════════════════════════════════════════════════════════════

if page == "overview":
    st.title("🏦 LoanVision AI — Executive Overview")
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Disclaimer:</strong> This is an educational analytics prototype using a synthetic/educational dataset.
    It must not be used to make actual lending or credit decisions.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    # ── KPI metrics ──────────────────────────────────────────────────────────────
    total = len(df)
    approved = int(df[TARGET_COL].sum())
    rejected = total - approved
    approval_rate = approved / total * 100
    rejection_rate = 100 - approval_rate
    avg_income = df["Applicant_Income"].mean()
    avg_credit = df["Credit_Score"].mean()
    avg_loan = df["Loan_Amount"].mean()

    kpis = [
        ("Total Applications", f"{total:,}"),
        ("Approved", f"{approved:,}"),
        ("Rejected", f"{rejected:,}"),
        ("Approval Rate", f"{approval_rate:.1f}%"),
        ("Rejection Rate", f"{rejection_rate:.1f}%"),
        ("Avg. Income", f"${avg_income:,.0f}"),
        ("Avg. Credit Score", f"{avg_credit:.0f}"),
        ("Avg. Loan Amount", f"${avg_loan:,.0f}"),
    ]

    cols = st.columns(4)
    for i, (label, value) in enumerate(kpis):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="kpi-card">
                <p class="kpi-value">{value}</p>
                <p class="kpi-label">{label}</p>
            </div>
            """, unsafe_allow_html=True)
        if i == 3:
            cols = st.columns(4)

    st.markdown("---")

    # ── Charts row 1 ──────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<p class="section-header">Approval vs Rejection</p>', unsafe_allow_html=True)
        fig_donut = go.Figure(go.Pie(
            labels=["Approved", "Rejected"],
            values=[approved, rejected],
            hole=0.55,
            marker_colors=["#22c55e", "#ef4444"],
        ))
        fig_donut.update_layout(height=320, margin=dict(t=20, b=20, l=20, r=20),
                                showlegend=True, legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        st.markdown('<p class="section-header">Loan Purpose Distribution</p>', unsafe_allow_html=True)
        purpose_counts = df["Loan_Purpose"].value_counts().reset_index()
        purpose_counts.columns = ["Purpose", "Count"]
        fig_purpose = px.bar(purpose_counts, x="Purpose", y="Count",
                             color="Count", color_continuous_scale="Blues",
                             labels={"Count": "Applications"})
        fig_purpose.update_layout(height=320, margin=dict(t=20, b=40, l=20, r=20),
                                  coloraxis_showscale=False)
        st.plotly_chart(fig_purpose, use_container_width=True)

    # ── Charts row 2 ──────────────────────────────────────────────────────────────
    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<p class="section-header">Employment Status Distribution</p>', unsafe_allow_html=True)
        emp_counts = df["Employment_Status"].value_counts().reset_index()
        emp_counts.columns = ["Status", "Count"]
        fig_emp = px.bar(emp_counts, x="Status", y="Count",
                         color="Count", color_continuous_scale="Purples",
                         labels={"Count": "Applications"})
        fig_emp.update_layout(height=300, margin=dict(t=10, b=40, l=20, r=20),
                              coloraxis_showscale=False)
        st.plotly_chart(fig_emp, use_container_width=True)

    with c4:
        st.markdown('<p class="section-header">Credit Score Distribution</p>', unsafe_allow_html=True)
        fig_cs = px.histogram(df, x="Credit_Score", nbins=25, color_discrete_sequence=["#3b82d4"])
        fig_cs.update_layout(height=300, margin=dict(t=10, b=40, l=20, r=20),
                             xaxis_title="Credit Score", yaxis_title="Count")
        st.plotly_chart(fig_cs, use_container_width=True)

    # ── Recent records ────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Representative Records (first 10)</p>', unsafe_allow_html=True)
    display_cols = ["Applicant_Income", "Credit_Score", "Loan_Amount", "Employment_Status",
                    "Loan_Purpose", "DTI_Ratio", TARGET_COL]
    show_df = df[display_cols].head(10).copy()
    show_df[TARGET_COL] = show_df[TARGET_COL].map({1: "✅ Approved", 0: "❌ Rejected"})
    st.dataframe(show_df, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE B — Exploratory Data Analysis
# ════════════════════════════════════════════════════════════════════════════════

elif page == "eda":
    st.title("📊 Exploratory Data Analysis")
    st.caption("All charts are based on the cleaned historical dataset. Patterns shown are correlational, not causal.")

    # ── Sidebar filters ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Filters")
        outcome_filter = st.multiselect("Outcome", ["Approved", "Rejected"], default=["Approved", "Rejected"])
        emp_filter = st.multiselect("Employment Status",
                                    VALID_CATEGORIES["Employment_Status"],
                                    default=VALID_CATEGORIES["Employment_Status"])
        cs_range = st.slider("Credit Score Range", 550, 800, (550, 800))
        income_range = st.slider("Applicant Income Range",
                                 int(df["Applicant_Income"].min()),
                                 int(df["Applicant_Income"].max()),
                                 (int(df["Applicant_Income"].min()), int(df["Applicant_Income"].max())))
        loan_range = st.slider("Loan Amount Range",
                               int(df["Loan_Amount"].min()),
                               int(df["Loan_Amount"].max()),
                               (int(df["Loan_Amount"].min()), int(df["Loan_Amount"].max())))
        if st.button("Reset Filters"):
            st.rerun()

    # Apply filters
    outcome_map = {"Approved": 1, "Rejected": 0}
    outcome_vals = [outcome_map[o] for o in outcome_filter]
    mask = (
        df[TARGET_COL].isin(outcome_vals) &
        df["Employment_Status"].isin(emp_filter) &
        df["Credit_Score"].between(cs_range[0], cs_range[1]) &
        df["Applicant_Income"].between(income_range[0], income_range[1]) &
        df["Loan_Amount"].between(loan_range[0], loan_range[1])
    )
    fdf = df[mask].copy()
    st.info(f"Showing **{len(fdf):,}** of {len(df):,} records after filters.")

    if fdf.empty:
        st.warning("No records match the current filters. Please adjust them.")
        st.stop()

    fdf["Outcome"] = fdf[TARGET_COL].map({1: "Approved", 0: "Rejected"})

    # ── Distributions ─────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Feature Distributions</p>', unsafe_allow_html=True)
    dist_cols = st.columns(2)
    num_cols_to_plot = [
        ("Applicant_Income", "Applicant Income"),
        ("Credit_Score", "Credit Score"),
        ("Loan_Amount", "Loan Amount"),
        ("DTI_Ratio", "Debt-to-Income Ratio"),
        ("Loan_Term", "Loan Term (months)"),
        ("Savings", "Savings"),
    ]
    for i, (col, label) in enumerate(num_cols_to_plot):
        with dist_cols[i % 2]:
            fig = px.histogram(fdf, x=col, color="Outcome", barmode="overlay", nbins=30,
                               color_discrete_map={"Approved": "#22c55e", "Rejected": "#ef4444"},
                               labels={col: label}, title=f"{label} Distribution by Outcome")
            fig.update_layout(height=280, margin=dict(t=40, b=40, l=20, r=20),
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, use_container_width=True)

    # ── Approval rates by category ────────────────────────────────────────────────
    st.markdown('<p class="section-header">Approval Rates by Category</p>', unsafe_allow_html=True)
    cat_plot_cols = [
        ("Employment_Status", "Employment Status"),
        ("Property_Area", "Property Area"),
        ("Loan_Purpose", "Loan Purpose"),
        ("Education_Level", "Education Level"),
    ]
    cat_display_cols = st.columns(2)
    for i, (col, label) in enumerate(cat_plot_cols):
        grp = fdf.groupby(col)[TARGET_COL].agg(["sum", "count"]).reset_index()
        grp["Approval Rate (%)"] = grp["sum"] / grp["count"] * 100
        with cat_display_cols[i % 2]:
            fig = px.bar(grp, x=col, y="Approval Rate (%)",
                         text=grp["Approval Rate (%)"].map(lambda x: f"{x:.1f}%"),
                         color="Approval Rate (%)",
                         color_continuous_scale="Blues",
                         title=f"Approval Rate by {label}")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=320, coloraxis_showscale=False,
                              margin=dict(t=50, b=60, l=20, r=20), yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

    # ── Credit score group approval rates ─────────────────────────────────────────
    st.markdown('<p class="section-header">Approval Rate Across Credit Score Groups</p>', unsafe_allow_html=True)
    fdf["CS_Group"] = pd.cut(fdf["Credit_Score"],
                             bins=[549, 599, 649, 699, 749, 800],
                             labels=["550–599", "600–649", "650–699", "700–749", "750–799"])
    cs_grp = fdf.groupby("CS_Group", observed=True)[TARGET_COL].agg(["sum", "count"]).reset_index()
    cs_grp["Approval Rate (%)"] = cs_grp["sum"] / cs_grp["count"] * 100
    fig_cs_grp = px.bar(cs_grp, x="CS_Group", y="Approval Rate (%)",
                        text=cs_grp["Approval Rate (%)"].map(lambda x: f"{x:.1f}%"),
                        color="Approval Rate (%)", color_continuous_scale="Greens",
                        labels={"CS_Group": "Credit Score Band"},
                        title="Approval Rate by Credit Score Band")
    fig_cs_grp.update_traces(textposition="outside")
    fig_cs_grp.update_layout(height=340, coloraxis_showscale=False, yaxis_range=[0, 100])
    st.plotly_chart(fig_cs_grp, use_container_width=True)

    # ── Scatter plots ──────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Scatter Analysis</p>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        fig_sc1 = px.scatter(fdf, x="Applicant_Income", y="Loan_Amount",
                             color="Outcome",
                             color_discrete_map={"Approved": "#22c55e", "Rejected": "#ef4444"},
                             opacity=0.6,
                             labels={"Applicant_Income": "Applicant Income", "Loan_Amount": "Loan Amount"},
                             title="Loan Amount vs. Applicant Income")
        fig_sc1.update_layout(height=380, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_sc1, use_container_width=True)
    with s2:
        fig_sc2 = px.scatter(fdf, x="Credit_Score", y="DTI_Ratio",
                             color="Outcome",
                             color_discrete_map={"Approved": "#22c55e", "Rejected": "#ef4444"},
                             opacity=0.6,
                             labels={"Credit_Score": "Credit Score", "DTI_Ratio": "Debt-to-Income Ratio"},
                             title="Credit Score vs. DTI Ratio")
        fig_sc2.update_layout(height=380, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_sc2, use_container_width=True)

    # ── Correlation heatmap ────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Numerical Feature Correlation Heatmap</p>', unsafe_allow_html=True)
    corr_cols = NUMERICAL_FEATURES + [TARGET_COL]
    corr = fdf[corr_cols].corr().round(2)
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, title="Pearson Correlation Heatmap",
                         aspect="auto")
    fig_corr.update_layout(height=480)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("⚠️ Correlation does not imply causation.")

    # ── Box plots: approved vs rejected ────────────────────────────────────────────
    st.markdown('<p class="section-header">Feature Comparison: Approved vs Rejected</p>', unsafe_allow_html=True)
    box_cols = st.columns(2)
    box_features = ["Credit_Score", "Applicant_Income", "Loan_Amount", "DTI_Ratio"]
    for i, col in enumerate(box_features):
        with box_cols[i % 2]:
            fig_box = px.box(fdf, x="Outcome", y=col, color="Outcome",
                             color_discrete_map={"Approved": "#22c55e", "Rejected": "#ef4444"},
                             title=f"{col} by Outcome")
            fig_box.update_layout(height=300, showlegend=False,
                                  margin=dict(t=50, b=40))
            st.plotly_chart(fig_box, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE C — Model Lab
# ════════════════════════════════════════════════════════════════════════════════

elif page == "model_lab":
    st.title("🤖 Machine Learning Model Lab")
    st.caption(
        "Three classifiers are trained and compared on held-out test data. "
        "Best model is selected by ROC-AUC — chosen because the dataset is class-imbalanced (~31% approvals)."
    )

    # ── Train button ──────────────────────────────────────────────────────────────
    saved_metrics = load_metrics()
    model_exists = Path("models/loanvision_pipeline.joblib").exists()

    if model_exists and saved_metrics:
        st.success(f"✅ Trained model loaded: **{saved_metrics['best_model']}** | "
                   f"ROC-AUC = {saved_metrics['all_metrics'][0]['roc_auc']:.4f}")
    else:
        st.warning("No trained model found. Click **Train Models** to train and evaluate.")

    if st.button("🔄 Train / Retrain All Models", type="primary"):
        with st.spinner("Training all models… this may take 30–60 seconds."):
            progress_bar = st.progress(0, "Starting…")
            statuses = []

            def cb(name, step, total):
                statuses.append(name)
                progress_bar.progress(step / total, f"Trained: {name}")

            result = run_full_training(progress_callback=cb)
            st.cache_data.clear()

        st.success(f"✅ Training complete! Best model: **{result['best_model']}**")
        saved_metrics = load_metrics()

    if not saved_metrics:
        st.info("Train the models above to see results.")
        st.stop()

    # ── Metrics table ──────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Model Comparison — Held-out Test Set</p>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame(saved_metrics["all_metrics"])[
        ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ].rename(columns={
        "model": "Model", "accuracy": "Accuracy", "precision": "Precision",
        "recall": "Recall", "f1": "F1-Score", "roc_auc": "ROC-AUC",
    })
    st.dataframe(metrics_df.style.highlight_max(
        subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        color="#bbf7d0"
    ), use_container_width=True)

    best_model_name = saved_metrics["best_model"]
    best_metrics = next(m for m in saved_metrics["all_metrics"] if m["model"] == best_model_name)
    st.info(
        f"**Selected model:** {best_model_name} | "
        f"Accuracy={best_metrics['accuracy']:.4f} | "
        f"F1={best_metrics['f1']:.4f} | "
        f"ROC-AUC={best_metrics['roc_auc']:.4f}"
    )

    # ── Grouped bar chart ──────────────────────────────────────────────────────────
    fig_compare = plot_metrics_comparison(saved_metrics["all_metrics"])
    st.plotly_chart(fig_compare, use_container_width=True)

    # ── Confusion matrices ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Confusion Matrices</p>', unsafe_allow_html=True)
    cm_cols = st.columns(3)
    for i, m in enumerate(saved_metrics["all_metrics"]):
        with cm_cols[i]:
            fig_cm = plot_confusion_matrix(m["confusion_matrix"], m["model"])
            st.plotly_chart(fig_cm, use_container_width=True)

    # ── ROC Curves — need fitted models ───────────────────────────────────────────
    with st.expander("📈 ROC Curves (requires re-training in this session)"):
        if "fitted_pipelines" not in st.session_state:
            if st.button("Generate ROC Curves (will re-train in memory)"):
                with st.spinner("Training for ROC curves…"):
                    X_train, X_test, y_train, y_test = get_train_test_split(df)
                    prep = build_preprocessor()
                    fitted = train_all_models(X_train, y_train, X_test, y_test, prep)
                    st.session_state["fitted_pipelines"] = fitted
                    st.session_state["X_test_roc"] = X_test
                    st.session_state["y_test_roc"] = y_test
        else:
            fig_roc = plot_roc_curves(
                st.session_state["fitted_pipelines"],
                st.session_state["X_test_roc"],
                st.session_state["y_test_roc"],
            )
            st.plotly_chart(fig_roc, use_container_width=True)

    # ── Model info ─────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Training Configuration</p>', unsafe_allow_html=True)
    st.markdown("""
    | Setting | Value |
    |---------|-------|
    | Train/Test Split | 80% / 20% |
    | Split Strategy | Stratified (preserves class ratio) |
    | Random Seed | 42 |
    | Numerical Imputation | Median |
    | Categorical Imputation | Most frequent |
    | Scaling | Standard Scaler |
    | Encoding | One-Hot Encoding (unknown categories ignored) |
    | Class Imbalance | Handled via `class_weight='balanced'` (LR, RF) |
    | Selection Criterion | Highest ROC-AUC on held-out test set |
    """)
    st.markdown("""
    **Limitations:**
    - Models are trained on a synthetic/educational dataset. Real-world performance will differ.
    - The held-out test set was used once for final evaluation to avoid optimistic bias.
    - No hyperparameter tuning was applied; results reflect default + manually chosen settings.
    """)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE D — Loan Approval Predictor
# ════════════════════════════════════════════════════════════════════════════════

elif page == "predictor":
    st.title("🎯 Loan Approval Predictor")
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Educational tool only.</strong> Predictions are generated by a model trained on a synthetic dataset
    and are <em>not</em> real lending decisions. Probabilities are model-estimated and not calibrated
    against real-world outcomes. Never use this tool for actual credit decisions.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    # ── Load model ────────────────────────────────────────────────────────────────
    try:
        pipeline, model_name = load_best_pipeline()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("👉 Go to **Model Lab** and click **Train Models** first.")
        st.stop()

    st.info(f"Using model: **{model_name}**")

    # ── Input form ────────────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        st.markdown('<p class="section-header">Applicant Details</p>', unsafe_allow_html=True)

        row1_cols = st.columns(3)
        with row1_cols[0]:
            applicant_income = st.number_input("Applicant Income ($)", min_value=0.0, max_value=100000.0,
                                               value=10000.0, step=500.0,
                                               help="Gross monthly or annual income of the primary applicant.")
        with row1_cols[1]:
            coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0.0, max_value=100000.0,
                                                 value=0.0, step=500.0)
        with row1_cols[2]:
            loan_amount = st.number_input("Loan Amount ($)", min_value=1000.0, max_value=50000.0,
                                          value=15000.0, step=500.0)

        row2_cols = st.columns(3)
        with row2_cols[0]:
            credit_score = st.slider("Credit Score", min_value=550, max_value=800, value=680)
        with row2_cols[1]:
            dti_ratio = st.slider("Debt-to-Income Ratio", min_value=0.10, max_value=0.60, value=0.35, step=0.01,
                                  help="Total monthly debt / gross monthly income.")
        with row2_cols[2]:
            loan_term = st.selectbox("Loan Term (months)", options=[12, 24, 36, 48, 60, 72, 84], index=4)

        row3_cols = st.columns(3)
        with row3_cols[0]:
            age = st.number_input("Age", min_value=18, max_value=75, value=35)
        with row3_cols[1]:
            savings = st.number_input("Savings ($)", min_value=0.0, max_value=50000.0, value=10000.0, step=500.0)
        with row3_cols[2]:
            collateral_value = st.number_input("Collateral Value ($)", min_value=0.0, max_value=100000.0,
                                               value=20000.0, step=500.0)

        row4_cols = st.columns(3)
        with row4_cols[0]:
            existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=1)
        with row4_cols[1]:
            dependents = st.number_input("Dependents", min_value=0, max_value=5, value=1)
        with row4_cols[2]:
            employment_status = st.selectbox("Employment Status",
                                             VALID_CATEGORIES["Employment_Status"])

        row5_cols = st.columns(3)
        with row5_cols[0]:
            marital_status = st.selectbox("Marital Status", VALID_CATEGORIES["Marital_Status"])
        with row5_cols[1]:
            education_level = st.selectbox("Education Level", VALID_CATEGORIES["Education_Level"])
        with row5_cols[2]:
            gender = st.selectbox("Gender", VALID_CATEGORIES["Gender"])

        row6_cols = st.columns(3)
        with row6_cols[0]:
            loan_purpose = st.selectbox("Loan Purpose", VALID_CATEGORIES["Loan_Purpose"])
        with row6_cols[1]:
            property_area = st.selectbox("Property Area", VALID_CATEGORIES["Property_Area"])
        with row6_cols[2]:
            employer_category = st.selectbox("Employer Category", VALID_CATEGORIES["Employer_Category"])

        submitted = st.form_submit_button("🔮 Predict Loan Outcome", type="primary", use_container_width=True)

    # ── Prediction ────────────────────────────────────────────────────────────────
    if submitted:
        form_data = {
            "Applicant_Income": float(applicant_income),
            "Coapplicant_Income": float(coapplicant_income),
            "Employment_Status": employment_status,
            "Age": float(age),
            "Marital_Status": marital_status,
            "Dependents": float(dependents),
            "Credit_Score": float(credit_score),
            "Existing_Loans": float(existing_loans),
            "DTI_Ratio": float(dti_ratio),
            "Savings": float(savings),
            "Collateral_Value": float(collateral_value),
            "Loan_Amount": float(loan_amount),
            "Loan_Term": float(loan_term),
            "Loan_Purpose": loan_purpose,
            "Property_Area": property_area,
            "Education_Level": education_level,
            "Gender": gender,
            "Employer_Category": employer_category,
        }

        with st.spinner("Running prediction…"):
            try:
                result = predict(pipeline, form_data)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()

        label = result["label"]
        prob = result.get("approved_probability")

        st.markdown("---")
        if label == "Approved":
            st.markdown(f"""
            <div class="pred-approved">
                <p class="pred-title" style="color:#16a34a;">✅ Likely Approved</p>
                {'<p class="pred-prob">Model-estimated approval probability: <strong>' + f'{prob*100:.1f}%</strong></p>' if prob is not None else ''}
                <p style="font-size:.82rem;color:#166534;margin-top:10px;">
                This is a model-estimated probability, not a guarantee or official bank decision.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-rejected">
                <p class="pred-title" style="color:#dc2626;">❌ Likely Rejected</p>
                {'<p class="pred-prob">Model-estimated approval probability: <strong>' + f'{prob*100:.1f}%</strong></p>' if prob is not None else ''}
                <p style="font-size:.82rem;color:#991b1b;margin-top:10px;">
                This is a model-estimated probability, not an official credit decision.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # ── Probability gauge ──────────────────────────────────────────────────────
        if prob is not None:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                title={"text": "Approval Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#22c55e" if label == "Approved" else "#ef4444"},
                    "steps": [
                        {"range": [0, 40], "color": "#fee2e2"},
                        {"range": [40, 60], "color": "#fef9c3"},
                        {"range": [60, 100], "color": "#dcfce7"},
                    ],
                },
                number={"suffix": "%"},
            ))
            fig_gauge.update_layout(height=260, margin=dict(t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Local explanation ──────────────────────────────────────────────────────
        st.markdown('<p class="section-header">Key Factors Influencing This Prediction</p>', unsafe_allow_html=True)
        try:
            local_df = get_local_explanation(pipeline, form_data)
            fig_local = plot_local_explanation(local_df, label, top_n=10)
            st.plotly_chart(fig_local, use_container_width=True)
            st.caption(
                "Chart shows which features had the greatest influence on this specific prediction. "
                "🟢 Green = associated with approval. 🔴 Red = associated with rejection. "
                "🔵 Blue = direction unclear for this model type. "
                "Associations are not causal."
            )
        except Exception as e:
            st.warning(f"Local explanation unavailable: {e}")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE E — Model Explainability
# ════════════════════════════════════════════════════════════════════════════════

elif page == "explainability":
    st.title("🔍 Model Explainability")
    st.caption(
        "Understand how the model makes predictions. "
        "All explanations are approximations — they show associations, not causes."
    )

    try:
        pipeline, model_name = load_best_pipeline()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("👉 Go to **Model Lab** and click **Train Models** first.")
        st.stop()

    st.info(f"Explaining: **{model_name}**")

    # ── Global feature importance ──────────────────────────────────────────────────
    st.markdown('<p class="section-header">Global Feature Importance</p>', unsafe_allow_html=True)
    global_imp = get_global_feature_importance(pipeline)
    fig_global = plot_global_importance(global_imp)
    st.plotly_chart(fig_global, use_container_width=True)
    st.dataframe(global_imp.rename(columns={"feature": "Feature", "importance": "Importance Score",
                                            "direction": "Direction (+1=Approval, -1=Rejection, 0=Unknown)"}),
                 use_container_width=True)

    clf_type = type(pipeline.named_steps["classifier"]).__name__
    if "GradientBoosting" in clf_type or "RandomForest" in clf_type:
        st.caption(
            "Importance scores are from the fitted tree's `feature_importances_` attribute, "
            "aggregated back to original features. They measure how much each feature reduces impurity. "
            "Direction is not directly derivable from tree importances alone — see local explanations for direction."
        )
    else:
        st.caption(
            "Importance shows absolute coefficient magnitude. "
            "Green = positive association with approval. Red = negative. "
            "Scores are aggregated from one-hot-encoded sub-features."
        )

    # ── Permutation importance ─────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Permutation Importance (Held-out Test Data)</p>', unsafe_allow_html=True)
    st.caption(
        "Permutation importance measures how much the ROC-AUC drops when a feature's values are shuffled. "
        "Computed on held-out test data only. Features with near-zero importance can often be removed without "
        "significant performance loss."
    )
    if st.button("Compute Permutation Importance (may take ~20s)"):
        with st.spinner("Computing permutation importance…"):
            X_train, X_test, y_train, y_test = get_train_test_split(df)
            perm_df = get_permutation_importance(pipeline, X_test, y_test, n_repeats=10)
            st.session_state["perm_df"] = perm_df

    if "perm_df" in st.session_state:
        perm_df = st.session_state["perm_df"]
        fig_perm = go.Figure(go.Bar(
            x=perm_df["importance_mean"],
            y=perm_df["feature"],
            orientation="h",
            error_x=dict(type="data", array=perm_df["importance_std"]),
            marker_color="#3b82d4",
        ))
        fig_perm.update_layout(
            title="Permutation Importance (mean ± std, n_repeats=10)",
            xaxis_title="Mean Decrease in ROC-AUC",
            height=max(350, len(perm_df) * 28),
            margin=dict(l=150, r=20, t=60, b=40),
        )
        st.plotly_chart(fig_perm, use_container_width=True)
        st.dataframe(perm_df.rename(columns={
            "feature": "Feature",
            "importance_mean": "Mean Drop in AUC",
            "importance_std": "Std Dev",
        }), use_container_width=True)

    # ── Responsible AI notes ───────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Responsible AI Considerations</p>', unsafe_allow_html=True)
    st.markdown("""
    **Potential Bias Sources:**
    - The model was trained on a synthetic educational dataset. Historical biases in real lending data
      (e.g., discrimination by gender, race, or area) are not fully captured or mitigated here.
    - Features like `Gender`, `Marital_Status`, and `Property_Area` may act as proxies for
      protected characteristics.
    - `class_weight='balanced'` was used for Logistic Regression and Random Forest to mitigate
      class imbalance, but this does not guarantee fairness across demographic groups.

    **Limitations of These Explanations:**
    - Feature importance aggregation from one-hot encoding can obscure nuances within categories.
    - Permutation importance may underestimate correlated features.
    - Local explanations are approximations, not exact attributions.
    - None of these explanations constitute a complete audit.

    **What this model should NOT be used for:**
    - Actual lending decisions
    - Credit scoring of real individuals
    - Regulatory compliance evaluation
    """)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE F — Data Explorer
# ════════════════════════════════════════════════════════════════════════════════

elif page == "data_explorer":
    st.title("📋 Data Explorer")
    st.caption("Browse, filter, and download the cleaned historical loan application dataset.")

    # ── Filters ───────────────────────────────────────────────────────────────────
    with st.expander("🔽 Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            outcome_sel = st.multiselect("Loan Outcome", ["Approved", "Rejected"],
                                         default=["Approved", "Rejected"])
            emp_sel = st.multiselect("Employment Status",
                                     VALID_CATEGORIES["Employment_Status"],
                                     default=VALID_CATEGORIES["Employment_Status"])
        with fc2:
            purpose_sel = st.multiselect("Loan Purpose", VALID_CATEGORIES["Loan_Purpose"],
                                         default=VALID_CATEGORIES["Loan_Purpose"])
            area_sel = st.multiselect("Property Area", VALID_CATEGORIES["Property_Area"],
                                      default=VALID_CATEGORIES["Property_Area"])
        with fc3:
            cs_sel = st.slider("Credit Score", 550, 800, (550, 800))
            income_sel = st.slider("Applicant Income",
                                   int(df["Applicant_Income"].min()),
                                   int(df["Applicant_Income"].max()),
                                   (int(df["Applicant_Income"].min()),
                                    int(df["Applicant_Income"].max())))
        if st.button("Reset All Filters"):
            st.rerun()

    # Apply
    outcome_vals_map = {"Approved": 1, "Rejected": 0}
    ov = [outcome_vals_map[o] for o in outcome_sel]
    exp_df = df[
        df[TARGET_COL].isin(ov) &
        df["Employment_Status"].isin(emp_sel) &
        df["Loan_Purpose"].isin(purpose_sel) &
        df["Property_Area"].isin(area_sel) &
        df["Credit_Score"].between(cs_sel[0], cs_sel[1]) &
        df["Applicant_Income"].between(income_sel[0], income_sel[1])
    ].copy()

    exp_df["Loan_Approved_Label"] = exp_df[TARGET_COL].map({1: "Approved", 0: "Rejected"})

    # Column selector
    available_cols = list(exp_df.columns)
    all_cols = [c for c in available_cols if c != TARGET_COL]
    selected_cols = st.multiselect("Select columns to display", all_cols,
                                   default=[c for c in all_cols if c != "Applicant_ID"])

    st.info(f"Showing **{len(exp_df):,}** records | {len(selected_cols)} columns selected")

    if selected_cols:
        # Pagination
        page_size = 25
        total_pages = max(1, (len(exp_df) + page_size - 1) // page_size)
        page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        start = (page_num - 1) * page_size
        end = start + page_size
        page_df = exp_df[selected_cols].iloc[start:end]
        st.dataframe(page_df, use_container_width=True)
        st.caption(f"Page {page_num} of {total_pages} | {len(exp_df)} total records")

        # Downloads
        dl1, dl2 = st.columns(2)
        with dl1:
            csv_data = exp_df[selected_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Filtered Records (CSV)",
                data=csv_data,
                file_name="loanvision_filtered_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl2:
            saved_metrics = load_metrics()
            if saved_metrics:
                metrics_csv = pd.DataFrame(saved_metrics["all_metrics"])[
                    ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
                ].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Model Comparison (CSV)",
                    data=metrics_csv,
                    file_name="loanvision_model_metrics.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info("Train models first to enable metrics download.")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE G — Project Information
# ════════════════════════════════════════════════════════════════════════════════

elif page == "info":
    st.title("ℹ️ Project Information")

    saved_metrics = load_metrics()

    st.markdown("""
    ## Project Objective

    **LoanVision AI** is an educational analytics platform that demonstrates:
    - Data preprocessing and exploratory analysis on a loan application dataset
    - Machine learning model training, comparison, and evaluation
    - Interactive visualizations powered by Plotly
    - Real-time loan approval prediction with model interpretation

    This is a prototype for learning and demonstration purposes only.
    It must not be used for actual lending decisions.
    """)

    st.markdown("---")
    st.markdown("## Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        | Property | Value |
        |----------|-------|
        | Total raw rows | 1,000 |
        | Rows dropped (missing target) | {clean_log.get('dropped_missing_target', 50)} |
        | Rows dropped (duplicates) | {clean_log.get('dropped_duplicates', 0)} |
        | **Valid records for modelling** | **{clean_log.get('final_row_count', '')}** |
        | Features used | {len(ALL_FEATURES)} |
        | Numerical features | {len(NUMERICAL_FEATURES)} |
        | Categorical features | {len(CATEGORICAL_FEATURES)} |
        | Target variable | Loan_Approved (Yes/No → 1/0) |
        | Approved | {int(df[TARGET_COL].sum())} ({int(df[TARGET_COL].sum())/len(df)*100:.1f}%) |
        | Rejected | {len(df) - int(df[TARGET_COL].sum())} ({(1-df[TARGET_COL].mean())*100:.1f}%) |
        """)
    with col2:
        st.markdown("""
        **Data Cleaning Decisions:**
        - Rows where `Loan_Approved` is missing → **dropped** (target unknown)
        - `Applicant_ID` excluded from features (identifier)
        - Numerical missing values → **median imputation** (robust to skew)
        - Categorical missing values → **mode imputation**
        - Empty string cells → treated as NaN
        - No valid observations were silently deleted beyond the above rules
        """)

    st.markdown("---")
    st.markdown("## Feature Descriptions")
    st.markdown("""
    | Feature | Type | Description |
    |---------|------|-------------|
    | Applicant_Income | Numerical | Primary applicant's income |
    | Coapplicant_Income | Numerical | Co-applicant's income (0 if none) |
    | Age | Numerical | Age of the primary applicant |
    | Credit_Score | Numerical | Credit bureau score (550–800 in dataset) |
    | Existing_Loans | Numerical | Number of existing active loans |
    | DTI_Ratio | Numerical | Debt-to-Income ratio (0.1–0.6) |
    | Savings | Numerical | Savings / liquid assets |
    | Collateral_Value | Numerical | Estimated value of pledged collateral |
    | Loan_Amount | Numerical | Requested loan amount |
    | Loan_Term | Numerical | Loan duration in months |
    | Dependents | Numerical | Number of financial dependents |
    | Employment_Status | Categorical | Contract / Salaried / Self-employed / Unemployed |
    | Marital_Status | Categorical | Married / Single |
    | Loan_Purpose | Categorical | Business / Car / Education / Home / Personal |
    | Property_Area | Categorical | Rural / Semiurban / Urban |
    | Education_Level | Categorical | Graduate / Not Graduate |
    | Gender | Categorical | Female / Male |
    | Employer_Category | Categorical | Business / Government / MNC / Private / Unemployed |
    """)

    st.markdown("---")
    st.markdown("## Models Trained")
    st.markdown("""
    | Model | Notes |
    |-------|-------|
    | Logistic Regression | Linear baseline; `class_weight='balanced'` |
    | Random Forest | 200 trees; `class_weight='balanced'` |
    | Gradient Boosting | 200 trees, lr=0.05, max_depth=4, subsample=0.8 |
    """)

    if saved_metrics:
        st.markdown("---")
        st.markdown("## Evaluation Results (Held-out Test Set, 20%)")
        metrics_df = pd.DataFrame(saved_metrics["all_metrics"])[
            ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
        ]
        st.dataframe(metrics_df, use_container_width=True)
        st.markdown(f"""
        **Best model:** {saved_metrics['best_model']}
        **Selection criterion:** ROC-AUC
        **Rationale:** Class imbalance (~31% approvals) makes accuracy misleading.
        ROC-AUC evaluates discriminative ability across all decision thresholds.
        """)

    st.markdown("---")
    st.markdown("""
    ## Ethical Considerations

    - This model may reflect biases present in the training data's labelling process.
    - Features like `Gender`, `Marital_Status`, and `Property_Area` may serve as proxies
      for protected characteristics. Their inclusion or exclusion should be evaluated carefully.
    - `class_weight='balanced'` mitigates class imbalance but does not guarantee demographic fairness.
    - No formal fairness audit has been performed on this educational dataset.
    - **This tool must not be used to make or influence actual lending decisions.**

    ## Limitations

    - The dataset is synthetic/educational; production performance is unknown.
    - No hyperparameter optimisation was performed.
    - The held-out test set size (~190 rows) is small; confidence intervals on metrics are wide.
    - Local explanations are approximations (perturbation-based), not exact attributions.
    - Calibration of probabilities has not been verified.

    ## How to Run

    ```bash
    cd loanvision-ai
    pip install -r requirements.txt
    streamlit run app.py
    ```

    Then navigate to http://localhost:8501 in your browser.
    """)

# ─── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-size:.75rem;color:#94a3b8;'>"
    "LoanVision AI · Educational Prototype · Not for production use"
    "</div>",
    unsafe_allow_html=True,
)
