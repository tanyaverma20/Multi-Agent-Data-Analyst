import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Allow imports from src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == "streamlit_app" else BASE_DIR
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for p in [PROJECT_ROOT, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from core.a2a_bus import A2ABus
from agents.model_agent import ModelAgent
from tools.model_tools import ModelTools

try:
    import google.generativeai as genai
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        genai.configure(api_key=gemini_key)
        HAS_GEMINI = True
    else:
        HAS_GEMINI = False
except Exception:
    HAS_GEMINI = False

# ------------------------------------------------------
# INIT GLOBAL A2A BUS + MODEL AGENT
# ------------------------------------------------------
a2a = A2ABus()
model_agent = ModelAgent(a2a_bus=a2a)

st.set_page_config(page_title="AutoML Dashboard", page_icon="🤖", layout="wide")
st.title("🤖 AutoML & Feature Engineering Studio")

# ------------------------------------------------------
# ENSURE DATA EXISTS
# ------------------------------------------------------
if "uploaded_df" not in st.session_state:
    sample_path = os.path.join(PROJECT_ROOT, "sample.csv")
    if os.path.exists(sample_path):
        st.info("💡 No uploaded dataset found. Loaded default `sample.csv` for demonstration.")
        st.session_state["uploaded_df"] = pd.read_csv(sample_path)
    else:
        st.warning("⚠️ Please upload a dataset first from the Home page.")
        st.stop()

df = st.session_state["uploaded_df"]

# ------------------------------------------------------
# TARGET & CONFIGURATION PANEL
# ------------------------------------------------------
st.subheader("🎯 AutoML Configuration")
columns = df.columns.tolist()

col_conf1, col_conf2, col_conf3, col_conf4 = st.columns([2, 1, 1, 1])

with col_conf1:
    default_idx = len(columns) - 1
    target = st.selectbox("Choose the target column to predict:", columns, index=default_idx)

with col_conf2:
    cv_folds = st.slider("CV Folds", min_value=2, max_value=10, value=5, step=1)

with col_conf3:
    outlier_strat = st.selectbox("Outliers", ["clip", "none"], index=0, help="Clip winsorizes values based on training IQR bounds.")

with col_conf4:
    tune_hyperparams = st.checkbox("Enable Tuning", value=True, help="Tune models with GridSearchCV / RandomizedSearchCV")
    tuning_method = st.selectbox(
        "Tuning Method",
        ["grid", "random"],
        format_func=lambda x: "GridSearchCV" if x == "grid" else "RandomizedSearchCV",
        disabled=not tune_hyperparams
    )

with st.expander("📄 Dataset Preview & Info", expanded=False):
    st.write(f"**Shape**: {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(10), use_container_width=True)

st.markdown("---")

def gemini_explain_model(result):
    if not HAS_GEMINI:
        return "ℹ️ Gemini API key not configured. Set GEMINI_API_KEY in your .env file to enable AI explanations."
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
        You are an elite ML engineer and data scientist. Explain this AutoML result clearly and comprehensively.

        MODEL OUTPUT JSON:
        {json.dumps(result, indent=2, default=str)}

        Please explain:
        1. Task type (classification or regression) and target variable
        2. Feature Engineering summary: derived features created, transformations applied, and outlier handling
        3. Hyperparameter tuning & Cross-Validation: which parameters were chosen and why
        4. Key driving features and their interpretability/impact
        5. Practical recommendations for production deployment
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"


# ======================================================
# TRAINING BUTTONS
# ======================================================
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("🚀 Train Model (Direct ModelTools)", key="manual_train_btn", use_container_width=True):
        with st.spinner("Training models with feature engineering, hyperparameter tuning, and CV..."):
            model_tool = ModelTools()
            result = model_tool.train(
                df,
                target_col=target,
                cv_folds=cv_folds,
                tune_hyperparameters=tune_hyperparams,
                tuning_method=tuning_method,
                outlier_strategy=outlier_strat,
            )

        if result.get("status") == "error":
            st.error(f"❌ Error: {result.get('error')}")
            st.stop()

        st.session_state["model_output"] = result
        st.success("🎉 Direct AutoML Pipeline execution completed successfully!")

with col_btn2:
    if st.button("🤖 Train Model via Agent (A2A Bus)", key="agent_train_btn", use_container_width=True):
        with st.spinner("ModelAgent executing pipeline with hyperparameter tuning and A2A events..."):
            result = model_agent.run(
                df,
                target_col=target,
                cv_folds=cv_folds,
                tune_hyperparameters=tune_hyperparams,
                tuning_method=tuning_method,
                outlier_strategy=outlier_strat,
            )

        if result.get("status") == "error":
            st.error(result.get("error"))
        else:
            st.session_state["model_output"] = result
            st.success("🤖 ModelAgent completed tuning and training, notified VerifierAgent via A2A!")

# ======================================================
# 🔄 A2A Auto Trigger (EDA → Model)
# ======================================================
auto_res = model_agent.poll_messages_and_run(df, target_col=target)
if auto_res:
    st.success("🚀 Auto-triggered model training from EDA event completed!")
    st.session_state["model_output"] = auto_res


# ======================================================
# RENDER RICH DASHBOARD SECTIONS
# ======================================================
if "model_output" in st.session_state and st.session_state["model_output"].get("status") == "success":
    result = st.session_state["model_output"]

    st.markdown("---")
    st.header("🏆 Best Model & Test Set Performance")

    best_name = result.get("best_model", result.get("model_name", "N/A"))
    task = result.get("task_type", "classification")
    test_metrics = result.get("test_metrics", result.get("metrics", {}))

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Selected Best Model", best_name.upper())
    col_m2.metric("Task Type", task.capitalize())

    if task == "classification":
        acc = test_metrics.get("accuracy")
        f1 = test_metrics.get("f1_score", test_metrics.get("f1"))
        col_m3.metric("Test Accuracy", f"{acc:.3f}" if acc is not None else "N/A")
        col_m4.metric("Test F1-Score", f"{f1:.3f}" if f1 is not None else "N/A")
    else:
        r2 = test_metrics.get("r2")
        rmse = test_metrics.get("rmse")
        col_m3.metric("Test R² Score", f"{r2:.3f}" if r2 is not None else "N/A")
        col_m4.metric("Test RMSE", f"{rmse:.3f}" if rmse is not None else "N/A")

    st.caption(f"💾 Model artifact saved to: `{result.get('model_path')}`")

    # --------------------------------------------------
    # 1. FEATURE ENGINEERING SECTION
    # --------------------------------------------------
    st.markdown("---")
    st.header("1️⃣ Feature Engineering & Preprocessing")

    fe = result.get("feature_engineering", {})
    created = fe.get("created_features", [])
    transformed = fe.get("transformed_features", [])
    num_cols = fe.get("numeric_features", [])
    cat_cols = fe.get("categorical_features", [])
    applied = fe.get("transformations_applied", [])

    fe_col1, fe_col2, fe_col3, fe_col4 = st.columns(4)
    fe_col1.metric("Numeric Features", len(num_cols))
    fe_col2.metric("Categorical Features", len(cat_cols))
    fe_col3.metric("Created Derived Features", len(created))
    fe_col4.metric("Transformed Features", len(transformed))

    st.write("##### 🛠 Applied Transformations Pipeline")
    if applied:
        st.write(" • " + "\n • ".join(applied))

    if created:
        with st.expander("✨ View Newly Created Features", expanded=True):
            st.write(", ".join([f"`{c}`" for c in created]))

    if transformed:
        with st.expander("📈 View Log/Power Transformed Skewed Features"):
            st.write(", ".join([f"`{t}`" for t in transformed]))

    # --------------------------------------------------
    # 2. OUTLIER DETECTION & HANDLING
    # --------------------------------------------------
    st.markdown("---")
    st.header("2️⃣ Outlier Detection & Handling")

    outliers = result.get("outliers", {})
    method = outliers.get("method", "IQR").upper()
    strat = outliers.get("handling_strategy", "clip").capitalize()
    total_outliers = outliers.get("total_outliers", 0)
    pct_affected = outliers.get("percentage_affected", 0.0)
    affected_cols = outliers.get("affected_columns", {})

    out_col1, out_col2, out_col3, out_col4 = st.columns(4)
    out_col1.metric("Detection Method", method)
    out_col2.metric("Handling Strategy", strat)
    out_col3.metric("Total Outlier Values", total_outliers)
    out_col4.metric("Rows Affected", f"{pct_affected}%")

    if affected_cols:
        st.write("##### 📋 Affected Columns Breakdown")
        outlier_rows = []
        for col_name, info in affected_cols.items():
            outlier_rows.append({
                "Column": col_name,
                "Outlier Count": info.get("outlier_count"),
                "Percentage (%)": f"{info.get('percentage')}%",
                "Lower Bound": info.get("lower_bound"),
                "Upper Bound": info.get("upper_bound"),
            })
        st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True)
    else:
        st.success("✅ No extreme outliers detected based on IQR boundaries.")

    # --------------------------------------------------
    # 3. AUTOMATED FEATURE SELECTION
    # --------------------------------------------------
    st.markdown("---")
    st.header("3️⃣ Automated Feature Selection")

    fs = result.get("feature_selection", {})
    orig_cnt = fs.get("original_count", 0)
    sel_cnt = fs.get("selected_count", 0)
    sel_method = fs.get("method", "k_best").upper()
    scores = fs.get("feature_scores", {})

    fs_col1, fs_col2, fs_col3 = st.columns(3)
    fs_col1.metric("Original Feature Count", orig_cnt)
    fs_col2.metric("Selected Feature Count", sel_cnt)
    fs_col3.metric("Selection Method", sel_method)

    if scores:
        score_df = pd.DataFrame(list(scores.items()), columns=["Feature", "Score"]).sort_values(by="Score", ascending=False)
        with st.expander("📊 Feature Importance Scores from Selector", expanded=True):
            if HAS_PLOTLY and not score_df.empty:
                fig_scores = px.bar(
                    score_df.head(15),
                    x="Score",
                    y="Feature",
                    orientation="h",
                    title="Top Selected Features by Relevance Score",
                    color="Score",
                    color_continuous_scale="Viridis",
                )
                fig_scores.update_layout(yaxis=dict(autorange="reversed"), height=350)
                st.plotly_chart(fig_scores, use_container_width=True)
            else:
                st.dataframe(score_df, use_container_width=True)

    # --------------------------------------------------
    # 4. CROSS-VALIDATION
    # --------------------------------------------------
    st.markdown("---")
    st.header("4️⃣ Cross-Validation Performance")

    cv = result.get("cross_validation", {})
    folds = cv.get("folds", 5)
    metric_name = cv.get("cv_metric", "Score").upper()
    cv_models = cv.get("models", {})

    st.write(f"**Evaluation Strategy**: {folds}-Fold Cross-Validation (Metric: `{metric_name}`)")

    if cv_models:
        cv_summary_rows = []
        for m_name, m_info in cv_models.items():
            row = {
                "Model": m_name.upper(),
                f"Mean CV {metric_name}": m_info.get("mean_cv_score"),
                "Std Dev (±)": m_info.get("std_cv_score"),
            }
            for idx, f_score in enumerate(m_info.get("fold_scores", []), start=1):
                row[f"Fold {idx}"] = f_score
            cv_summary_rows.append(row)

        cv_df = pd.DataFrame(cv_summary_rows).sort_values(by=f"Mean CV {metric_name}", ascending=False)
        st.dataframe(cv_df, use_container_width=True)

        if HAS_PLOTLY and not cv_df.empty:
            fig_cv = px.bar(
                cv_df,
                x="Model",
                y=f"Mean CV {metric_name}",
                error_y="Std Dev (±)",
                title=f"Candidate Model Cross-Validation Comparison ({folds} Folds)",
                color="Model",
                text_auto=True,
            )
            fig_cv.update_layout(height=350)
            st.plotly_chart(fig_cv, use_container_width=True)

    # --------------------------------------------------
    # 5. HYPERPARAMETER TUNING SECTION
    # --------------------------------------------------
    st.markdown("---")
    st.header("5️⃣ Hyperparameter Tuning")

    ht = result.get("hyperparameter_tuning", {})
    if ht and ht.get("enabled"):
        t_method = ht.get("method", "GridSearchCV")
        tuned_models = ht.get("models", {})
        best_overall = ht.get("best_tuned_model", "N/A")
        best_params_overall = ht.get("best_overall_params", {})

        ht_col1, ht_col2, ht_col3 = st.columns(3)
        ht_col1.metric("Search Strategy", t_method)
        ht_col2.metric("Models Tuned", len(tuned_models))
        ht_col3.metric("Winning Tuned Model", best_overall.upper())

        st.write("##### 🔧 Tuned Model Parameters & Performance")
        tuning_rows = []
        for m_name, m_data in tuned_models.items():
            tuning_rows.append({
                "Model": m_name.upper(),
                "Search Method": m_data.get("parameter_search_method", t_method),
                "Best CV Score": m_data.get("best_cv_score"),
                "CV Std Dev (±)": m_data.get("cv_std"),
                "Candidates Evaluated": m_data.get("candidates_evaluated"),
                "Duration (s)": m_data.get("search_duration_sec"),
                "Best Parameters": str(m_data.get("best_params")),
            })

        st.dataframe(pd.DataFrame(tuning_rows), use_container_width=True)

        with st.expander(f"✨ Optimal Hyperparameters for {best_overall.upper()} (Winning Model)", expanded=True):
            st.json(best_params_overall)
    else:
        st.info("Hyperparameter tuning was disabled for this run.")

    # --------------------------------------------------
    # 6. FEATURE IMPORTANCE ANALYSIS
    # --------------------------------------------------
    st.markdown("---")
    st.header("6️⃣ Feature Importance Analysis")

    fi = result.get("feature_importance", [])
    if fi:
        fi_df = pd.DataFrame(fi)
        fi_col1, fi_col2 = st.columns([1.2, 1])

        with fi_col1:
            st.write("##### 🌟 Ranked Feature Importance Table")
            st.dataframe(fi_df, use_container_width=True)

        with fi_col2:
            if HAS_PLOTLY:
                fig_fi = px.bar(
                    fi_df.head(10),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top 10 Feature Importances",
                    color="importance",
                    color_continuous_scale="Blues",
                )
                fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=350)
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.bar_chart(fi_df.set_index("feature")["importance"].head(10))
    else:
        st.info("Feature importance not available for this model type.")

    # --------------------------------------------------
    # 7. SAMPLE PREDICTIONS
    # --------------------------------------------------
    if "sample_predictions" in result:
        with st.expander("🔮 Sample Predictions on Held-out Test Set"):
            preds = result.get("sample_predictions", [])
            st.json(preds)

    # --------------------------------------------------
    # 8. GEMINI INSIGHTS
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("✨ Gemini AI Explanation & Insights")

    if st.button("💡 Explain Pipeline & Model with Gemini", key="gemini_btn"):
        with st.spinner("Generating AI analysis with Gemini..."):
            explanation = gemini_explain_model(result)
        st.markdown(explanation)
