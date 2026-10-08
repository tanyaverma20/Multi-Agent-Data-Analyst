# Multi-Agent AutoML Data Analyst

> **Autonomous End-to-End Tabular Machine Learning, Feature Engineering, and Statistical Intelligence Platform Powered by Multi-Agent Architecture, Model Context Protocol (MCP), and Google Gemini.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6%2B-F7931E.svg)](https://scikit-learn.org/)
[![Gemini 2.0 Flash](https://img.shields.io/badge/Google%20Gemini-2.0%20Flash-4285F4.svg)](https://deepmind.google/technologies/gemini/)
[![Tests](https://img.shields.io/badge/Tests-32%20Passed-success.svg)](https://pytest.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Multi--Agent%20%2B%20MCP-8A2BE2.svg)](#architecture)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

🚀 **Live Demo (Render Deployment):** [https://multiagent-data-analyst.onrender.com/](https://multiagent-data-analyst.onrender.com/)  
📦 **Repository:** [https://github.com/tanyaverma20/multi-agent-data-analyst](https://github.com/tanyaverma20/multi-agent-data-analyst)

---

<p align="center">
  <img width="2452" height="1286" alt="Multi-Agent AutoML Architecture" src="https://github.com/user-attachments/assets/92641b2f-fceb-493a-b481-345e5e341de4" />
</p>

---

## Overview

The **Multi-Agent AutoML Data Analyst** is an enterprise-pattern data science automation framework. It coordinates a team of six specialized autonomous agents communicating over an in-memory, thread-safe Agent-to-Agent (`A2ABus`) message bus and executing atomic data tools registered through the **Model Context Protocol (MCP)**.

Given any raw tabular dataset (CSV or JSON payload), the system executes the entire analytical lifecycle with **zero human intervention**:

- **Automated Data Profiling**: Schema detection, missingness inspection, and datatype inference.
- **Exploratory Data Analysis (EDA)**: 3-metric correlation matrices (Pearson, Spearman, Kendall), distribution moments, VIF multicollinearity, and visual heatmap generation.
- **Leakage-Safe Feature Engineering**: Non-destructive IQR outlier clipping, non-linear interaction terms, ratios with zero-division protection, log1p skew transforms, and median/most-frequent imputation.
- **Adaptive Feature Selection**: Task-aware univariate ANOVA F-tests and model-based importance ranking.
- **Task Detection & Multi-Model AutoML**: Automatic classification/regression detection with candidate training across Random Forest, Logistic Regression, Gradient Boosting, Ridge, and Support Vector Regressors.
- **Cross-Validation & Hyperparameter Tuning**: Systematic parameter optimization via `GridSearchCV` or `RandomizedSearchCV` within cross-validation folds on training data only.
- **Algorithmic Quality Auditing**: Automated verification of model reliability, CV variance stability, and outlier impact.
- **Jupyter Notebook Synthesis**: Assembling a presentation-ready `.ipynb` report embedded with Google Gemini 2.0 Flash executive insights.
- **Dual-Surface Serving**: An interactive multi-page **Streamlit Studio** and a production-grade **FastAPI REST API** with automated end-to-end inference preprocessing.

---

## Why This Project?

Traditional tabular machine learning and exploratory data analysis suffer from tedious manual overhead and critical engineering pitfalls:

```text
Dataset Inspection ──► Cleaning ──► EDA ──► Preprocessing ──► Feature Engineering ──► Model Selection ──► Tuning ──► Evaluation ──► Reporting
```

1. **Subtle Data Leakage**: In typical workflows, imputers, scalers, outlier bounds, and feature selectors are mistakenly fitted on the full dataset before cross-validation, producing overly optimistic test results that fail in production.
2. **Brittle Preprocessing Disconnect**: Models trained in isolation require inference endpoints to manually reimplement transformation logic, leading to schema mismatches and prediction crashes on unseen data.
3. **Disjointed Hyperparameter Search**: Practitioners frequently tune parameters using ad-hoc scripts that leak validation splits or ignore cross-validation stability.

### The Multi-Agent Solution
This project decomposes the workflow into modular, collaborating agents. Each agent possesses bounded responsibilities, emits observable lifecycle events, and executes standardized tools via an MCP tool registry. The core ML engine guarantees **mathematical zero data leakage** by fitting all preprocessing, transformations, and parameter searches strictly within training folds.

---

## Key Capabilities

| Capability | Implementation | Code Location |
|---|---|---|
| **Data Profiling** | Automated schema detection, null distributions, and type inference | `src/agents/profiler_agent.py` |
| **Exploratory Analysis (EDA)** | Pearson, Spearman, Kendall correlations; skewness, kurtosis, VIF | `src/agents/eda_agent.py` |
| **Outlier Handling** | Non-destructive IQR boundary calculation and winsorization/clipping | `src/core/feature_engineering.py` |
| **Feature Creation** | Skew log1p transforms, pairwise ratios with zero-division safeguards, differences, interactions | `src/core/feature_engineering.py` |
| **Categorical Encoding** | `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` | `src/tools/model_tools.py` |
| **Numerical Scaling** | Median imputation and `StandardScaler` fitted strictly on training data | `src/tools/model_tools.py` |
| **Feature Selection** | Task-aware `SelectKBest` (`f_classif` / `f_regression`) with adaptive $k$ | `src/core/feature_engineering.py` |
| **Task Detection** | Automated Classification vs. Regression target inference | `src/tools/model_tools.py` |
| **Candidate AutoML** | Random Forest, Logistic Regression, Gradient Boosting, Ridge, SVR | `src/tools/model_tools.py` |
| **Hyperparameter Tuning** | Systematic `GridSearchCV` and `RandomizedSearchCV` over CV folds | `src/tools/model_tools.py` |
| **Cross-Validation** | StratifiedKFold (classification) and KFold (regression), default 5 folds | `src/tools/model_tools.py` |
| **Evaluation Metrics** | Accuracy, Precision, Recall, Weighted F1, ROC-AUC, MAE, MSE, RMSE, $R^2$ | `src/tools/model_tools.py` |
| **Feature Importance** | Tree importances, linear coefficient magnitude/direction, permutation importance | `src/tools/model_tools.py` |
| **Quality Verification** | Heuristic audit of model strength, CV stability ($\text{std} \le 0.08$), and outliers | `src/agents/verifier_agent.py` |
| **Notebook Synthesis** | Automated `.ipynb` report creation via `nbformat` | `src/agents/notebook_synthesizer_agent.py` |
| **LLM Insights** | Google Gemini 2.0 Flash automated data and model explanations | `src/tools/notebook_tools.py` |
| **Agent Messaging** | Thread-safe in-memory publish/subscribe event bus (`A2ABus`) | `src/core/a2a_bus.py` |
| **Tool Protocol (MCP)** | Standardized Model Context Protocol tool registry (`mcp/manifest.json`) | `src/tools/agent_tools.py` |
| **REST API** | Production FastAPI backend (`/health`, `/predict`, `/train`, `/model-info`) | `src/api/main.py` |
| **Web UI** | 7-page interactive Streamlit analytical studio | `streamlit_app/app.py` |

---

## Architecture

The system coordinates agents through the `A2ABus` and equips them with atomic operations via the `ToolRegistry`:

```text
                               ┌─────────────────────────────────┐
                               │     Model Context Protocol      │
                               │        (mcp/manifest.json)      │
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │           ToolRegistry          │
                               │  (dataset, feature, model, etc) │
                               └────────────────┬────────────────┘
                                                │
                                                ▼
Dataset (CSV / JSON) ──► ProfilerAgent ──► EDAAgent ──► Feature Engineering Pipeline
                                                               │
                                                               ▼
FastAPI REST API     ◄── FullAutoMLPipeline ◄── ModelAgent (AutoML & Hyperparameter Tuning)
        │                      │                               │
        │                      ▼                               ▼
        │             Winning Model Artifact             VerifierAgent
        │              (models/*.pkl)                          │
        ▼                                                      ▼
Streamlit Studio     ◄─────────────────────────────── NotebookSynthesizerAgent ──► Gemini 2.0
                                                               │
                                                               ▼
                                                     reports/auto_report.ipynb
```

```text
                  ┌───────────────────────────────────────────────┐
                  │            A2A Communication Bus              │
                  │   Thread-safe, Topic Queuing, Audit Log       │
                  └───────────────────────┬───────────────────────┘
                                          │
       profiler.completed ──► eda.completed ──► feature_engineering.started
                                                       │
       model.trained ◄── hyperparameter_tuning.completed ◄── cross_validation.completed
             │
             └──► verifier.completed ──► notebook.synthesized
```

<p align="center">
  <img width="2452" height="1286" alt="System Architecture Diagram" src="https://github.com/user-attachments/assets/92641b2f-fceb-493a-b481-345e5e341de4" />
</p>

---

## Agent Responsibilities

| Agent | Source File | Responsibilities |
|---|---|---|
| **ProfilerAgent** | `src/agents/profiler_agent.py` | Ingests raw data, calculates dimensions, detects data types, audits null values, and publishes `profiler.completed`. |
| **EDAAgent** | `src/agents/eda_agent.py` | Computes Pearson, Spearman, and Kendall correlation matrices; calculates skewness, kurtosis, and VIF; generates distribution plots and heatmaps; publishes `eda.completed`. |
| **ModelAgent** | `src/agents/model_agent.py` | Coordinates AutoML, runs feature engineering, executes `GridSearchCV` / `RandomizedSearchCV`, logs fold scores, evaluates held-out test data, and publishes lifecycle events. |
| **VerifierAgent** | `src/agents/verifier_agent.py` | Audits model performance ($R^2$, F1), evaluates cross-validation stability ($\text{std} \le 0.08$), audits outlier clipping, and emits quality verdicts (`Good`, `Acceptable`, `Weak`). |
| **NotebookSynthesizerAgent** | `src/agents/notebook_synthesizer_agent.py` | Ingests analytical outputs from all previous agents, compiles Markdown and code cells into a valid `.ipynb` notebook, injects Gemini AI insights, and saves `reports/auto_report.ipynb`. |
| **Gemini Integration** | `src/tools/notebook_tools.py` | Uses Google Gemini 2.0 Flash to synthesize natural-language summaries of EDA findings, model strengths/weaknesses, and deployment recommendations. |

---

## Feature Engineering Pipeline

The feature engineering layer (`src/core/feature_engineering.py`) is designed as a modular, scikit-learn-compatible pipeline that fits exclusively on training data:

```text
Raw Tabular Data
       │
       ▼
1. OutlierHandler (IQR detection & safe clipping)
       │
       ▼
2. FeatureCreationTransformer (Log1p transforms, ratios, differences, interactions)
       │
       ▼
3. ColumnTransformer (Median/mode imputation, StandardScaler, OneHotEncoder)
       │
       ▼
4. AutomatedFeatureSelector (ANOVA F-tests / Regression F-tests with adaptive k)
       │
       ▼
Transformed, Scaled, Selected Matrix ──► Model Training
```

### 1. Missing Value Imputation
- **Numerical Columns**: Imputed using `SimpleImputer(strategy="median")` to resist outlier skewing.
- **Categorical Columns**: Imputed using `SimpleImputer(strategy="most_frequent")`.

### 2. Outlier Detection & Non-Destructive Handling
- Calculates first quartile ($Q_1$) and third quartile ($Q_3$) per numerical column on $X_{\text{train}}$.
- Sets clipping boundaries: $[\text{lower}, \text{upper}] = [Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$.
- **Non-Destructive Clipping**: Rather than dropping observations (which reduces sample size and crashes on outlier test records), anomalies are winsorized to boundary limits using `np.clip()`. Columns with $\text{IQR} \le 10^{-9}$ fallback to $[\min, \max]$.

### 3. Derived Feature Creation & Mathematical Safeguards
- **Skewness Correction**: Non-negative columns with skewness $> 1.0$ receive a safe $\log(1 + x)$ transform via `np.log1p(np.maximum(0, x))`.
- **Informative Variance Pairs**: Features are sorted by variance; candidate interactions are evaluated on the top variable columns to avoid combinatorial explosion.
- **Pairwise Ratios**: $c_1 / c_2$ with zero-division safeguard (`.replace(0, np.nan)` and sanitizing $\pm\infty \to \text{NaN}$).
- **Pairwise Differences**: $c_1 - c_2$.
- **Interaction Products**: $c_1 \times c_2$.
- **Dimension Capping**: Enforces `max_created_features=15` to preserve efficiency.

### 4. Categorical Encoding & Scaling
- Categorical features are encoded using `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`, ensuring unseen categories during inference produce all-zero indicator vectors without throwing runtime exceptions.
- Numeric features are scaled using `StandardScaler()`.

### 5. Automated Task-Aware Feature Selection
- Automatically selects the top informative features:
  - **Classification**: Evaluates features using ANOVA F-value (`f_classif`).
  - **Regression**: Evaluates features using linear model F-value (`f_regression`).
  - **Model-Based**: Supports Random Forest importance-based selection.
- **Adaptive $k$**: Automatically retains $100\%$ of features if $N \le 5$, $85\%$ if $N \le 12$, and $70\%$ if $N > 12$.

---

## AutoML & Model Training

### 1. Task Detection
Implemented in `ModelTools._detect_task()`:
- Categorical target $\to$ **Classification**
- Numeric target with $> 20$ unique values $\to$ **Regression**
- Floating-point target with $> 10$ unique values $\to$ **Regression**
- Numeric target with $\le 10$ unique values $\to$ **Classification**

### 2. Candidate Models

#### Classification
- **Random Forest Classifier** (`rf`)
- **Logistic Regression** (`logreg`)
- **Gradient Boosting Classifier** (`gbm`)

#### Regression
- **Random Forest Regressor** (`rf`)
- **Ridge Regression** (`ridge`)
- **Support Vector Regressor** (`svr`)

### 3. Cross-Validation & Hyperparameter Tuning
- **Cross-Validation Splitters**: `StratifiedKFold` for classification (preserving class balance) and `KFold` for regression (default $k=5$).
- **Search Algorithms**:
  - `GridSearchCV`: Exhaustive evaluation across defined parameter combinations.
  - `RandomizedSearchCV`: Subsampled parameter evaluations (`n_iter=8`) when parameter spaces exceed 8 combinations.
- **Task-Specific Search Grids**:

```python
# Classification Parameter Grids
"rf":     {"n_estimators": [50, 100], "max_depth": [None, 5, 10], "min_samples_split": [2, 5], "min_samples_leaf": [1, 2]}
"logreg": {"C": [0.1, 1.0, 10.0], "solver": ["lbfgs"]}
"gbm":    {"n_estimators": [50, 80], "learning_rate": [0.05, 0.1], "max_depth": [3, 5], "min_samples_split": [2, 5]}

# Regression Parameter Grids
"rf":     {"n_estimators": [50, 100], "max_depth": [None, 5, 10], "min_samples_split": [2, 5], "min_samples_leaf": [1, 2]}
"ridge":  {"alpha": [0.1, 1.0, 10.0, 50.0]}
"svr":    {"C": [0.5, 1.0, 5.0], "epsilon": [0.1, 0.2], "kernel": ["rbf", "linear"]}
```

### 4. Best Model Selection & Data Leakage Safeguards
- The winning candidate is selected based strictly on mean cross-validation score on $X_{\text{train}}$ (`f1_weighted` for classification, $R^2$ for regression).
- The winning model is bundled into a `FullAutoMLPipeline` and evaluated **once** on the untouched, held-out test split ($X_{\text{test}}$).

---

## Evaluation Metrics

```text
Classification Metrics               Regression Metrics
├── Accuracy                         ├── Mean Absolute Error (MAE)
├── Precision (Weighted)             ├── Mean Squared Error (MSE)
├── Recall (Weighted)                ├── Root Mean Squared Error (RMSE)
├── F1 Score (Weighted)              └── Coefficient of Determination (R²)
└── ROC-AUC (Binary & Multiclass OVR)
```

---

## Feature Importance Analysis

The feature importance engine (`ModelTools._compute_feature_importance`) maps importance scores back to human-readable post-transformation feature names:
- **Tree-Based Models**: Reads `feature_importances_`.
- **Linear Models**: Reads absolute coefficient values `abs(coef_)` and extracts directionality (`positive`, `negative`, or `multiclass`).
- **Kernel & Non-Linear Models (SVR)**: Executes `permutation_importance(estimator, X_val, y_val, n_repeats=5)`.

---

## Multi-Agent Communication (A2A Bus)

The `A2ABus` (`src/core/a2a_bus.py`) provides thread-safe, decoupled event messaging with chronological audit logging:

```text
Agent Lifecycle Events Emitted:
├── profiler.completed                (Published by ProfilerAgent -> Consumed by EDAAgent)
├── eda.completed                     (Published by EDAAgent -> Consumed by ModelAgent)
├── feature_engineering.started       (Published by ModelAgent -> Broadcast)
├── feature_engineering.completed     (Published by ModelAgent -> Broadcast)
├── hyperparameter_tuning.started     (Published by ModelAgent -> Broadcast)
├── hyperparameter_tuning.completed   (Published by ModelAgent -> Broadcast)
├── cross_validation.completed        (Published by ModelAgent -> Broadcast)
├── feature_selection.completed       (Published by ModelAgent -> Broadcast)
├── feature_importance.completed      (Published by ModelAgent -> Broadcast)
├── model.trained                     (Published by ModelAgent -> Consumed by VerifierAgent)
└── verifier.completed                (Published by VerifierAgent -> Consumed by NotebookSynthesizerAgent)
```

<p align="center">
  <img width="2940" height="1774" alt="A2A Communication Console" src="https://github.com/user-attachments/assets/9a0250a0-2415-432f-a3a0-b2c05aea8a7e" />
</p>

---

## Model Context Protocol (MCP) Tools

Tools are declared in `mcp/manifest.json` and registered in `ToolRegistry` (`src/tools/agent_tools.py`):

| MCP Tool Name | Description | Python Implementation |
|---|---|---|
| `dataset.describe` | Computes column stats, dimensions, and missing counts | `DatasetTools.describe()` |
| `dataset.missing` | Tallies null values per column | `DatasetTools.missing_values()` |
| `feature.engineer` | Generates derived ratios, interactions, and log transforms | `FeatureTools.engineer_features()` |
| `feature.outliers` | Calculates numerical outlier bounds and affected percentages | `FeatureTools.detect_outliers()` |
| `feature.select` | Executes task-aware automated feature selection | `FeatureTools.select_features()` |
| `model.tune` | Runs cross-validated hyperparameter search across candidate models | `FeatureTools.tune_model()` |
| `model.cross_validate` | Runs StratifiedKFold or KFold cross-validation | `ModelTools.train()` |
| `model.predict` | Executes model inference via `FullAutoMLPipeline` | `FullAutoMLPipeline.predict()` |
| `model.feature_importance`| Calculates ranked feature importances with direction | `ModelTools._compute_feature_importance()` |
| `notebook.create` | Generates structured Jupyter Notebook report | `NotebookTools.create_notebook()` |
| `file.read` / `file.write` | Sandboxed local file input/output in `project_storage/` | `FileTools.read()`, `FileTools.write()` |
| `memory.read` / `write` | Persistent key-value memory storage | `MemoryTools.load()`, `MemoryTools.save()` |

---

## FastAPI REST API

The application provides a production-grade REST API backend implemented with FastAPI (`src/api/main.py`) and validated with Pydantic schemas (`src/api/schemas.py`).

### Endpoints

| HTTP Method | Route | Description | Request Body | Response Model |
|---|---|---|---|---|
| `GET` | `/health` | Service health & liveness check | None | `HealthResponse` |
| `GET` | `/model-info` | Metadata, parameters, & metrics of active model | None | `ModelInfoResponse` |
| `POST` | `/predict` | Automated inference on raw feature records | `PredictionRequest` | `PredictionResponse` |
| `POST` | `/train` | Triggers AutoML training, tuning, and CV | `TrainRequest` | `TrainResponse` |

### Automated Preprocessing During Inference
Inference requests sent to `/predict` do **not** require the caller to handle missing values, scale features, encode categories, or compute interaction terms. The deserialized `FullAutoMLPipeline` executes all learned transformations automatically:

```bash
# Example Prediction Request (Single Raw Record)
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "data": {
         "age": 35,
         "income": 65000,
         "employment_type": "employed",
         "credit_score": 710,
         "loan_amount": 20000
       }
     }'
```

```json
// Example Response (HTTP 200 OK)
{
  "status": "success",
  "model_name": "rf",
  "task_type": "classification",
  "predictions": [0]
}
```

Start the API server:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.

---

## Streamlit Application

The multi-page Streamlit web studio (`streamlit_app/`) provides an interactive interface for data exploration, model training, and agent monitoring.

### 1. Dataset Upload & MCP Panel (`app.py`)
<p align="center">
  <img width="2928" height="1746" alt="Dataset Upload and MCP Panel" src="https://github.com/user-attachments/assets/6e6dddb5-a33b-47be-a2f0-44a6f26a06ec" />
</p>

### 2. Profiler Agent (`pages/Profiler.py`)
<p align="center">
  <img width="2940" height="1528" alt="Profiler Output View" src="https://github.com/user-attachments/assets/228e3f40-883a-41a4-abb9-58597050d94d" />
  <img width="2898" height="1560" alt="Profiler Metric Summary" src="https://github.com/user-attachments/assets/3dc7df6d-6486-4019-87c3-c837a85398db" />
</p>

### 3. EDA Dashboard (`pages/EDA_Dashboard.py`)
<p align="center">
  <img width="2938" height="1760" alt="EDA Correlation Matrix Heatmap" src="https://github.com/user-attachments/assets/ffc64ed4-f32c-49af-ae7d-b5b1de07770c" />
  <img width="2284" height="1518" alt="EDA Statistical Summaries" src="https://github.com/user-attachments/assets/ecaa796b-8470-4cdc-89f1-a9ee5437e221" />
  <img width="2260" height="936" alt="EDA Numerical Distributions" src="https://github.com/user-attachments/assets/68182f66-62b3-4dbd-b6aa-e57ba8bb532c" />
</p>

### 4. AutoML & Hyperparameter Tuning Studio (`pages/AutoML.py`)
<p align="center">
  <img width="2894" height="1566" alt="AutoML Results and CV Performance" src="https://github.com/user-attachments/assets/214fd60b-055d-4143-bee5-7283aca09528" />
  <img width="2940" height="1584" alt="AutoML Feature Importance Analysis" src="https://github.com/user-attachments/assets/a27fe305-5c8c-4799-b049-fce96a0a1326" />
</p>

### 5. Verifier Agent (`pages/Verifier.py`)
<p align="center">
  <img width="2310" height="1342" alt="Verifier Agent Audit Report" src="https://github.com/user-attachments/assets/21e769b2-bd25-45ef-a54c-c274eaa28262" />
</p>

### 6. Gemini AI Explanations
<p align="center">
  <img width="2326" height="1528" alt="Gemini AI Automated Explanations" src="https://github.com/user-attachments/assets/2f79d85b-d176-4b21-8a45-0c70d5e0e845" />
</p>

---

## Generated Notebook Report

The `NotebookSynthesizerAgent` compiles a reproducible Jupyter notebook saved to `reports/auto_report.ipynb`:
- Markdown data profile and shape analysis.
- Structured feature engineering and outlier summaries.
- Cross-validation comparison table with fold-level scores.
- Ranked feature importance table with directionality.
- Natural-language insights generated by Google Gemini 2.0 Flash.

<p align="center">
  <img width="2354" height="1500" alt="Generated Jupyter Notebook Report" src="https://github.com/user-attachments/assets/7d5ab7c6-97e1-436e-885d-798fb015f72f" />
</p>

---

## Data Leakage & Reliability Safeguards

Data leakage is strictly eliminated across all pipeline stages:

1. **Initial Train/Test Split**: An 80/20 train/test split is performed immediately. $X_{\text{test}}$ is held out and untouched.
2. **Training-Set-Only Transformations**:
   - `OutlierHandler`: Computes IQR bounds exclusively on $X_{\text{train}}$. Test outliers are clipped to training bounds without recomputing.
   - `FeatureCreationTransformer`: Skewness and variance rankings are calculated exclusively on $X_{\text{train}}$.
   - `ColumnTransformer`: Imputation medians, standard scaling means, and one-hot categories are learned exclusively on $X_{\text{train}}$.
   - `AutomatedFeatureSelector`: ANOVA F-statistics are calculated strictly on training folds.
3. **Hyperparameter Tuning Isolation**: Parameter searches (`GridSearchCV` / `RandomizedSearchCV`) use cross-validation splits generated exclusively from $X_{\text{train}}$. $X_{\text{test}}$ is never used for parameter selection.
4. **Final Single Evaluation**: The winning pipeline is evaluated once on $X_{\text{test}}$ for final test metrics.

---

## Testing & Quality Assurance

The codebase contains **32 automated tests**, all passing:

```text
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
collected 32 items

tests/test_agents_integration.py::test_tool_registry_includes_feature_tool PASSED [  3%]
tests/test_agents_integration.py::test_model_agent_a2a_lifecycle_events PASSED [  6%]
tests/test_agents_integration.py::test_verifier_agent_evaluates_automl_output PASSED [  9%]
tests/test_agents_integration.py::test_notebook_synthesizer_with_rich_automl PASSED [ 12%]
tests/test_api.py::test_get_health PASSED                                [ 15%]
tests/test_api.py::test_get_model_info PASSED                            [ 18%]
tests/test_api.py::test_post_predict_single_record PASSED                [ 21%]
tests/test_api.py::test_post_predict_batch_records PASSED                [ 25%]
tests/test_api.py::test_post_predict_invalid_data PASSED                 [ 28%]
tests/test_api.py::test_post_train_inline_data PASSED                    [ 31%]
tests/test_automl_pipeline.py::test_classification_end_to_end PASSED     [ 34%]
tests/test_automl_pipeline.py::test_regression_end_to_end PASSED         [ 37%]
tests/test_automl_pipeline.py::test_missing_values_handling PASSED       [ 40%]
tests/test_automl_pipeline.py::test_unseen_categorical_values PASSED     [ 43%]
tests/test_automl_pipeline.py::test_outlier_handling_in_pipeline PASSED  [ 46%]
tests/test_automl_pipeline.py::test_constant_column_dataset PASSED       [ 50%]
tests/test_automl_pipeline.py::test_small_dataset PASSED                 [ 53%]
tests/test_automl_pipeline.py::test_feature_importance_mapping PASSED    [ 56%]
tests/test_feature_engineering.py::test_outlier_detection_and_clipping PASSED [ 59%]
tests/test_feature_engineering.py::test_outlier_handler_no_data_leakage PASSED [ 62%]
tests/test_feature_engineering.py::test_feature_creation_transformer PASSED [ 65%]
tests/test_feature_engineering.py::test_feature_creation_safeguards_division_by_zero PASSED [ 68%]
tests/test_feature_engineering.py::test_feature_creation_constant_column PASSED [ 71%]
tests/test_feature_engineering.py::test_automated_feature_selector_classification PASSED [ 75%]
tests/test_feature_engineering.py::test_automated_feature_selector_regression PASSED [ 78%]
tests/test_feature_engineering.py::test_feature_tools_mcp_standalone PASSED [ 81%]
tests/test_hyperparameter_tuning.py::test_hyperparameter_tuning_classification PASSED [ 84%]
tests/test_hyperparameter_tuning.py::test_hyperparameter_tuning_regression PASSED [ 87%]
tests/test_hyperparameter_tuning.py::test_randomized_search_cv PASSED    [ 90%]
tests/test_hyperparameter_tuning.py::test_no_data_leakage_during_hyperparameter_tuning PASSED [ 93%]
tests/test_hyperparameter_tuning.py::test_model_agent_emits_tuning_a2a_events PASSED [ 96%]
tests/test_hyperparameter_tuning.py::test_feature_tools_tune_mcp_method PASSED [100%]

============================== 32 passed in 233s ==============================
```

- **API Tests** (`tests/test_api.py`): Endpoint testing via `TestClient` covering `/health`, `/model-info`, `/predict`, `/train`, and HTTP 400/404 error cases.
- **Hyperparameter Tuning Tests** (`tests/test_hyperparameter_tuning.py`): Parameter search execution, RandomizedSearchCV, zero test-set leakage, and tuning A2A events.
- **AutoML Pipeline Tests** (`tests/test_automl_pipeline.py`): End-to-end classification/regression, missing values, unseen categorical levels, outlier clipping, and feature importance name recovery.
- **Feature Engineering Tests** (`tests/test_feature_engineering.py`): Outlier clipping bounds, zero-division safeguards, constant feature handling, and statistical selection.
- **Agent Integration Tests** (`tests/test_agents_integration.py`): ToolRegistry resolution, A2A bus lifecycle events, Verifier audits, and notebook report generation.
- **CLI & Compilation Verification**: `python src/orchestrator.py` runs end-to-end with exit code 0; all Streamlit pages and FastAPI code compile with `py_compile` with zero errors.

---

## Visual Gallery & Generated Visualizations

In addition to the UI dashboards, the EDA agent generates publication-quality plots saved in `reports/`:

<p align="center">
  <img width="48%" alt="Pearson Correlation Heatmap" src="reports/corr_pearson.png" />
  <img width="48%" alt="Spearman Rank Correlation Heatmap" src="reports/corr_spearman.png" />
</p>
<p align="center">
  <img width="48%" alt="Missing Values Heatmap" src="reports/missing_heatmap.png" />
  <img width="48%" alt="Income Distribution Histogram" src="reports/hist_income.png" />
</p>
<p align="center">
  <img width="48%" alt="Income Outlier Boxplot" src="reports/box_income.png" />
  <img width="48%" alt="Credit Score Distribution" src="reports/hist_credit_score.png" />
</p>

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Git

### Installation Steps
```bash
# 1. Clone the repository
git clone https://github.com/tanyaverma20/multi-agent-data-analyst.git
cd multi-agent-data-analyst

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Configure Google Gemini API Key for AI Insights
# Create a .env file in the root directory:
echo "GEMINI_API_KEY=your_google_gemini_api_key_here" > .env
```

---

## Running the Project

### 1. Execute the Test Suite
```bash
pytest tests/ -v
```

### 2. Start the FastAPI Production Server
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Launch the Streamlit Analytical Dashboard
```bash
streamlit run streamlit_app/app.py
```
Web Studio: [http://localhost:8501](http://localhost:8501)

### 4. Run the Autonomous Multi-Agent Orchestrator CLI
```bash
python src/orchestrator.py
```

---

## Repository Structure

```text
multi-agent-data-analyst/
├── README.md                           # Project case study and documentation
├── CONTEXT.md                          # Exhaustive 40-section technical context document
├── requirements.txt                    # Project dependencies (FastAPI, Streamlit, Scikit-Learn, etc.)
├── sample.csv                          # Benchmark financial dataset (42 records, default prediction)
│
├── mcp/
│   └── manifest.json                   # Model Context Protocol tool specifications
│
├── src/                                # Application source code
│   ├── orchestrator.py                 # Multi-Agent pipeline orchestration entrypoint
│   ├── agents/                         # Autonomous analytical agents
│   │   ├── profiler_agent.py           # ProfilerAgent: Shape, nulls, and schema inspection
│   │   ├── eda_agent.py                # EDAAgent: Correlations, distributions, skew, VIF
│   │   ├── model_agent.py              # ModelAgent: AutoML, tuning, and A2A lifecycle events
│   │   ├── verifier_agent.py           # VerifierAgent: Model auditing and stability checks
│   │   ├── notebook_synthesizer_agent.py # NotebookSynthesizerAgent: .ipynb report compilation
│   │   └── memory_agent.py             # MemoryAgent: Historical key-value logging
│   │
│   ├── api/                            # Production FastAPI REST backend
│   │   ├── main.py                     # Route controllers, model loader, CORS middleware
│   │   └── schemas.py                  # Pydantic request and response schemas
│   │
│   ├── core/                           # Foundational architecture and ML engine
│   │   ├── a2a_bus.py                  # Thread-safe in-memory Agent-to-Agent message bus
│   │   └── feature_engineering.py      # OutlierHandler, FeatureCreationTransformer, FeatureSelector
│   │
│   └── tools/                          # Reusable MCP-compliant tools
│       ├── agent_tools.py              # ToolRegistry central tool mapping
│       ├── feature_tools.py            # FeatureTools MCP wrapper
│       ├── model_tools.py              # Core AutoML engine & FullAutoMLPipeline
│       ├── notebook_tools.py           # Jupyter notebook assembler & Gemini LLM client
│       ├── dataset_tools.py            # Dataset summary and profiling utilities
│       ├── memory_tools.py             # Persistent JSON key-value store
│       └── file_tools.py               # Sandboxed file management utilities
│
├── streamlit_app/                      # Interactive Streamlit studio
│   ├── app.py                          # Streamlit application home page
│   └── pages/                          # Multi-page studio views
│       ├── Dataset_Explorer.py         # Searchable dataset explorer & auto-dtypes
│       ├── Profiler.py                 # Profiler agent execution console
│       ├── EDA_Dashboard.py            # Correlation matrices & distribution plots
│       ├── AutoML.py                   # AutoML training, tuning, & feature importance studio
│       ├── Verifier.py                 # Model verification and quality auditing
│       ├── Notebook_Report.py          # Jupyter notebook generation and download portal
│       └── A2A_Dashboard.py            # Agent-to-Agent message monitoring console
│
├── models/                             # Serialized FullAutoMLPipeline artifacts (*.pkl)
├── reports/                            # Generated plots and synthesized auto_report.ipynb
└── tests/                              # Comprehensive test suite (32 tests)
    ├── test_api.py                     # FastAPI REST API tests via TestClient
    ├── test_hyperparameter_tuning.py   # GridSearchCV, RandomizedSearchCV, zero-leakage tests
    ├── test_automl_pipeline.py         # End-to-end ML, imputation, scaling, importance tests
    ├── test_feature_engineering.py     # Outlier clipping, feature creation, selector tests
    └── test_agents_integration.py      # Agent A2A communication and MCP integration tests
```

---

## Technology Stack

| Category | Technologies |
|---|---|
| **Core Language** | Python 3.10+ (tested on Python 3.13.5) |
| **Backend API** | FastAPI, Uvicorn, Pydantic, Starlette |
| **Frontend UI** | Streamlit, Plotly Express |
| **Machine Learning**| Scikit-Learn (Random Forest, Logistic Regression, GBM, Ridge, SVR, GridSearchCV) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Generative AI** | Google Generative AI (Gemini 2.0 Flash) |
| **Agent Protocols**| Custom Thread-Safe A2ABus, Model Context Protocol (MCP) Manifest |
| **Reporting** | nbformat (Jupyter Notebook synthesis) |
| **Testing** | Pytest, TestClient (HTTPX) |
| **Persistence** | Joblib, Pickle, JSON Key-Value Store |

---

## Engineering Highlights

- **Complete Data Leakage Isolation**: All feature engineering transformations, outlier clipping thresholds, categorical encodings, scaling moments, and hyperparameter grids fit exclusively on training data.
- **Production-Ready Composite Estimator**: `FullAutoMLPipeline` encapsulates all transformation steps into a single deployable object, allowing `/predict` to accept raw feature dictionaries directly.
- **Robust Mathematical Protections**: Division-by-zero safeguards replace zero denominators with `NaN` and sanitize infinities. Non-negative checks protect logarithmic transforms.
- **Decoupled Multi-Agent Coordination**: Asynchronous lifecycle events coordinate independent analytical agents without rigid monolithic coupling.
- **Standardized Tool Architecture**: Atomic tools are exposed via Model Context Protocol conventions, ready for integration with external agentic systems.

---

## Current Limitations

- **In-Memory Scale**: Designed for tabular datasets that fit comfortably in system RAM (in-memory pandas DataFrames). Out-of-core streaming (e.g., Dask, Spark) is not implemented.
- **Single Active Model In Memory**: The FastAPI server loads and serves the most recently trained model artifact from disk rather than maintaining a multi-tenant registry.
- **Exhaustive Grid Search Latency**: High-dimensional datasets with many folds can take longer under exhaustive `GridSearchCV`. In such scenarios, `RandomizedSearchCV` (configurable via API and UI) is recommended.

---

## Future Improvements

- **Persistent Model Registry**: Integrating MLflow or an S3-compatible object store for multi-model versioning and artifact tracking.
- **Asynchronous Task Queue**: Adding Celery or Redis for distributed, non-blocking background model training.
- **Containerization**: Providing a multi-stage `Dockerfile` and `docker-compose.yml` for unified FastAPI and Streamlit container deployments.
- **API Authentication**: Implementing OAuth2 / JWT bearer tokens and rate limiting for secure production hosting.

---

## Interview Talking Points

- **Multi-Agent Systems**: Demonstrates how to design collaborating autonomous agents using an event bus (`A2ABus`) and structured protocols (**MCP**).
- **Production Machine Learning**: Shows deep understanding of data leakage prevention, scikit-learn composite pipelines (`BaseEstimator`), and cross-validated hyperparameter optimization.
- **End-to-End Software Engineering**: Bridges machine learning and software engineering by providing unit and integration tests (32 tests passing), a production **FastAPI** backend, and an interactive **Streamlit** dashboard.
- **Generative AI Application**: Practical application of Google Gemini 2.0 Flash to synthesize structured data analysis results into human-readable business insights.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
