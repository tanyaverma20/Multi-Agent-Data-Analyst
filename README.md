# Multi-Agent AutoML Data Analyst (MCP + A2A + Gemini Powered)

Automated Profiler → EDA → Feature Engineering → AutoML → Hyperparameter Tuning → Cross-Validation → Feature Selection → Feature Importance → Verifier → Notebook Synthesizer → Gemini Insights

🚀 Live Demo (Render Deployment):  
https://multiagent-data-analyst.onrender.com/

Track: Enterprise Agents  
Tech: Python, FastAPI, Uvicorn, Streamlit, Scikit-Learn, Gemini, MCP Tools, A2A Bus, Multi-Agent Architecture  

---

## 📌 Overview

The **Multi-Agent Data Analyst** is a production-grade, end-to-end data analysis, machine learning, and inference system powered by collaborative autonomous agents and REST API services.
It profiles schemas, performs exploratory analysis, applies feature engineering and outlier clipping, conducts hyperparameter tuning over cross-validation across multiple model families, selects the optimal candidate, explains the model with Gemini AI, generates Jupyter notebook reports, and serves predictions via FastAPI.

Key Highlights:
- ✔ **Multi-Agent Systems**: Modular autonomous agents collaborating asynchronously over an in-memory & persistent A2A Bus.
- ✔ **Hyperparameter Tuning**: Systematic parameter exploration via `GridSearchCV` and `RandomizedSearchCV` across task-specific grids.
- ✔ **Zero Data Leakage**: All learned preprocessing, outlier clipping, feature selection, and hyperparameter tuning steps are fitted strictly on training folds.
- ✔ **FastAPI REST API**: Production API with `/health`, `/predict`, `/train`, and `/model-info` endpoints.
- ✔ **Reusable Feature Engineering**: Categorical encoding, numeric scaling, log/power transforms, and interaction creation with mathematical safeguards.
- ✔ **Automated Outlier Detection & Handling**: IQR-based outlier boundaries with safe winsorization/clipping.
- ✔ **Automated Feature Selection**: Adaptive statistical tests and model-based selection for classification & regression.
- ✔ **K-Fold Cross-Validation**: StratifiedKFold for classification and KFold for regression with fold scores and standard deviation tracking.
- ✔ **Feature Importance Analysis**: Accurate name mapping through all transformers with tree-based, linear magnitude/direction, and permutation importance.
- ✔ **MCP Tool-Based Execution**: Model Context Protocol tools for files, datasets, features, models, memory, and notebooks.
- ✔ **Gemini Integration**: LLM-generated explanations for non-technical stakeholders and engineers.
- ✔ **Interactive Streamlit UI**: Multi-page dashboard with real-time analytics, Plotly charts, and agent consoles.

---

## 🏗 System Architecture

```text
       FastAPI REST API
              ↓
    ML Inference / Training
              ↓
          Profiler
              ↓
             EDA
              ↓
     Feature Engineering
              ↓
            AutoML
              ↓
     Hyperparameter Tuning
              ↓
        Cross-Validation
              ↓
       Feature Selection
              ↓
      Feature Importance
              ↓
           Verifier
              ↓
      Notebook Synthesizer
              ↓
            Gemini

       MCP Tools  ↕  A2A Communication Bus
```

### Agent Roles

1. **Profiler Agent** (`src/agents/profiler_agent.py`): Inspects column types, row counts, null values, and publishes `profiler.completed`.
2. **EDA Agent** (`src/agents/eda_agent.py`): Computes descriptive stats, correlations (Pearson, Spearman, Kendall), VIF, skewness, kurtosis, generates plots via MCP, and publishes `eda.completed`.
3. **Model Agent** (`src/agents/model_agent.py`): Coordinates the complete AutoML process, runs hyperparameter tuning across candidate model families, selects the winning model, and emits A2A lifecycle events (`feature_engineering.started`, `feature_engineering.completed`, `hyperparameter_tuning.started`, `hyperparameter_tuning.completed`, `cross_validation.completed`, `feature_selection.completed`, `feature_importance.completed`, and `model.trained`).
4. **Verifier Agent** (`src/agents/verifier_agent.py`): Audits model quality ("Good", "Acceptable", "Weak"), verifies cross-validation stability, outlier clipping, and optimal hyperparameter selection.
5. **Notebook Synthesizer Agent** (`src/agents/notebook_synthesizer_agent.py`): Assembles markdown & code cells containing profiling, EDA summaries, feature engineering metrics, cross-validation tables, hyperparameter configurations, and feature importance rankings into a downloadable `.ipynb` file.
6. **Gemini Agent**: Synthesizes human-friendly explanations and recommendations from structured JSON outputs.

---

## ⚙️ Hyperparameter Tuning

The AutoML system performs systematic hyperparameter search across model families:

1. **Search Approaches Supported**:
   - `GridSearchCV`: Exhaustive evaluation over the defined parameter grid.
   - `RandomizedSearchCV`: Randomized exploration over candidate spaces for larger parameter combinations.
2. **Task-Specific Hyperparameter Grids**:
   - **Random Forest** (Classifier & Regressor): `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`.
   - **Logistic Regression**: `C`, `solver`.
   - **Gradient Boosting**: `n_estimators`, `learning_rate`, `max_depth`, `min_samples_split`.
   - **Ridge Regression**: `alpha`.
   - **Support Vector Regressor (SVR)**: `C`, `epsilon`, `kernel`.
3. **Cross-Validation Integration**:
   - Reuses StratifiedKFold (classification) and KFold (regression) over training folds.
   - Computes mean CV score, standard deviation, and fold-level scores for each candidate parameter combination.
4. **Zero Data Leakage**:
   - Tuning is executed exclusively on `X_train_selected` (derived from `X_train`). The held-out `X_test` is NEVER touched during parameter search.
   - The winning model is fitted on full `X_train` and evaluated once on `X_test`.

---

## 🌐 FastAPI REST API

The application includes a FastAPI backend (`src/api/main.py`) with Pydantic request/response schemas:

### Endpoints

| Method | Endpoint | Description | Request Body | Response |
|---|---|---|---|---|
| `GET` | `/health` | Service health & liveness probe | None | `{"status": "healthy", "service": "..."}` |
| `POST` | `/predict` | Model inference on raw feature data | `{"data": {...}}` or `{"data": [{...}]}` | `{"status": "success", "predictions": [...], "model_name": "...", "task_type": "..."}` |
| `POST` | `/train` | Trigger AutoML training & tuning | `{"target_col": "...", "data": [...], "cv_folds": 5, "tune_hyperparameters": true}` | `{"status": "success", "best_model": "...", "best_cv_score": ..., "test_metrics": {...}}` |
| `GET` | `/model-info` | Active model metadata, metrics, & tuning info | None | Detailed model architecture, selected features, CV score, test metrics, and tuning parameters |

#### Example: Prediction Request
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"data": {"age": 35, "income": 65000, "employment_type": "employed"}}'
```
*Note: Raw features are passed directly. The pipeline's internal transformers handle imputation, scaling, one-hot encoding, outlier clipping, and feature creation automatically.*

---

## 📊 Structured Result Data Contract

```json
{
  "status": "success",
  "task_type": "classification",
  "model_name": "rf",
  "best_model": "rf",
  "model_path": "models/rf_model.pkl",
  "feature_engineering": {
    "created_features": ["ratio_income_over_loan_amount", "diff_income_minus_loan_amount"],
    "transformed_features": ["log1p_income"],
    "numeric_features": ["age", "income", "credit_score", "loan_amount"],
    "categorical_features": ["employment_type"],
    "transformations_applied": [
      "IQR Outlier Clipping",
      "Log1p Skew Transform",
      "Pairwise Ratios & Differences",
      "Standard Scaling (Numeric)",
      "One-Hot Encoding (Categorical)",
      "Automated Feature Selection (k_best)"
    ]
  },
  "outliers": {
    "method": "iqr",
    "affected_columns": {
      "income": {"outlier_count": 2, "percentage": 5.0, "lower_bound": 12000.0, "upper_bound": 135000.0}
    },
    "total_outliers": 2,
    "percentage_affected": 5.0,
    "handling_strategy": "clip"
  },
  "feature_selection": {
    "method": "k_best",
    "original_count": 14,
    "selected_count": 10,
    "selected_features": ["income", "credit_score", "ratio_income_over_loan_amount"],
    "feature_scores": {"credit_score": 14.52}
  },
  "hyperparameter_tuning": {
    "enabled": true,
    "method": "GridSearchCV",
    "cv_folds": 5,
    "models": {
      "rf": {
        "best_params": {"max_depth": 10, "min_samples_split": 2, "n_estimators": 100},
        "best_cv_score": 0.885,
        "cv_std": 0.021,
        "candidates_evaluated": 12,
        "search_duration_sec": 0.45
      }
    },
    "best_tuned_model": "rf",
    "best_overall_params": {"max_depth": 10, "min_samples_split": 2, "n_estimators": 100}
  },
  "cross_validation": {
    "folds": 5,
    "cv_metric": "f1_score",
    "models": {
      "rf": {
        "mean_cv_score": 0.885,
        "std_cv_score": 0.021,
        "fold_scores": [0.85, 0.90, 0.87, 0.89, 0.91]
      }
    }
  },
  "feature_importance": [
    {"rank": 1, "feature": "credit_score", "importance": 0.38, "direction": "N/A"},
    {"rank": 2, "feature": "ratio_income_over_loan_amount", "importance": 0.24, "direction": "N/A"}
  ],
  "metrics": {"accuracy": 0.88, "f1_score": 0.88},
  "test_metrics": {"accuracy": 0.88, "f1_score": 0.88, "precision": 0.89, "recall": 0.88},
  "sample_predictions": [0, 1, 0, 0, 1]
}
```

---

## 🛠 MCP Tools Available

The project exposes tools adhering to the Model Context Protocol (see `mcp/manifest.json`):
- `model.tune`: Run hyperparameter tuning with cross-validation across candidate models.
- `model.cross_validate`: Run stratified or standard cross-validation.
- `model.load`, `model.predict`: Model loading and inference.
- `model.feature_importance`: Analyze and rank feature importances.
- `feature.engineer`: Run automated feature creation and transformations.
- `feature.outliers`: Detect numerical outliers and calculate clipping bounds.
- `feature.select`: Select informative features for regression and classification targets.
- `dataset.describe`, `dataset.missing`: Dataset profiling tools.
- `file.read`, `file.write`, `file.list`: Project storage management.
- `notebook.create`: Automated Jupyter Notebook report synthesis.
- `memory.read`, `memory.write`: Shared state storage.

---

## 💻 Installation & Usage

### 1. Setup Environment
```bash
git clone https://github.com/yourusername/multiagent-data-analyst.git
cd multiagent-data-analyst
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional for Gemini Insights)
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run the FastAPI Server
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```
Interactive Swagger API documentation: `http://localhost:8000/docs`

### 4. Run the Streamlit Dashboard
```bash
streamlit run streamlit_app/app.py
```

### 5. Run the Multi-Agent Pipeline via Orchestrator
```bash
python src/orchestrator.py
```

### 6. Run the Test Suite
```bash
pytest tests/ -v
```

---

## 🧪 Testing Coverage

The test suite in `tests/` contains **32 automated tests**:
- **API Tests** (`tests/test_api.py`): `/health`, `/model-info`, `/predict` (single and batch records), invalid input error handling (HTTP 400), `/train` inline training.
- **Hyperparameter Tuning Tests** (`tests/test_hyperparameter_tuning.py`): Classification tuning, regression tuning, RandomizedSearchCV, zero data leakage checks, A2A tuning lifecycle events, MCP `model.tune` method.
- **AutoML Pipeline Tests** (`tests/test_automl_pipeline.py`): End-to-end classification, regression, missing value imputation, unseen categorical levels, extreme outliers, constant columns, small datasets, feature importance name mapping.
- **Feature Engineering Tests** (`tests/test_feature_engineering.py`): IQR outlier detection and clipping, data leakage safeguards, pairwise ratios & differences, division-by-zero safeguards, automated feature selection.
- **Agent Integration Tests** (`tests/test_agents_integration.py`): ToolRegistry, ModelAgent A2A lifecycle events, VerifierAgent quality checks, NotebookSynthesizerAgent report compilation.
