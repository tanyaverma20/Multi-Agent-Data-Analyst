# CONTEXT.md: Multi-Agent AutoML Data Analyst
**Comprehensive Project Architecture, Implementation, and System Context Document**

---

> **Notice to Consuming AI Agents and Models:**
> This document is the definitive technical source of truth for the **Multi-Agent AutoML Data Analyst** repository. It was generated through deep static and dynamic inspection of the actual codebase, runtime paths, test suites, and configurations. It contains complete implementation details, class and method signatures, data flows, execution contracts, API endpoints, agent communication protocols, machine learning pipelines, and design decisions. Use this document to reason about, answer questions regarding, debug, extend, or explain any aspect of this repository without needing direct access to the source code.

---

## 1. Project Overview

### 1.1 Project Name
**Multi-Agent AutoML Data Analyst** (Internal Service Identifier: `multi-agent-automl-data-analyst`, Version: `2.0.0`)

### 1.2 One-Sentence Explanation
An autonomous, collaborative multi-agent machine learning and exploratory data analysis platform that orchestrates specialized agents across an in-memory message bus and Model Context Protocol (MCP) tool registry to perform data profiling, statistical EDA, leakage-free feature engineering, hyperparameter-tuned AutoML, rule-based verification, and Gemini-synthesized Jupyter reporting, backed by a dual-surface interface featuring a Streamlit interactive dashboard and a production-grade FastAPI REST API.

### 1.3 Interview-Ready Explanation (30–60 Seconds)
"The Multi-Agent AutoML Data Analyst is an enterprise-pattern data science automation framework built in Python. Unlike typical monolithic AutoML scripts that risk subtle target leakage or run disconnected black-box optimizations, this project structures the analytical lifecycle as a collaborative system of six specialized agents: Profiler, EDA, Model/AutoML, Verifier, Notebook Synthesizer, and Memory Agent. These agents communicate over a decoupled, thread-safe Agent-to-Agent (A2A) event bus and execute atomic capabilities registered via an MCP tool registry.

The core machine learning engine includes an automated feature engineering pipeline that handles outlier detection and IQR clipping, derived interaction/ratio generation with mathematical zero-division safeguards, categorical encoding with unseen-category tolerance, and adaptive feature selection—all strictly fitted on training splits. Hyperparameters are tuned via GridSearchCV or RandomizedSearchCV within cross-validation folds on training data only, holding out test data until final evaluation. Inference is exposed both through a multi-page Streamlit web studio and a high-performance FastAPI backend that accepts raw unstructured data records and automatically evaluates them through a serialized `FullAutoMLPipeline` composite transformer."

### 1.4 Detailed Technical Explanation
The repository addresses the fragmentation between exploratory data analysis, data pre-processing, machine learning model selection, and reporting. Traditional data science workflows suffer from manual overhead, fragile transformation scripts, and frequent data leakage—specifically when imputers, encoders, outlier bounds, or feature selectors are mistakenly fitted across an entire dataset prior to cross-validation or train/test splits.

To solve this, the architecture separates analytical concerns across distinct multi-agent layers and provides strict encapsulation:
1. **Agent Coordination & Communication**: Five functional agents and one memory agent publish and consume typed lifecycle events (e.g., `profiler.completed`, `eda.completed`, `feature_engineering.started`, `hyperparameter_tuning.completed`, `model.trained`, `verifier.completed`) across an in-memory, thread-safe message bus (`A2ABus`).
2. **Tool Execution via Model Context Protocol (MCP)**: Agent tools are decoupled from agent logic through a central `ToolRegistry` and defined in an MCP manifest (`mcp/manifest.json`), standardizing operations such as `dataset.describe`, `feature.engineer`, `feature.outliers`, `feature.select`, `model.tune`, `model.predict`, and `memory.read`.
3. **End-to-End AutoML & Feature Engineering Pipeline**: Encapsulated within `FullAutoMLPipeline`, a custom scikit-learn composite estimator chaining `OutlierHandler` (IQR clipping), `FeatureCreationTransformer` (pairwise ratios, differences, interaction products, log1p transforms), `ColumnTransformer` (median numeric imputation, standard scaling, one-hot encoding with `handle_unknown='ignore'`), and `AutomatedFeatureSelector` (univariate ANOVA F-tests / regression F-tests or model-based importance).
4. **Task-Specific Hyperparameter Search**: Automates candidate evaluation (Random Forest, Logistic Regression, Gradient Boosting for classification; Random Forest Regressor, Ridge, SVR for regression) using StratifiedKFold or KFold cross-validation on $X_{\text{train}}$, selecting winning models based on mean cross-validation score without touching $X_{\text{test}}$.
5. **Quality Verification & Synthesis**: The `VerifierAgent` performs algorithmic audits on trained models, inspecting CV standard deviations, outlier ratios, and metric thresholds to output human-readable verdicts (`Good`, `Acceptable`, `Weak`, `Unreliable`). The `NotebookSynthesizerAgent` consumes outputs from all agents and invokes Google's Gemini 2.0 Flash generative model (via `google-generativeai`) to generate structured executive insights embedded inside an auto-generated `.ipynb` Jupyter notebook.
6. **Dual-Surface Serving**: Users interact via an interactive multi-page Streamlit application (`streamlit_app/`) or integrate programmatically using a REST API (`src/api/main.py`) powered by FastAPI and Pydantic schemas.

### 1.5 Primary Target Users & Use Cases
- **Data Scientists & ML Engineers**: Rapid baseline exploration, zero-leakage feature engineering benchmarks, and automated candidate model tuning.
- **Data Analysts & Business Intelligence Teams**: Instant statistical profiling, correlation discovery (Pearson, Spearman, Kendall), VIF multicollinearity audits, and automated presentation-ready Jupyter notebooks.
- **Software Engineers & Application Developers**: Direct integration of tabular machine learning models into external applications via REST API endpoints (`POST /predict`, `POST /train`).

### 1.6 Current Implementation Status
The project is fully implemented, operational, and thoroughly verified.
- **All 32 test cases pass** across `tests/` covering unit, agent integration, feature engineering safeguards, hyperparameter tuning, and FastAPI REST endpoints.
- **End-to-end orchestration** (`python src/orchestrator.py`) runs from cold start, executing all agents sequentially and emitting artifacts without human intervention.
- **Streamlit web application** and **FastAPI REST server** run and compile cleanly with zero syntax or import errors.

### 1.7 Key Technical Highlights
- **Mathematical Zero-Division & Inf Safeguards**: Derived feature creation safeguards denominators with `.replace(0, np.nan)`, replaces infinities, and clips log1p inputs at non-negative bounds.
- **Zero Test-Set Leakage Guarantee**: Outlier boundaries, feature variance sorting, categorical encodings, scaling parameters, feature selection masks, and hyperparameter selections are fitted strictly on training folds. Test sets remain completely untouched until final evaluation.
- **Dynamic Single-Record Inference**: Callers can send raw, un-preprocessed JSON dictionaries directly to `POST /predict`. The pipeline automatically handles missing keys, unscaled numerical values, and unseen categorical levels internally.
- **Graceful External Fallbacks**: LLM features (Gemini AI insights) and VIF calculations degrade gracefully to descriptive heuristics if API keys or optional C-libraries (`statsmodels`) are not present, ensuring core functionality never crashes.

### 1.8 Important Limitations
- **In-Memory Message Bus**: The `A2ABus` uses threading locks and local file snapshots (`memory.json`). It is not a distributed message broker (such as RabbitMQ or Apache Kafka) across multi-node clusters.
- **Single Active Model In Memory**: The FastAPI application loads and serves the most recently trained model artifact from the `models/` directory or memory storage. Multi-tenant model routing or concurrent model versioning is not yet implemented.
- **Dataset Scale**: Designed for tabular datasets that fit comfortably in system RAM (in-memory pandas DataFrames). Out-of-core or distributed DataFrame processing (e.g., Dask, Spark) is not implemented.

---

## 2. Project Goals and Problem Statement

### 2.1 The Motivating Problem
Building production-grade tabular ML pipelines requires tedious, error-prone manual labor. Practitioners frequently encounter:
1. **Data Leakage in Feature Pipelines**: Preprocessing (imputing medians, standard scaling, selecting top features) is commonly applied to the entire dataset prior to cross-validation or splitting. This produces overly optimistic cross-validation scores that collapse in production.
2. **Brittle Feature Engineering**: Manual feature interaction generation frequently encounters division by zero, floating-point overflows, infinite values, or explosive feature dimensionality.
3. **Inference Pipeline Decoupling**: Models are often trained using standalone scripts, leaving downstream inference APIs to reimplement manual encoding and imputation logic. Mismatches between training-time preprocessing and inference-time transformations lead to production prediction failures.
4. **Disconnected Hyperparameter Search**: Practitioners frequently tune hyperparameters using ad-hoc scripts that leak validation data into model selection or ignore cross-validation stability.

### 2.2 System Objectives
- Autonomously ingest any tabular CSV dataset.
- Infer column types, identify missing value patterns, compute multi-metric correlations, and detect distribution anomalies.
- Safely engineer features (outlier clipping, interaction features, log transforms, adaptive feature selection) exclusively within training boundaries.
- Train, cross-validate, and hyperparameter-tune task-appropriate candidate algorithms for classification and regression.
- Audit pipeline results with automated heuristics (CV stability, metric strength, outlier impacts).
- Synthesize an interactive, reproducible Jupyter notebook containing all analytical phases and optional LLM-generated business insights.
- Expose complete model inference through an HTTP REST API with automated preprocessing.

### 2.3 What the System Does NOT Attempt to Solve
- Unstructured data processing (computer vision, natural language processing, audio/video).
- Distributed big data processing across clusters (Petabyte-scale datasets).
- Deep learning neural architecture search (NAS) or PyTorch/TensorFlow training.
- Multi-node distributed agent consensus protocols.

### 2.4 Inputs and Outputs
- **Expected Inputs**:
  - Tabular datasets in CSV format or raw JSON feature payloads (`list[dict]` or `dict`).
  - Target column name to predict.
  - Configuration flags: cross-validation fold count ($k \in [2, 10]$), tuning method (`grid` or `random`), outlier strategy (`clip` or `none`), feature selection count ($k$).
- **Expected Outputs**:
  - Serialized composite pipeline artifact (`models/<model_name>_model.pkl`).
  - Comprehensive analytical dictionary detailing all transformation metadata, CV fold metrics, test metrics, and feature importances.
  - Production REST responses with predictions, class probabilities, and model health.
  - Multi-page interactive Streamlit dashboard.
  - Formatted Jupyter notebook (`reports/auto_report.ipynb`) with visualizations and AI executive summaries.

---

## 3. Technology Stack

### 3.1 Technology Inventory

| Category | Technology | Version | Location / Usage | Role & Justification | Dependent Components |
|---|---|---|---|---|---|
| **Language** | Python | `>=3.10` (tested on `3.13.5`) | Root / All files | Core programming language | Entire codebase |
| **Backend API** | FastAPI | `^0.115.6` | `src/api/main.py` | High-performance asynchronous REST API framework | Web clients, automated tests |
| **ASGI Server** | Uvicorn | `^0.34.0` | CLI / Serving | Production-ready ASGI server for FastAPI | `src.api.main:app` |
| **Validation** | Pydantic | Core to FastAPI | `src/api/schemas.py` | Request/response data validation and typing | `src/api/main.py` |
| **Frontend UI** | Streamlit | `^1.30+` | `streamlit_app/` | Multi-page interactive analytical dashboard | Web users |
| **Data Processing** | Pandas | `^2.2.3` | Core / Agents / Tools | Data manipulation, DataFrame analysis, CSV I/O | All tools, agents, APIs |
| **Math & Arrays** | NumPy | `^1.26+` | Core / Tools | Vectorized math, arrays, metric calculations | Feature engineering, model tools |
| **Machine Learning**| Scikit-Learn | `^1.6.0` | `src/core/`, `src/tools/` | Pipelines, estimators, GridSearchCV, metrics | `ModelTools`, `FullAutoMLPipeline` |
| **Model Persistence**| Joblib / Pickle | Standard / Joblib | `src/tools/model_tools.py` | Serializing `FullAutoMLPipeline` to `.pkl` | Model storage, FastAPI loader |
| **Visualization** | Matplotlib | `^3.8+` | `src/agents/eda_agent.py` | Generating static PNG plots for reports and EDA | `EDAAgent`, Streamlit |
| **Visualization** | Seaborn | `^0.13+` | `src/agents/eda_agent.py` | Statistical visualization (heatmaps, boxplots) | `EDAAgent` |
| **Visualization** | Plotly | `^5.18+` | `streamlit_app/pages/` | Interactive charts in Streamlit pages | `EDA_Dashboard`, `AutoML` |
| **Generative AI** | Google Generative AI | `^0.8+` | `src/tools/notebook_tools.py` | Gemini 2.0 Flash integration for data insights | `NotebookTools`, Streamlit |
| **Notebook Engine**| nbformat | `^5.9+` | `src/tools/notebook_tools.py` | Synthesizing valid Jupyter Notebook (`.ipynb`) files| `NotebookSynthesizerAgent` |
| **Environment** | python-dotenv | `^1.0+` | Root / App startup | Loading `.env` file variables (`GEMINI_API_KEY`) | Streamlit, Gemini tools |
| **Testing** | Pytest | `^9.1.1` | `tests/` | Unit, integration, and API test suite execution | All test suites |
| **API Testing** | TestClient (httpx)| Bundled / Starlette | `tests/test_api.py` | In-memory integration testing of FastAPI routes | `tests/test_api.py` |
| **Stats (Optional)**| Statsmodels | Optional | `src/agents/eda_agent.py` | Multicollinearity analysis via Variance Inflation Factor | `EDAAgent` (gracefully degrades) |

---

## 4. Complete Repository Structure

```text
MultiAgent-Data-Analyst-main/
├── .gitignore                          # Git exclusions (.pytest_cache, project_storage, models, reports)
├── package.json                        # Minimal package metadata
├── package-lock.json                   # NPM lock file
├── README.md                           # Human-facing project overview, diagrams, and API docs
├── requirements.txt                    # Python dependency requirements
├── sample.csv                          # Benchmark dataset: 42 financial records (loan default prediction)
├── CONTEXT.md                          # This comprehensive context document
│
├── mcp/
│   └── manifest.json                   # Model Context Protocol tool specifications and declarations
│
├── memory/
│   ├── mcp_memory.json                 # Persistent tool and interaction memory snapshot
│   └── project_memory.json             # Key-value history storage for pipeline execution
│
├── models/                             # Output directory for serialized trained model artifacts
│   ├── gbm_model.pkl                   # Gradient Boosting composite pipeline artifact
│   ├── logreg_model.pkl                # Logistic Regression composite pipeline artifact
│   ├── rf_model.pkl                    # Random Forest composite pipeline artifact
│   └── ridge_model.pkl                 # Ridge Regressor composite pipeline artifact
│
├── project_storage/                    # FileTools root storage directory
│   ├── memory/                         # Storage for local memory dumps
│   ├── reports/                        # Storage for generated report artifacts
│   └── test.txt                        # Test file verifying FileTools read/write capability
│
├── reports/                            # Output directory for generated EDA plots and reports
│   ├── auto_report.ipynb               # Synthesized Jupyter notebook report
│   ├── eda_summary.json                # Structured JSON output from EDAAgent
│   ├── missing_heatmap.png             # Visual missing-value heatmap
│   ├── corr_pearson.png                # Pearson correlation matrix heatmap
│   ├── corr_spearman.png               # Spearman rank correlation matrix heatmap
│   ├── corr_kendall.png                # Kendall tau correlation matrix heatmap
│   ├── hist_*.png                      # Histograms for numerical columns
│   └── box_*.png                       # Boxplots for numerical columns
│
├── src/                                # Application source code
│   ├── orchestrator.py                 # CLI entry point running end-to-end multi-agent pipeline
│   ├── sample.csv                      # Source copy of benchmark loan default dataset
│   │
│   ├── agents/                         # Agent definitions (Logic, Coordination, Event Publishing)
│   │   ├── __init__.py                 # Agents package init
│   │   ├── profiler_agent.py           # ProfilerAgent: Initial dataset structure & missingness
│   │   ├── eda_agent.py                # EDAAgent: Stats, correlations, skew, VIF, visualizations
│   │   ├── model_agent.py              # ModelAgent: AutoML coordinator, tuning, lifecycle events
│   │   ├── verifier_agent.py           # VerifierAgent: Rule-based audit of model and CV stability
│   │   ├── notebook_synthesizer_agent.py # NotebookSynthesizerAgent: Compiles .ipynb + Gemini insights
│   │   └── memory_agent.py             # MemoryAgent: Historical key-value logging with timestamps
│   │
│   ├── api/                            # Production FastAPI REST backend
│   │   ├── __init__.py                 # API package init
│   │   ├── main.py                     # FastAPI server, route controllers, and CORS middleware
│   │   └── schemas.py                  # Pydantic request and response models
│   │
│   ├── core/                           # Foundational architecture and ML transformation engines
│   │   ├── a2a_bus.py                  # Thread-safe in-memory Agent-to-Agent message bus
│   │   └── feature_engineering.py      # OutlierHandler, FeatureCreationTransformer, FeatureSelector
│   │
│   └── tools/                          # Atomic tools callable by agents or registered via MCP
│       ├── a2a_tools.py                # Lightweight singleton global bus for Streamlit UI
│       ├── agent_tools.py              # ToolRegistry central registry mapping tools to MCP names
│       ├── dataset_tools.py            # Dataset summary, column profiling, duplicate detection
│       ├── feature_tools.py            # FeatureTools MCP wrapper for feature engineering ops
│       ├── file_tools.py               # Safe sandboxed text and binary file I/O operations
│       ├── job_tools.py                # Simulated long-running background job execution
│       ├── memory_tools.py             # Persistent JSON-backed key-value memory store
│       ├── model_tools.py              # Core AutoML engine, FullAutoMLPipeline, tuning, CV
│       ├── notebook_tools.py           # Jupyter notebook assembler and Gemini LLM client
│       └── plot_tools.py               # MetricValidatorTool for quick metric checks
│
├── streamlit_app/                      # Interactive multi-page Streamlit web dashboard
│   ├── app.py                          # Streamlit application entry point and Home page
│   ├── pages/                          # Streamlit multi-page interface routing
│   │   ├── A2A_Dashboard.py            # Live console inspecting inter-agent message queues
│   │   ├── AutoML.py                   # Model training, hyperparameter tuning & results studio
│   │   ├── Dataset_Explorer.py         # Tabular data inspection, duplicates, and auto-dtypes
│   │   ├── EDA_Dashboard.py            # Correlation matrices, distributions, skew, and outliers
│   │   ├── Notebook_Report.py          # Visual notebook generation and .ipynb download portal
│   │   ├── Profiler.py                 # Dataset schema, nulls, and data types
│   │   └── Verifier.py                 # Model quality auditing and recommendation viewer
│   │
│   └── streamlit_app_storage/          # Persistent session storage for Streamlit uploads and memory
│       ├── memory/
│       │   └── memory.json             # Active key-value memory database
│       └── uploads/                    # Cached uploaded CSV files
│
└── tests/                              # Pytest test suite (32 tests total)
    ├── test_agents_integration.py      # Integration tests for Agent-to-Agent communication (4 tests)
    ├── test_api.py                     # FastAPI REST API endpoint tests via TestClient (6 tests)
    ├── test_automl_pipeline.py         # End-to-end ML, imputation, scaling, importance (8 tests)
    ├── test_feature_engineering.py     # Outlier clipping, feature creation, selector (8 tests)
    └── test_hyperparameter_tuning.py   # GridSearchCV, RandomizedSearchCV, zero-leakage (6 tests)
```

---

## 5. System Architecture

### 5.1 High-Level Architecture Diagram

```text
       +-------------------------------------------------------------+
       |                        USER INTERFACE                       |
       |  Streamlit Multi-Page Studio   OR   External REST Client    |
       +--------------------+------------------------+---------------+
                            |                        |
             Direct UI Mode |                        | HTTP REST Calls
                            v                        v
       +--------------------+-------+       +--------+---------------+
       |    Streamlit Pages         |       |      FastAPI Backend   |
       | (AutoML, EDA, Explorer,    |       |  GET /health           |
       |  Profiler, Verifier, A2A)  |       |  POST /predict         |
       +------------+---------------+       |  POST /train           |
                    |                       |  GET /model-info       |
                    | Calls                 +--------+---------------+
                    v                                |
+----------------------------------------------------+--------------------------------+
|                        AGENT-TO-AGENT (A2A) COMMUNICATION LAYER                     |
|                                                                                    |
|   A2ABus (Thread-safe, Topic Queuing, Audit Log, Memory Snapshot Persistence)      |
|                                                                                    |
|    Topics: profiler.completed  -->  eda.completed  -->  model.trained             |
|            feature_engineering.started/completed                                   |
|            hyperparameter_tuning.started/completed                                 |
|            cross_validation.completed  -->  verifier.completed                    |
+-------------------+--------------------+--------------------+----------------------+
                    |                    |                    |
                    v                    v                    v
+-------------------+---+  +-------------+-----+  +-----------+----------+
|     ProfilerAgent     |  |     EDAAgent      |  |      ModelAgent      |
| Ingests raw dataframe |  | Stats, correlations| | Orchestrates AutoML, |
| Computes row/col shape|  | Heatmaps, boxplots | | tuning, CV, feature  |
| Detects data types    |  | Skewness, VIF      | | engineering pipeline |
+-----------------------+  +-------------------+  +-----------+----------+
                                                              |
                                                              v
+-----------------------+  +-------------------+  +-----------+----------+
|  NotebookSynthesizer  |  |   VerifierAgent   |  |   Core ML Engine     |
| Assembles .ipynb cells|  | Audits model R2/F1|  | FullAutoMLPipeline   |
| Embeds Gemini insights|  | Evaluates CV std  |  | OutlierHandler       |
| Outputs auto_report   |  | Verifies outliers |  | FeatureCreator       |
+-----------+-----------+  +---------+---------+  | FeatureSelector      |
            |                        |            | GridSearchCV / Rand  |
            v                        v            +-----------+----------+
+-----------+------------------------+------------------------+----------+
|                      MODEL CONTEXT PROTOCOL (MCP) TOOL LAYER                |
|                                                                             |
|  ToolRegistry:                                                              |
|   • dataset.*   (describe, missing, profile_column, duplicate_analysis)     |
|   • feature.*   (engineer, outliers, select)                                |
|   • model.*     (tune, cross_validate, predict, feature_importance)         |
|   • file.*      (read, write, write_binary, list, delete)                   |
|   • memory.*    (read, write, save, load)                                   |
|   • notebook.*  (create, append)                                            |
+------------------------------------+----------------------------------------+
                                     |
                                     v
+------------------------------------+----------------------------------------+
|                          PERSISTENCE & STORAGE                              |
|                                                                             |
|  • File System: project_storage/, reports/, models/*.pkl                    |
|  • Key-Value Store: memory.json (History arrays, snapshot recovery)         |
|  • External Services: Google Gemini 2.0 Flash API (Optional AI Insights)    |
+-----------------------------------------------------------------------------+
```

### 5.2 Component Interaction Model
The system supports two execution paths:
1. **Orchestrated Multi-Agent Autonomous Flow (`orchestrator.py`)**:
   Agents execute in a choreographed sequence where each agent finishes its analysis, persists outputs into `MemoryTools`, and publishes a typed message to `A2ABus`. The downstream agent polls or fetches its inbox, ingests the predecessor's payload, and triggers its analytical task.
2. **Direct Synchronous API / Tool Execution (`src/api/` and `src/tools/`)**:
   Endpoints in FastAPI directly leverage `ModelTools` and `FullAutoMLPipeline` to perform immediate training or inference, bypassing asynchronous agent polling while maintaining complete feature engineering and hyperparameter tuning parity.

---

## 6. End-to-End System Workflow

### 6.1 System Startup Sequence
1. **Environment Initialization**:
   - `python-dotenv` loads environment variables from `.env` (principally `GEMINI_API_KEY` or `GOOGLE_API_KEY`).
   - Root and `src/` directories are dynamically added to `sys.path` to guarantee robust module resolution regardless of working directory.
2. **Directory & Storage Verification**:
   - `ModelTools` verifies or creates `models/`.
   - `MemoryTools` verifies or creates `streamlit_app_storage/memory/` and initializes `memory.json` with an empty object `{}` if absent or corrupt.
   - `FileTools` verifies or creates `project_storage/`.
   - `EDAAgent` verifies or creates `reports/`.
3. **Tool & Agent Instantiation**:
   - `ToolRegistry` registers tool instances under namespaces: `file`, `dataset`, `model`, `notebook`, `job`, `memory`, `feature`.
   - `A2ABus` initializes threading locks, registers agent inboxes (`profiler`, `eda`, `model`, `verifier`, `notebook`), and restores persistent message snapshots from memory.

### 6.2 Primary Autonomous Pipeline Execution Flow (`src/orchestrator.py`)

```text
[Dataset CSV on Disk]
       │
       ▼
1. ProfilerAgent.run(df)
       │  • Profiles shape, types, missingness
       │  • Saves 'profiler_output' to MemoryTools
       │  • Publishes topic 'profiler.completed' to EDAAgent
       ▼
2. EDAAgent.run(df)
       │  • Computes summary statistics, correlation matrices (Pearson, Spearman, Kendall)
       │  • Generates distribution histograms, boxplots, missing-value heatmap PNGs
       │  • Computes skewness, kurtosis, and VIF
       │  • Saves 'eda_output' to MemoryTools
       │  • Publishes topic 'eda.completed' to ModelAgent
       ▼
3. ModelAgent.run(df, target_col)
       │  • Receives 'eda.completed' payload
       │  • Emits 'feature_engineering.started'
       │  • Emits 'hyperparameter_tuning.started'
       │  • Executes ModelTools.train():
       │      a. Train/Test split (80/20 held out)
       │      b. OutlierHandler (IQR bounds learned on X_train)
       │      c. FeatureCreationTransformer (Ratios, differences, interactions on X_train)
       │      d. ColumnTransformer (Median imputation, scaling, one-hot encoding on X_train)
       │      e. AutomatedFeatureSelector (ANOVA F-tests / Regression F-tests on X_train)
       │      f. Hyperparameter Search (GridSearchCV/RandomizedSearchCV over CV folds on X_train)
       │      g. Selects best model by mean CV score
       │      h. Bundles winning estimator into FullAutoMLPipeline
       │      i. Single final evaluation on untouched X_test
       │      j. Computes ranked feature importance
       │      k. Serializes FullAutoMLPipeline to models/<model_name>_model.pkl
       │  • Emits 'feature_engineering.completed', 'hyperparameter_tuning.completed',
       │    'cross_validation.completed', 'feature_selection.completed', 'feature_importance.completed'
       │  • Saves 'model_output' to MemoryTools
       │  • Emits 'model.trained' to VerifierAgent
       ▼
4. VerifierAgent.poll_messages_and_run()
       │  • Consumes 'model.trained' message
       │  • Evaluates test metric strength (R2 or F1 score) -> Quality rating
       │  • Evaluates CV stability: checks if std <= 0.08
       │  • Audits outlier counts clipped and features retained
       │  • Audits hyperparameter tuning results
       │  • Saves 'verifier_output' to MemoryTools
       │  • Publishes 'verifier.completed' to NotebookSynthesizerAgent
       ▼
5. NotebookSynthesizerAgent.run(...)
       │  • Gathers profiler_output, eda_output, model_output, verifier_output
       │  • Synthesizes Markdown and Code cells via nbformat
       │  • Formats Feature Engineering, CV table, and Feature Importance tables
       │  • Invokes Google Gemini 2.0 Flash for automated executive insights
       │  • Serializes and saves reports/auto_report.ipynb
       │  • Saves 'notebook_output' to MemoryTools
       ▼
[Completed Run Artifacts Available in reports/ and models/]
```

### 6.3 REST API Prediction Lifecycle (`POST /predict`)
1. Client issues HTTP `POST` request to `http://localhost:8000/predict` with raw JSON feature records.
2. `FastAPI` passes request body through Pydantic model `PredictionRequest`, verifying data is a dictionary or list of dictionaries.
3. Controller invokes `_load_latest_model()`. The loader queries `MemoryTools` for `model_output.model_path`, falling back to the most recently modified `.pkl` in `models/`.
4. If no model is found, returns HTTP `404 Not Found`.
5. The raw records are converted to a `pandas.DataFrame`.
6. Controller calls `model.predict(df)` on the deserialized `FullAutoMLPipeline`:
   - `OutlierHandler.transform(df)` clips numeric values to training IQR bounds.
   - `FeatureCreationTransformer.transform(df)` generates identical interaction terms and ratios.
   - `ColumnTransformer.transform(df)` executes trained imputation, standard scaling, and one-hot encoding.
   - `AutomatedFeatureSelector.transform(df)` filters columns to the exact feature mask chosen during training.
   - Underlying fitted estimator (`RandomForestClassifier`, `Ridge`, etc.) executes `.predict()`.
7. Controller formats predictions into a `PredictionResponse` JSON payload and returns HTTP `200 OK`.

---

## 7. Feature-by-Feature Breakdown

### 7.1 Automated Data Profiling & Type Detection
- **Purpose**: Establishes structural baselines before processing.
- **Implementation**: Handled by `ProfilerAgent` (`src/agents/profiler_agent.py`) and `DatasetTools` (`src/tools/dataset_tools.py`).
- **Processing**: Extracts row/column counts, missing value tallies per column, detects numeric vs categorical column types, and captures sample rows. `DatasetTools.auto_detect_dtype()` applies heuristic testing to detect datetimes, integers, floats, booleans, and categorical strings.
- **Output**: JSON dictionary saved under key `profiler_output`.

### 7.2 Exploratory Data Analysis & Statistical Auditing
- **Purpose**: Computes comprehensive bivariate and univariate statistics with zero user configuration.
- **Implementation**: Handled by `EDAAgent` (`src/agents/eda_agent.py`).
- **Processing**:
  - Computes `df.describe(include="all")`.
  - Generates three separate correlation matrices: Pearson (linear), Spearman (rank order), and Kendall Tau (concordant pairs).
  - Saves static heatmap PNGs (`corr_pearson.png`, etc.) and distribution plots (`hist_<col>.png`, `box_<col>.png`).
  - Computes skewness and kurtosis per numeric feature.
  - Calculates Variance Inflation Factor (VIF) to detect multicollinearity if `statsmodels` is installed; gracefully skips if absent.
- **Output**: JSON dictionary saved under key `eda_output`, images written to `reports/`.

### 7.3 Outlier Detection & Non-Destructive Handling
- **Purpose**: Prevents extreme anomalies from skewing linear or tree models without dropping training observations.
- **Implementation**: `OutlierHandler` in `src/core/feature_engineering.py`.
- **Processing**:
  - `fit(X_train)`: Computes $Q_1$ (25th percentile) and $Q_3$ (75th percentile) per numeric column. Calculates $\text{IQR} = Q_3 - Q_1$. Sets bounds $[\text{lower}, \text{upper}] = [Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$. Columns with $\text{IQR} \le 10^{-9}$ fallback to $[\min, \max]$.
  - `transform(X)`: Clips features to learned training bounds using `np.clip()`. Outlier bounds are never recalculated on test or validation data.
- **Output**: Summary dictionary containing outlier counts and percentage of affected rows.

### 7.4 Derived Feature Creation & Mathematical Transformations
- **Purpose**: Augments tabular feature representations through interaction terms and non-linear transformations.
- **Implementation**: `FeatureCreationTransformer` in `src/core/feature_engineering.py`.
- **Processing**:
  - **Skewness Transformation**: Checks columns with skewness $> 1.0$ and non-negative values, applying $\log(1 + x)$ via `np.log1p()`.
  - **Pairwise Interactions**: Identifies top numeric features ranked by variance. Generates pairwise ratios ($c_1 / c_2$), differences ($c_1 - c_2$), and interaction products ($c_1 \times c_2$).
  - **Zero-Division Safeguard**: In ratio calculations, zeros in denominators are converted to `np.nan`, and resulting infinite values (`inf`, `-inf`) are sanitized to `np.nan`.
  - **Dimensionality Cap**: Automatically limits generated features to `max_created_features=15` to avoid combinatorial feature explosion.

### 7.5 Automated Adaptive Feature Selection
- **Purpose**: Reduces dimensionality, removes non-informative features, and mitigates overfitting.
- **Implementation**: `AutomatedFeatureSelector` in `src/core/feature_engineering.py`.
- **Processing**:
  - Adapts to task type: uses `f_classif` (ANOVA F-value) for classification and `f_regression` for regression. Supports `model_based` selection via `RandomForest` feature importances.
  - Automatically calculates effective $k$: retains all features if total $\le 5$; selects $85\%$ if $\le 12$; selects $70\%$ if $> 12$ (or an explicit user-specified $k$).
  - Imputes intermediate NaNs via median `SimpleImputer` prior to running statistical hypothesis tests.
  - Extracts and maintains the boolean support mask to guarantee identical column filtering during downstream transformation.

### 7.6 Task-Specific Hyperparameter Tuning
- **Purpose**: Maximizes predictive accuracy by optimizing candidate hyperparameters over cross-validation folds.
- **Implementation**: `ModelTools.train()` (`src/tools/model_tools.py`) and `ModelAgent.tune()` (`src/agents/model_agent.py`).
- **Processing**:
  - Supported classification models: Random Forest, Logistic Regression, Gradient Boosting.
  - Supported regression models: Random Forest Regressor, Ridge, Support Vector Regressor (SVR).
  - Search algorithms: `GridSearchCV` (exhaustive) or `RandomizedSearchCV` (`n_iter=8` when search space $> 8$).
  - Folds: StratifiedKFold (classification) or KFold (regression), default $k=5$.
  - Evaluation: Candidate models are tuned exclusively on $X_{\text{train}}$. The winning model is selected based on mean CV score on training folds. The winning pipeline is evaluated once on held-out $X_{\text{test}}$.
- **Output**: Structured `hyperparameter_tuning` section in the result contract.

### 7.7 Automated Quality Verification & Auditing
- **Purpose**: Evaluates trained models against reliability standards and provides deployment advice.
- **Implementation**: `VerifierAgent` (`src/agents/verifier_agent.py`).
- **Processing**:
  - Evaluates metric thresholds: classifies models as `Good`, `Acceptable`, `Weak`, or `Unreliable` based on $R^2$ ($> 0.7$ is Good) or F1 Score ($> 0.75$ is Good).
  - Inspects Cross-Validation Stability: flags models with standard deviation $\le 0.08$ as stable generalizations; warns if $> 0.08$.
  - Audits outlier mitigation and feature selection ratios.

### 7.8 Generative AI Notebook Synthesis (Gemini 2.0 Flash)
- **Purpose**: Produces an executive-level, reproducible Jupyter Notebook report summarizing all agent findings.
- **Implementation**: `NotebookSynthesizerAgent` (`src/agents/notebook_synthesizer_agent.py`) and `NotebookTools` (`src/tools/notebook_tools.py`).
- **Processing**:
  - Assembles structured Markdown cells displaying tabular summaries of outlier handling, feature selection, cross-validation results, and feature importance tables.
  - Serializes raw JSON outputs from all four analytical steps into readable code/markdown blocks.
  - Sends a structured prompt containing EDA and model results to Google Gemini 2.0 Flash via `google-generativeai`.
  - Embeds the generated Markdown analysis into the final notebook and serializes to `reports/auto_report.ipynb`.

### 7.9 Model Context Protocol (MCP) Integration
- **Purpose**: Standardizes tool invocations across agents and external tool consumers.
- **Implementation**: `ToolRegistry` (`src/tools/agent_tools.py`), `FeatureTools` (`src/tools/feature_tools.py`), and `mcp/manifest.json`.
- **Exposed Tools**: `dataset.describe`, `dataset.missing`, `feature.engineer`, `feature.outliers`, `feature.select`, `model.load`, `model.predict`, `model.cross_validate`, `model.tune`, `model.feature_importance`, `file.read`, `file.write`, `memory.read`, `memory.write`.

---

## 8. Frontend Architecture (Streamlit Studio)

### 8.1 Framework & Architecture
The frontend is a multi-page interactive web application built with Streamlit (`streamlit_app/app.py` and `streamlit_app/pages/`). It operates as an analytical studio allowing users to upload datasets, inspect data profiles, trigger individual agents, configure AutoML parameters, visualize charts, inspect inter-agent messages, and download notebooks.

### 8.2 Application Entry Point (`streamlit_app/app.py`)
- Sets page configuration and title: `🤖 Multi-Agent Data Analyst`.
- Contains CSV file uploader (`st.file_uploader`), caching uploaded datasets into `st.session_state["uploaded_df"]`. If no dataset is uploaded, automatically loads `sample.csv` as a default demonstration dataset.
- Provides a four-column **MCP Tools Panel** allowing immediate execution of:
  - `FileTools`: List files, write test files, read test files.
  - `DatasetTools`: View shape, inspect column list.
  - `MemoryTools`: Display stored JSON memory dump.
  - `FeatureTools`: Trigger standalone outlier detection or feature creation.

### 8.3 Multi-Page Structure & Capabilities

#### 1. `Dataset_Explorer.py`
- Renders metric cards for total rows, total columns, numeric columns, and categorical columns.
- Provides a real-time, case-insensitive text search filtering the entire DataFrame.
- Displays interactive correlation heatmaps using Seaborn/Matplotlib.
- Includes a dedicated **Column Profiler** dropdown displaying datatype, null counts, distinct values, numerical moments (min, max, mean, std), and categorical frequency distributions.
- Analyzes row duplicates and outputs duplicate counts per column.
- Displays automated datatype suggestions generated by `DatasetTools.auto_dtype_suggestions()`.

#### 2. `Profiler.py`
- Features a single-click button to trigger `ProfilerAgent.run(df)`.
- Displays the resulting JSON structure detailing column data types, missing counts, and preview records.
- Updates session state and synchronizes memory.

#### 3. `EDA_Dashboard.py`
- Triggers `EDAAgent.run(df)`.
- Renders dataset summary metrics and correlation matrices with a dropdown to toggle between Pearson, Spearman, and Kendall heatmaps.
- Lists the top 8 correlated feature pairs by absolute value.
- Renders missing-value tables, outlier tallies, skewness/kurtosis tables, and VIF multicollinearity reports.

#### 4. `AutoML.py` (AutoML & Feature Engineering Studio)
- **Configuration Bar**:
  - Target variable dropdown (`st.selectbox`).
  - CV fold slider (`st.slider`, 2 to 10 folds, default 5).
  - Outlier handling strategy (`clip` vs `none`).
  - Hyperparameter tuning toggle checkbox (`Enable Tuning`).
  - Tuning method selector (`GridSearchCV` vs `RandomizedSearchCV`).
- **Dual Execution Controls**:
  - `🚀 Train Model (Direct ModelTools)`: Executes training synchronously through `ModelTools`.
  - `🤖 Train Model via Agent (A2A Bus)`: Executes training through `ModelAgent`, broadcasting lifecycle events over `A2ABus`.
- **Results Sections**:
  1. Pipeline Overview: Model name, task type, and test set evaluation metrics.
  2. Feature Engineering & Transformations: Lists created interaction terms, log transforms, and outlier counts.
  3. Automated Feature Selection: Displays reduction statistics and feature importance scores.
  4. Cross-Validation Stability: Renders mean CV score, standard deviation, and individual fold scores.
  5. **Hyperparameter Tuning**: Displays tuning method, candidate combinations evaluated, optimal parameters found, best CV score, and search duration.
  6. Feature Importance: Interactive horizontal bar chart (via Plotly) ranking feature contributions.
  7. Gemini AI Insights: Generates an automated natural-language explanation of model behavior.

#### 5. `Verifier.py`
- Audits the most recently trained model loaded from `st.session_state` or `MemoryTools`.
- Displays quality badge (`Good`, `Acceptable`, `Weak`, `Unreliable`).
- Lists structured audit notes covering CV stability, outlier clipping, and parameter optimizations.

#### 6. `Notebook_Report.py`
- Checks whether all prerequisite analytical outputs exist (`profiler_output`, `eda_output`, `model_output`, `verifier_output`).
- Renders a checklist showing the availability of each stage.
- Triggers `NotebookSynthesizerAgent.run()` to compile `reports/auto_report.ipynb`.
- Provides an immediate one-click download button (`st.download_button`) for the `.ipynb` notebook file.

#### 7. `A2A_Dashboard.py`
- Real-time monitoring console for inter-agent communication.
- Allows manual message dispatching (`Sender`, `Receiver`, `Topic`, `Payload`).
- Renders the UI inbox and the complete chronological audit log of all inter-agent messages.

---

## 9. Backend Architecture (FastAPI REST API)

### 9.1 Server Entry Point & Middleware (`src/api/main.py`)
- Server instance: `app = FastAPI(title="Multi-Agent AutoML Data Analyst API", version="2.0.0")`.
- Middleware: Configured with `CORSMiddleware` permitting all origins (`allow_origins=["*"]`), credentials, methods, and headers, ensuring seamless consumption by web and mobile frontends.

### 9.2 Request Lifecycle & Execution Path
```text
HTTP Request (e.g., POST /predict)
       │
       ▼
1. CORS Middleware (Validates headers and preflight OPTIONS)
       │
       ▼
2. Pydantic Parsing & Validation (src/api/schemas.py)
       │  • Rejects malformed JSON with HTTP 422 Unprocessable Entity
       │  • Validates field presence and types
       ▼
3. Route Controller (src/api/main.py)
       │  • Resolves model artifact via _load_latest_model()
       │  • If no model exists: Raises HTTP 404 Not Found
       │  • Validates dataset non-emptiness: Raises HTTP 400 Bad Request
       ▼
4. FullAutoMLPipeline Execution (src/tools/model_tools.py)
       │  • OutlierHandler.transform()
       │  • FeatureCreationTransformer.transform()
       │  • ColumnTransformer.transform()
       │  • AutomatedFeatureSelector.transform()
       │  • BaseEstimator.predict()
       ▼
5. Response Formatting & Serialization
       │  • Constructs PredictionResponse Pydantic model
       ▼
HTTP Response (JSON, HTTP 200 OK)
```

### 9.3 Dynamic Model Artifact Loader (`_load_latest_model()`)
The API dynamically locates the active model artifact without hardcoding file paths:
1. Queries `MemoryTools().load("model_output")`. If a valid `model_path` key exists and the file is present on disk, loads it via `pickle.load()`.
2. Fallback: Scans `models/` directory for any `.pkl` files, sorts them by operating system modification time (`os.path.getmtime`), and deserializes the most recently generated pipeline.
3. Returns `None` if no model files exist or deserialization fails.

---

## 10. API Documentation

### 10.1 GET `/health`
- **Purpose**: Liveness and readiness probe for health monitoring and container orchestrators.
- **Authentication**: None required.
- **Request Parameters / Body**: None.
- **Response Format**: `HealthResponse`
- **Example Response (HTTP 200)**:
  ```json
  {
    "status": "healthy",
    "service": "multi-agent-automl-data-analyst"
  }
  ```

---

### 10.2 POST `/predict`
- **Purpose**: Generates predictions on raw feature data using the trained `FullAutoMLPipeline`. Preprocessing, missing-value imputation, outlier clipping, and feature creation occur automatically.
- **Authentication**: None required.
- **Request Body**: `PredictionRequest`
  - Accepts a single feature dictionary or a list of feature dictionaries.
- **Example Request 1 (Single Record)**:
  ```json
  {
    "data": {
      "age": 35,
      "income": 65000,
      "employment_type": "employed",
      "loan_amount": 20000,
      "credit_score": 710
    }
  }
  ```
- **Example Request 2 (Batch Records)**:
  ```json
  {
    "data": [
      { "age": 22, "income": 28000, "employment_type": "unemployed", "credit_score": 580 },
      { "age": 45, "income": 120000, "employment_type": "self_employed", "credit_score": 790 }
    ]
  }
  ```
- **Example Response (HTTP 200)**:
  ```json
  {
    "status": "success",
    "model_name": "rf",
    "task_type": "classification",
    "predictions": [0, 0]
  }
  ```
- **Error Responses**:
  - `HTTP 400 Bad Request`: When `data` is empty (`{"data": []}`) or cannot be converted to a DataFrame.
  - `HTTP 404 Not Found`: When no trained model artifact exists on the server.
  - `HTTP 500 Internal Server Error`: If an unexpected transformation exception occurs during estimator inference.

---

### 10.3 POST `/train`
- **Purpose**: Triggers AutoML pipeline training, automated feature engineering, cross-validation, and hyperparameter tuning.
- **Authentication**: None required.
- **Request Body**: `TrainRequest`
  - `target_col` (str, required): Name of target column to predict.
  - `data` (list[dict], optional): Inline tabular dataset rows.
  - `csv_path` (str, optional): Absolute or relative path to CSV file on disk.
  - `cv_folds` (int, optional, default=5): Number of cross-validation folds ($2 \le k \le 10$).
  - `tune_hyperparameters` (bool, optional, default=true): Enable hyperparameter tuning.
  - `tuning_method` (str, optional, default="grid"): Search strategy (`"grid"` or `"random"`).
- **Example Request**:
  ```json
  {
    "target_col": "default",
    "csv_path": "sample.csv",
    "cv_folds": 3,
    "tune_hyperparameters": true,
    "tuning_method": "grid"
  }
  ```
- **Example Response (HTTP 200)**:
  ```json
  {
    "status": "success",
    "task_type": "classification",
    "best_model": "rf",
    "best_cv_score": 0.8842,
    "test_metrics": {
      "accuracy": 0.8889,
      "f1_score": 0.8756,
      "precision": 0.8889,
      "recall": 0.8889
    },
    "hyperparameter_tuning": {
      "enabled": true,
      "method": "GridSearchCV",
      "cv_folds": 3,
      "best_tuned_model": "rf",
      "best_overall_params": {
        "max_depth": 5,
        "n_estimators": 50
      }
    },
    "message": "Model trained and hyperparameter-tuned successfully"
  }
  ```
- **Error Responses**:
  - `HTTP 400 Bad Request`: If `csv_path` does not exist, `target_col` is missing from the dataset, or training fails.

---

### 10.4 GET `/model-info`
- **Purpose**: Retrieves architectural metadata, feature counts, cross-validation performance, and evaluation metrics for the currently active model.
- **Authentication**: None required.
- **Request Parameters / Body**: None.
- **Response Format**: `ModelInfoResponse`
- **Example Response (HTTP 200)**:
  ```json
  {
    "status": "success",
    "model_name": "rf",
    "task_type": "classification",
    "model_path": "models/rf_model.pkl",
    "feature_count": 8,
    "selected_feature_count": 6,
    "selected_features": [
      "income",
      "credit_score",
      "loan_amount",
      "ratio_income_over_loan_amount",
      "diff_income_minus_loan_amount",
      "employment_type_employed"
    ],
    "cv_metric": "f1_score",
    "cv_score": 0.8842,
    "test_metrics": {
      "accuracy": 0.8889,
      "f1_score": 0.8756,
      "precision": 0.8889,
      "recall": 0.8889
    },
    "hyperparameter_tuning": {
      "enabled": true,
      "method": "GridSearchCV",
      "cv_folds": 3
    },
    "timestamp": null
  }
  ```
- **Error Responses**:
  - `HTTP 404 Not Found`: If no model has been trained yet.

---

## 11. Database Architecture & Memory Persistence

### 11.1 Persistence Strategy
The system does not require an external SQL or NoSQL database server. Persistence is achieved through JSON-based document storage and serialized pickle binaries:

1. **`MemoryTools` Database (`streamlit_app_storage/memory/memory.json`)**:
   - Acts as a local key-value document store.
   - Initialized automatically if missing or corrupted.
   - Operations:
     - `save(key, value)`: Updates or inserts a top-level key in `memory.json`.
     - `load(key=None)`: Retrieves value for `key` or the entire dictionary if `key=None`.
     - `load_all()`: Returns the full stored JSON object.
2. **`MemoryAgent` Historical Log (`memory/project_memory.json`)**:
   - Manages a chronological event ledger.
   - Schema:
     ```json
     {
       "history": [
         {
           "timestamp": "2026-10-08 14:30:15",
           "key": "eda_output",
           "value": { ... }
         }
       ]
     }
     ```
   - Includes automatic serialization sanitization via `_make_json_safe()`, converting NaN values to `None`, ndarrays to lists, and unhandled objects to strings.
3. **`A2ABus` Message Snapshots**:
   - Automatically writes message inbox snapshots to memory under the key `a2a_snapshot` and in the audit list under `_audit`.
4. **Model Binary Storage (`models/*.pkl`)**:
   - Pickled instances of `FullAutoMLPipeline`, preserving complete internal transformation parameters, scaling moments, and estimator weights.

---

## 12. Authentication and Authorization

### 12.1 Observed State
- **Current Implementation**: Authentication and authorization mechanisms are **not implemented** in the repository.
- **API Endpoints**: All FastAPI endpoints (`/health`, `/predict`, `/train`, `/model-info`) are completely open and unauthenticated.
- **Streamlit Interface**: Open without user login, role checks, or session authentication.
- **Architectural Implication**: Suitable for internal analytical use, trusted VPC microservice deployments, or local developer workstations. Production exposure on public networks would require an API gateway, OAuth2 bearer token middleware, or reverse proxy authentication.

---

## 13. AI / Machine Learning Architecture

### 13.1 Automated Task Detection
Implemented in `ModelTools._detect_task(y: pd.Series)`:
- Categorical / object targets $\to$ `classification`.
- Numeric targets with $> 20$ unique values $\to$ `regression`.
- Floating-point numeric targets with $> 10$ unique values $\to$ `regression`.
- Numeric targets with $\le 10$ unique values $\to$ `classification`.

### 13.2 Candidate Estimator Suite

#### Classification Candidates
1. **Random Forest (`rf`)**: `sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=42)`
2. **Logistic Regression (`logreg`)**: `sklearn.linear_model.LogisticRegression(max_iter=1000, random_state=42)`
3. **Gradient Boosting (`gbm`)**: `sklearn.ensemble.GradientBoostingClassifier(n_estimators=80, random_state=42)`

#### Regression Candidates
1. **Random Forest Regressor (`rf`)**: `sklearn.ensemble.RandomForestRegressor(n_estimators=100, random_state=42)`
2. **Ridge Regression (`ridge`)**: `sklearn.linear_model.Ridge()`
3. **Support Vector Regressor (`svr`)**: `sklearn.svm.SVR()`

### 13.3 Configurable Hyperparameter Grids

```python
# Classification Parameter Spaces
"rf": {
    "n_estimators": [50, 100],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}
"logreg": {
    "C": [0.1, 1.0, 10.0],
    "solver": ["lbfgs"],
}
"gbm": {
    "n_estimators": [50, 80],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "min_samples_split": [2, 5],
}

# Regression Parameter Spaces
"rf": {
    "n_estimators": [50, 100],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}
"ridge": {
    "alpha": [0.1, 1.0, 10.0, 50.0],
}
"svr": {
    "C": [0.5, 1.0, 5.0],
    "epsilon": [0.1, 0.2],
    "kernel": ["rbf", "linear"],
}
```

### 13.4 Strict Zero-Data-Leakage Architecture
The repository enforces complete data isolation between training and testing data:
1. **Train/Test Split**: Initial split is performed at the start of `ModelTools.train()` ($80\%$ train, $20\%$ test). Stratification is applied for classification if minimum class count $\ge 2$.
2. **$X_{\text{train}}$-Only Fitting**:
   - `outlier_handler.fit(X_train)`: Calculates IQR bounds exclusively on training features.
   - `feature_creator.fit(X_train_clean, y_train)`: Computes feature variances and skewness exclusively on training records.
   - `preprocessor.fit(X_train_engineered)`: Computes median imputation values, numeric means, and standard deviations exclusively on training records. One-hot categories are indexed strictly on training classes.
   - `feature_selector.fit(X_train_preprocessed, y_train)`: Computes ANOVA F-scores or regression F-scores strictly on training data.
   - `GridSearchCV` / `RandomizedSearchCV`: Cross-validation splits are created exclusively from $X_{\text{train}}$. Candidate estimators are fitted on $(k-1)$ folds and evaluated on the remaining internal validation fold.
3. **Single Held-Out Test Evaluation**:
   - The winning estimator is wrapped inside `FullAutoMLPipeline`.
   - $X_{\text{test}}$ is passed to `final_pipeline.predict(X_test)` exactly once to compute held-out performance metrics.
   - Test metrics are never used to choose models, select hyperparameters, or calculate feature statistics.

### 13.5 Composite Pipeline: `FullAutoMLPipeline`
`FullAutoMLPipeline` is a custom scikit-learn estimator defined in `src/tools/model_tools.py` that encapsulates all transformation steps:
```python
class FullAutoMLPipeline(BaseEstimator):
    def __init__(self, outlier_handler, feature_creator, preprocessor, feature_selector, estimator, ...):
        ...
    def transform_features(self, X):
        X_out = self.outlier_handler.transform(X)
        X_out = self.feature_creator.transform(X_out)
        X_prep = self.preprocessor.transform(X_out)
        X_sel = self.feature_selector.transform(X_prep)
        return X_sel

    def predict(self, X):
        return self.estimator.predict(self.transform_features(X))

    def predict_proba(self, X):
        return self.estimator.predict_proba(self.transform_features(X))
```

### 13.6 Feature Importance Interpretation Engine
Implemented in `ModelTools._compute_feature_importance()`:
- **Tree-Based Models**: Reads `estimator.feature_importances_`.
- **Linear Models**: Reads `estimator.coef_`, computes absolute value for magnitude, and classifies directionality as `positive`, `negative`, or `multiclass`.
- **Black-Box Models (SVR Fallback)**: Computes `permutation_importance(estimator, X_val, y_val, n_repeats=5)`.
- **Name Recovery**: Correctly maps numeric feature indices back to human-readable names generated after one-hot encoding, ratios, differences, and log transforms.

---

## 14. Generative AI & LLM Architecture

### 14.1 LLM Provider and Configuration
- **Provider**: Google Generative AI (via official package `google-generativeai`).
- **Target Model**: `gemini-2.0-flash`.
- **Authentication**: Reads `os.environ.get("GOOGLE_API_KEY")` or `os.environ.get("GEMINI_API_KEY")`.

### 14.2 Prompt Construction & Information Injection
The system employs structured JSON injection in `NotebookTools._generate_gemini_insights()`:
```python
prompt = f"""
You are a senior ML engineer generating insights for a professional ML report.

Write:
1. Key Data Findings from EDA & Preprocessing
2. Model Strengths & Weaknesses (including cross-validation stability)
3. Most Important Features & Interpretability
4. Suggested Next Steps & Risk Factors

Use the following JSON:
EDA: {json.dumps(eda_output, indent=2)}
MODEL: {json.dumps(model_output, indent=2)}
"""
```

### 14.3 Fallbacks and Error Handling
- If `google-generativeai` is not installed, returns an informative string: `ℹ️ google-generativeai package not installed.`
- If API keys are missing, returns: `ℹ️ Gemini API key not found. Set GEMINI_API_KEY to generate automated AI insights.`
- Network or quota errors are caught in a `try...except` block, returning a clean error message rather than halting pipeline execution.

---

## 15. Retrieval-Augmented Generation (RAG) Architecture

### 15.1 Architectural Status
- **RAG Implementation**: **Not implemented** in this repository.
- There are no document loaders, text chunkers, vector embedding models (such as SentenceTransformers), or vector databases (such as Chroma, FAISS, Pinecone, Qdrant).
- Information passed to Gemini is derived strictly from runtime in-memory JSON summaries generated by `EDAAgent` and `ModelAgent`.

---

## 16. Agent & Multi-Agent Architecture

### 16.1 Agent Roles and Capabilities

```text
┌───────────────────────────┬─────────────────────────────────────────────────────────────┐
│ Agent Name                │ Primary Responsibilities & Execution                        │
├───────────────────────────┼─────────────────────────────────────────────────────────────┤
│ ProfilerAgent             │ • Ingests raw dataframe; computes shape and column types    │
│                           │ • Detects null distributions; publishes 'profiler.completed'│
├───────────────────────────┼─────────────────────────────────────────────────────────────┤
│ EDAAgent                  │ • Computes Pearson, Spearman, Kendall correlation matrices  │
│                           │ • Computes skewness, kurtosis, and VIF                      │
│                           │ • Generates distribution and heatmap PNG plots              │
│                           │ • Publishes 'eda.completed'                                 │
├───────────────────────────┼─────────────────────────────────────────────────────────────┤
│ ModelAgent                │ • Orchestrates end-to-end AutoML and feature engineering    │
│                           │ • Coordinates hyperparameter search (Grid/Random)           │
│                           │ • Publishes lifecycle events:                               │
│                           │     - feature_engineering.started / completed               │
│                           │     - hyperparameter_tuning.started / completed             │
│                           │     - cross_validation.completed                            │
│                           │     - feature_selection.completed                           │
│                           │     - feature_importance.completed                          │
│                           │     - model.trained                                         │
├───────────────────────────┼─────────────────────────────────────────────────────────────┤
│ VerifierAgent             │ • Consumes 'model.trained' message                          │
│                           │ • Evaluates metric strength and CV stability thresholds     │
│                           │ • Audits outlier mitigation and feature selection           │
│                           │ • Emits audit verdict and publishes 'verifier.completed'    │
├───────────────────────────┼─────────────────────────────────────────────────────────────┤
│ NotebookSynthesizerAgent  │ • Gathers outputs from all previous four agents             │
│                           │ • Compiles Markdown and Code cells via nbformat             │
│                           │ • Injects Gemini 2.0 Flash automated insights               │
│                           │ • Writes final reports/auto_report.ipynb                    │
├───────────────────────────┼─────────────────────────────────────────────────────────────┤
│ MemoryAgent               │ • Append-only historical key-value persistence              │
│                           │ • Sanitizes data types with timestamps                      │
└───────────────────────────┴─────────────────────────────────────────────────────────────┘
```

### 16.2 Agent-to-Agent Bus (`A2ABus`) Implementation
Implemented in `src/core/a2a_bus.py`:
- **Thread Safety**: Uses `threading.Lock()` to protect dictionary reads and writes across concurrent threads.
- **Addressing**: Supports direct agent addressing (`to="model"`), comma-separated multi-recipient routing (`to="verifier,notebook"`), and broadcasting (`to="broadcast"`).
- **Message Contract**:
  ```python
  {
      "from": from_agent,
      "to": to,
      "topic": topic,
      "payload": payload,
      "meta": meta or {},
      "timestamp": "YYYY-MM-DD HH:MM:SS"
  }
  ```
- **Audit Logging**: Every message dispatched on the bus is appended to an internal `_audit` queue, inspectable via `a2a.audit_log()`.
- **Queue Semantics**: `fetch(agent_name, consume=True)` returns and removes messages from the recipient's inbox. `peek(agent_name)` inspects messages non-destructively.

---

## 17. AI/ML + Application Integration

### 17.1 Runtime Coupling
The integration bridge between machine learning and serving surfaces is governed by two contracts:
1. **Model Storage Contract (`FullAutoMLPipeline`)**:
   Saved to `models/<best_model>_model.pkl`. It is completely self-contained; loading it requires no external column names, transformation statistics, or preprocessing scripts.
2. **Memory Information Contract (`model_output`)**:
   Saved to `MemoryTools` under key `model_output`. Contains the full analytical contract:
   - `status`: `"success"`
   - `task_type`: `"classification"` or `"regression"`
   - `best_model`: Name of the winning candidate model (e.g. `"rf"`)
   - `model_path`: File path to the `.pkl` artifact
   - `metrics`: Dictionary of test metrics (`accuracy`, `f1_score`, `precision`, `recall`, `roc_auc` or `mse`, `rmse`, `mae`, `r2`)
   - `hyperparameter_tuning`: Method, folds, candidate counts, and optimal parameters
   - `cross_validation`: Mean CV score and standard deviations per candidate model
   - `feature_importance`: Ranked list of top features with impact directions
   - `feature_engineering`: Transformations applied, created features, transformed features
   - `outliers`: Outlier counts, affected columns, and clipping bounds

---

## 18. Data Flow

### 18.1 Detailed Tabular Training Data Flow

```text
Raw Tabular CSV (e.g., sample.csv)
  │
  ▼ [pandas.read_csv]
DataFrame: shape (N, M)
  │
  ▼ [train_test_split(test_size=0.2, random_state=42)]
  ├────────────────────────────────────────────────┐
  │                                                │
  ▼                                                ▼
X_train (80%), y_train                    X_test (20%), y_test
  │                                       (HELD OUT - UNTOUCHED)
  ▼
OutlierHandler.fit_transform(X_train)
  │ • Computes IQR = Q3 - Q1
  │ • Establishes lower/upper bounds
  │ • Clips numeric anomalies
  ▼
X_train_clean
  │
  ▼
FeatureCreationTransformer.fit_transform(X_train_clean, y_train)
  │ • Log1p transform on skewed positive columns
  │ • Pairwise ratios, differences, interaction terms
  │ • Zero-division protection
  ▼
X_train_engineered
  │
  ▼
ColumnTransformer.fit_transform(X_train_engineered)
  │ • Numeric Pipeline: SimpleImputer(median) -> StandardScaler()
  │ • Categorical Pipeline: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')
  ▼
X_train_preprocessed (Dense 2D NumPy array)
  │
  ▼
AutomatedFeatureSelector.fit_transform(X_train_preprocessed, y_train)
  │ • Classification: SelectKBest(f_classif, k=effective_k)
  │ • Regression: SelectKBest(f_regression, k=effective_k)
  │ • Generates boolean feature mask
  ▼
X_train_selected
  │
  ▼
Cross-Validation & Hyperparameter Tuning Loop
  │ • StratifiedKFold (classification) or KFold (regression)
  │ • GridSearchCV / RandomizedSearchCV evaluates candidate grids
  │ • Computes mean CV score on internal folds
  ▼
Winning Estimator Identified
  │
  ▼
FullAutoMLPipeline Instantiated
  │ (OutlierHandler + FeatureCreator + ColumnTransformer + FeatureSelector + Winning Estimator)
  │
  ▼ [Final Evaluation Step]
FinalPipeline.predict(X_test)
  │ • Passes raw X_test through pipeline
  │ • Computes accuracy, F1, Precision, Recall or MSE, RMSE, MAE, R2
  ▼
Feature Importance Analysis (Tree importances / Linear coefficients / Permutation)
  │
  ▼
Pickle Dump -> models/<model_name>_model.pkl
```

---

## 19. Important Classes, Functions, and Modules

### 19.1 `src/core/feature_engineering.py`
- **`OutlierHandler(BaseEstimator, TransformerMixin)`**:
  - `fit(X, y=None)`: Learns IQR clipping thresholds per numerical feature on training data.
  - `transform(X)`: Clips features to learned thresholds.
  - `get_summary()`: Returns bounds, affected columns, and outlier counts.
- **`FeatureCreationTransformer(BaseEstimator, TransformerMixin)`**:
  - `fit(X, y=None)`: Discovers skewed features and prioritizes variance pairs for ratios and interaction products.
  - `transform(X)`: Computes safe log1p transforms, ratios with zero-division protection, differences, and interactions.
  - `get_summary()`: Returns lists of created and transformed feature names.
- **`AutomatedFeatureSelector(BaseEstimator, TransformerMixin)`**:
  - `fit(X, y)`: Runs statistical tests (`f_classif` / `f_regression`) or Random Forest feature selection to learn feature masks.
  - `transform(X)`: Filters columns according to learned support mask.

### 19.2 `src/tools/model_tools.py`
- **`FullAutoMLPipeline(BaseEstimator)`**:
  - Encapsulates preprocessors, transformers, selectors, and estimators into a single deployable unit.
  - `predict(X)` / `predict_proba(X)`: Exposes safe inference directly on raw feature inputs.
- **`ModelTools`**:
  - `train(...)`: Full AutoML execution method with cross-validation and hyperparameter tuning.
  - `tune(...)`: Dedicated tuning method exposed as an MCP tool.
  - `_detect_task(y)`: Automatic task type classification.
  - `_get_hyperparameter_grids(task)`: Supplies sensible, computationally efficient parameter grids.
  - `_compute_feature_importance(...)`: Calculates feature importance rankings with reconstructed feature names.

### 19.3 `src/core/a2a_bus.py`
- **`A2ABus`**:
  - `publish(from_agent, to, topic, payload, meta)`: Dispatches messages to recipient inboxes and audit logs.
  - `fetch(agent_name, consume=True)`: Retrieves messages for an agent with optional consumption.
  - `peek(agent_name)`: Reads agent messages without removing them.
  - `audit_log()`: Returns the global chronological message history.

### 19.4 `src/api/main.py`
- **`_load_latest_model()`**: Locates and deserializes the active model artifact from `models/` or memory.
- **`health_check()`**: Handler for `GET /health`.
- **`predict(request)`**: Handler for `POST /predict`.
- **`train_model(request)`**: Handler for `POST /train`.
- **`get_model_info()`**: Handler for `GET /model-info`.

---

## 20. Configuration

### 20.1 Configuration Constants
- **`test_size`**: `0.2` (Fixed 80% train / 20% test split).
- **`cv_folds`**: Default `5` (configurable between 2 and 10).
- **`max_created_features`**: `15` (Prevents exponential column growth).
- **`skew_threshold`**: `1.0` (Applies log1p when skewness exceeds 1.0).
- **`random_state`**: `42` (Fixed pseudo-random seed for deterministic reproducibility).
- **`outlier_factor`**: `1.5` (Standard Tukey IQR multiplier).

---

## 21. Environment Variables

| Variable | Purpose | Required | Used By | Safe Placeholder |
|---|---|---|---|---|
| `GEMINI_API_KEY` | Authentication for Google Gemini Generative AI API | No (Optional) | `NotebookTools`, `streamlit_app/pages/AutoML.py` | `<your-gemini-api-key>` |
| `GOOGLE_API_KEY` | Fallback environment variable for Google Gemini API | No (Optional) | `NotebookTools`, `streamlit_app/pages/AutoML.py` | `<your-google-api-key>` |

*Note: All core features (profiling, EDA, feature engineering, AutoML, tuning, verifier, API serving) operate successfully without Gemini API keys configured.*

---

## 22. External Services and APIs

### 22.1 Google Gemini API
- **Service Name**: Google Generative AI (Gemini API)
- **Role**: Automated natural-language insight generation.
- **Model**: `gemini-2.0-flash`.
- **Authentication**: API key via `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
- **Failure Behavior**: If the key is absent or requests fail, the application logs an informational notice and proceeds without breaking.

---

## 23. Error Handling

### 23.1 Validation Errors
- **API Request Validation**: Handled automatically by Pydantic; invalid payloads return `422 Unprocessable Entity`.
- **Empty Dataset Protection**: Empty data lists return `400 Bad Request` with an explicit explanation.

### 23.2 Mathematical and Transformation Errors
- **Division by Zero**: Safeguarded in `FeatureCreationTransformer` by replacing zero denominators with `np.nan`.
- **Infinities**: Sanitized using `.replace([np.inf, -np.inf], np.nan)`.
- **Negative Values in Log Transforms**: Clipped to zero (`np.maximum(0, val)`) before applying `np.log1p()`.
- **Constant Features**: Columns with near-zero variance fallback to min/max clipping bounds.

### 23.3 Machine Learning Pipeline Failures
- **Unseen Categorical Levels**: Handled gracefully at inference time by `OneHotEncoder(handle_unknown='ignore')`.
- **Single-Class Target or Statistical Failures**: `AutomatedFeatureSelector` catches statistical exceptions and falls back to selecting all features.

---

## 24. Security

### 24.1 Security Posture
- **Secret Management**: API keys are accessed exclusively via environment variables (`os.environ.get(...)`); no keys are hardcoded in the codebase.
- **Cross-Origin Resource Sharing (CORS)**: Configured with `allow_origins=["*"]` on FastAPI.
- **File System Sandboxing**: `FileTools` confines file operations to `project_storage/`.
- **Observed Production Gaps**:
  - Pickle deserialization (`pickle.load()`) is used to load saved models. Only trusted model files should be loaded.
  - Endpoints have no rate limiting or authentication.

---

## 25. CORS, Networking, and Ports

- **FastAPI Backend**:
  - Default Port: `8000`
  - Command: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload`
- **Streamlit Frontend**:
  - Default Port: `8501`
  - Command: `streamlit run streamlit_app/app.py`

---

## 26. Docker and Deployment

### 26.1 Containerization Status
- Dockerfiles and `docker-compose.yml` are **not present** in the repository root.
- The project runs directly in any Python 3.10+ virtual environment.

---

## 27. Build and Runtime Process

### 27.1 Installation
```bash
# Clone repository and enter directory
cd MultiAgent-Data-Analyst-main

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 27.2 Running the Application
```bash
# 1. Run the Multi-Agent Autonomous Orchestrator CLI
python src/orchestrator.py

# 2. Launch the Streamlit Analytical Studio
streamlit run streamlit_app/app.py

# 3. Start the FastAPI Production REST API Server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Execute the Complete Test Suite
pytest tests/ -v
```

---

## 28. Testing

### 28.1 Test Inventory (32 Tests Passing)
- **`tests/test_agents_integration.py` (4 tests)**:
  - `test_tool_registry_includes_feature_tool`: Verifies `ToolRegistry` contains all required namespaces.
  - `test_model_agent_a2a_lifecycle_events`: Verifies `ModelAgent` emits complete lifecycle events to `A2ABus`.
  - `test_verifier_agent_evaluates_automl_output`: Verifies `VerifierAgent` generates quality ratings and audit notes.
  - `test_notebook_synthesizer_with_rich_automl`: Verifies `NotebookSynthesizerAgent` creates `.ipynb` notebooks.
- **`tests/test_api.py` (6 tests)**:
  - `test_get_health`: Verifies `GET /health` returns status 200.
  - `test_get_model_info`: Verifies `GET /model-info` returns active model metadata.
  - `test_post_predict_single_record`: Verifies `POST /predict` handles single dictionary inputs.
  - `test_post_predict_batch_records`: Verifies `POST /predict` handles batch record lists.
  - `test_post_predict_invalid_data`: Verifies `POST /predict` returns 400 for empty payloads.
  - `test_post_train_inline_data`: Verifies `POST /train` executes inline AutoML training.
- **`tests/test_automl_pipeline.py` (8 tests)**:
  - `test_classification_end_to_end`: Tests full classification pipeline with categorical and numeric data.
  - `test_regression_end_to_end`: Tests full regression pipeline and metrics.
  - `test_missing_values_handling`: Validates imputation of NaNs in both numeric and categorical columns.
  - `test_unseen_categorical_values`: Validates prediction robustness on unseen categories.
  - `test_outlier_handling_in_pipeline`: Validates outlier detection and clipping.
  - `test_constant_column_dataset`: Validates pipeline robustness on zero-variance columns.
  - `test_small_dataset`: Validates execution on small sample counts.
  - `test_feature_importance_mapping`: Validates feature importance names match post-transformation features.
- **`tests/test_feature_engineering.py` (8 tests)**:
  - `test_outlier_detection_and_clipping`: Validates IQR bounds and clipping.
  - `test_outlier_handler_no_data_leakage`: Proves outlier bounds do not change when transforming test data.
  - `test_feature_creation_transformer`: Validates ratio, difference, and interaction term generation.
  - `test_feature_creation_safeguards_division_by_zero`: Validates zero-division safeguards.
  - `test_feature_creation_constant_column`: Validates feature creation on constant inputs.
  - `test_automated_feature_selector_classification`: Validates classification feature selection.
  - `test_automated_feature_selector_regression`: Validates regression feature selection.
  - `test_feature_tools_mcp_standalone`: Validates standalone `FeatureTools` methods.
- **`tests/test_hyperparameter_tuning.py` (6 tests)**:
  - `test_hyperparameter_tuning_classification`: Validates grid search across classification models.
  - `test_hyperparameter_tuning_regression`: Validates grid search across regression models.
  - `test_randomized_search_cv`: Validates `RandomizedSearchCV` execution path.
  - `test_no_data_leakage_during_hyperparameter_tuning`: Proves model selection is driven strictly by CV scores on training data.
  - `test_model_agent_emits_tuning_a2a_events`: Validates emission of tuning start and completion events.
  - `test_feature_tools_tune_mcp_method`: Validates MCP `tune_model()` execution.

---

## 29. Performance and Optimization

1. **Lightweight Search Spaces**: Parameter grids are curated to evaluate high-impact parameters (`n_estimators`, `max_depth`, `C`, `alpha`) without combinatorial explosion.
2. **Adaptive Search Strategy**: Automatically switches from `GridSearchCV` to `RandomizedSearchCV` (`n_iter=8`) when total combinations exceed 8.
3. **Variance Sorting Before Interaction**: Sorts features by variance and caps candidates to top 6, preventing factorial feature explosion ($O(N^2)$).
4. **Vectorized NumPy & Pandas Operations**: All transformations (ratios, differences, clipping, log transforms) use vectorized operations.

---

## 30. Important Design Decisions

1. **Composite `FullAutoMLPipeline` Over Disconnected Preprocessors**:
   - *Rationale*: Storing preprocessors separately from estimators often leads to preprocessing mismatches during production inference. Packaging all transformations inside a single object guarantees identical inference transformations.
2. **IQR Clipping Over Row Dropping**:
   - *Rationale*: Deleting rows containing outliers distorts sample distributions and causes crashes during inference if a single incoming record contains an outlier. Clipping winsorizes anomalies safely without dropping rows.
3. **In-Memory Thread-Safe Bus Over External Broker**:
   - *Rationale*: Avoids heavy external infrastructure dependencies (such as Redis or RabbitMQ) while demonstrating agent-to-agent decoupling.
4. **Graceful Fallbacks for External Services (Gemini / Statsmodels)**:
   - *Rationale*: Ensures the analytical pipeline remains fully functional even in offline, private, or air-gapped environments without external API keys.

---

## 31. Technical Trade-Offs

- **Pickle vs ONNX**: Pickle allows serializing complex scikit-learn composite pipelines with custom classes, but requires Python environments for deserialization.
- **In-Memory MemoryTools vs PostgreSQL**: JSON files require zero setup and are inspectable by humans, but do not support multi-process ACID transactions.
- **GridSearch vs Bayesian Optimization (Optuna)**: Standard scikit-learn search classes minimize dependencies and maximize stability on small-to-medium tabular datasets.

---

## 32. Current Limitations

- **In-Memory Data Constraint**: Datasets must fit comfortably in RAM; out-of-core streaming is not supported.
- **Single Model Caching in API**: The FastAPI backend serves the most recently trained model rather than maintaining a multi-model registry.
- **Pickle Security**: Deserializing model binaries requires trusting the model storage directory.
- **Stateless Agent Execution**: Agents process a dataset run-by-run without cross-session reinforcement or historical active learning.

---

## 33. Known Issues and Inconsistencies

- **Filename vs Class Discrepancy**: `src/tools/plot_tools.py` defines `MetricValidatorTool` (for metric checks) rather than plotting utilities. (Plotting is handled inside `EDAAgent`).
- **Dual Bus References**: `src/core/a2a_bus.py` contains the primary `A2ABus` class, while `src/tools/a2a_tools.py` provides a lightweight global bus instance (`A2A_GLOBAL_BUS`) utilized by the Streamlit A2A console.
- **Scikit-learn Deprecation Warnings**: Optimization warnings (`Unknown solver options: iprint` in LogisticRegression with certain solvers) occur occasionally during test runs but do not affect execution or accuracy.

---

## 34. Important Constants and Business Rules

- **IQR Outlier Multiplier**: `1.5` ($[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$).
- **Log Transformation Skew Threshold**: `1.0` (Applied to non-negative columns when skewness $> 1.0$).
- **Maximum Derived Features**: `15` (`max_created_features`).
- **Feature Selection Ratios**:
  - Total features $\le 5 \to$ Retain $100\%$.
  - Total features $\le 12 \to$ Retain $\max(4, \text{int}(N \times 0.85))$.
  - Total features $> 12 \to$ Retain $\max(8, \text{int}(N \times 0.70))$.
- **Verifier Quality Thresholds**:
  - Regression: $R^2 > 0.7 \to$ `Good`; $> 0.4 \to$ `Acceptable`; else `Weak`.
  - Classification: $F_1 > 0.75 \to$ `Good`; $> 0.5 \to$ `Acceptable`; else `Weak`.
  - Cross-Validation: $\text{std} \le 0.08 \to$ Stable generalization.

---

## 35. Complete Dependency Map

```text
Streamlit Dashboard (streamlit_app/)
    └── agents/ (ModelAgent, EDAAgent, ProfilerAgent, VerifierAgent, NotebookSynthesizerAgent)
    └── core/a2a_bus.py (A2ABus)
    └── tools/ (ModelTools, DatasetTools, MemoryTools, FeatureTools)

FastAPI Application (src/api/)
    ├── api/schemas.py (Pydantic models)
    ├── tools/model_tools.py (ModelTools, FullAutoMLPipeline)
    └── tools/memory_tools.py (MemoryTools)

Orchestrator CLI (src/orchestrator.py)
    ├── core/a2a_bus.py
    ├── agents/
    │    ├── profiler_agent.py -> tools/file_tools, tools/memory_tools
    │    ├── eda_agent.py      -> tools/file_tools, tools/memory_tools, matplotlib, seaborn
    │    ├── model_agent.py    -> tools/model_tools.py, core/a2a_bus.py
    │    │                         └── core/feature_engineering.py
    │    ├── verifier_agent.py -> tools/memory_tools.py, core/a2a_bus.py
    │    └── notebook_synthesizer_agent.py -> tools/notebook_tools.py, tools/file_tools.py
    │                                             └── google.generativeai, nbformat
    └── tools/memory_tools.py

MCP Architecture:
    mcp/manifest.json <──> src/tools/agent_tools.py (ToolRegistry)
```

---

## 36. Important Runtime Sequences

### 36.1 Full AutoML Training via REST API (`POST /train`)
1. Client sends JSON payload to `POST /train` specifying `target_col`, optional `csv_path` or inline `data`, and tuning options.
2. `train_model()` in `src/api/main.py` parses payload via `TrainRequest`.
3. Loads data from `csv_path` or inline dictionary records into a DataFrame.
4. Validates that `target_col` exists in DataFrame columns.
5. Calls `ModelTools.train(df, target_col, cv_folds, tune_hyperparameters, tuning_method)`.
6. Executes feature engineering, cross-validation, and hyperparameter tuning.
7. Saves the trained `FullAutoMLPipeline` to `models/<best_model>_model.pkl`.
8. Stores full execution results into `MemoryTools` under key `model_output`.
9. Returns structured `TrainResponse` with best model name, CV score, test metrics, and hyperparameter tuning details.

---

## 37. Interview Perspective

### 37.1 30-Second Explanation
"I built a Multi-Agent AutoML Data Analyst platform that automates the tabular data science lifecycle from profiling to deployment. Six collaborative agents communicate over a thread-safe event bus and execute tools registered via the Model Context Protocol. The system incorporates an automated feature engineering pipeline with strict zero-leakage safeguards, task-specific hyperparameter search, algorithmic quality verification, Gemini-synthesized reporting, and a production FastAPI REST backend."

### 37.2 2-Minute Technical Explanation
"The Multi-Agent AutoML platform addresses two fundamental challenges in automated machine learning: subtle data leakage and the disconnect between training pipelines and production inference.

Architecture-wise, the system uses a decoupled multi-agent paradigm. Specialized agents—Profiler, EDA, Model, Verifier, and Notebook Synthesizer—communicate over an in-memory `A2ABus`. Atomic capabilities are registered in an MCP-compliant `ToolRegistry`.

The core ML engine implements an end-to-end composite estimator, `FullAutoMLPipeline`. We fit all learned transformations—IQR outlier clipping, non-linear interaction features with zero-division protections, median imputers, standard scalers, one-hot encoders with unseen-category tolerance, and ANOVA feature selectors—strictly on training folds. Hyperparameter tuning evaluates candidate models (Random Forest, Logistic Regression, GBM, Ridge, SVR) using GridSearchCV or RandomizedSearchCV over cross-validation folds on $X_{\text{train}}$. The held-out test set ($X_{\text{test}}$) is never touched until final evaluation.

For serving, the winning pipeline is serialized and loaded by a FastAPI backend. Callers can post raw un-preprocessed JSON records directly to `/predict`; the composite pipeline handles all transformations internally and returns predictions in milliseconds. We also provide a multi-page Streamlit dashboard and Gemini 2.0 Flash integration for natural-language reporting."

### 37.3 Key Technical Questions an Interviewer Might Ask

1. **How do you guarantee that there is no data leakage during feature engineering and hyperparameter tuning?**
   *Key Answer Points*: The pipeline splits the dataset immediately into $X_{\text{train}}$ and $X_{\text{test}}$. All transformers (`OutlierHandler`, `FeatureCreationTransformer`, `ColumnTransformer`, `AutomatedFeatureSelector`) call `.fit()` exclusively on $X_{\text{train}}$. Hyperparameter optimization operates exclusively on training folds. The test set is only passed to `predict()` once at the very end.

2. **How does the system handle inference on raw data when new, unseen categorical values or missing fields arrive?**
   *Key Answer Points*: `FullAutoMLPipeline` encapsulates all transformations. Numeric features pass through `SimpleImputer(strategy='median')`. Categorical features pass through `OneHotEncoder(handle_unknown='ignore')`. Unseen categories produce all-zero indicator vectors rather than throwing exceptions. Missing values in derived ratios are handled safely without infinite values.

3. **Why did you decouple the system using an Agent-to-Agent message bus instead of direct function calls?**
   *Key Answer Points*: Decoupling allows agents to operate autonomously, enables audit logging of every analytical milestone, supports asynchronous multi-step workflows, and allows individual agents (like the Verifier or Notebook Synthesizer) to be swapped, enhanced, or distributed without modifying core ML code.

4. **What are the primary performance bottlenecks, and how are they mitigated?**
   *Key Answer Points*: Combinatorial explosion in pairwise feature creation is mitigated by ranking features by variance and capping to 15 created features. Hyperparameter search latency is mitigated by using focused search spaces and automatically falling back to `RandomizedSearchCV` when candidate counts exceed 8.

---

## 38. Project Knowledge Checklist for Consuming AI Models

Any AI reasoning about this repository using `CONTEXT.md` should be able to answer:
- [x] **What is the repository name and purpose?** Multi-Agent AutoML Data Analyst; automated tabular EDA, feature engineering, tuning, and API serving.
- [x] **What agents exist?** ProfilerAgent, EDAAgent, ModelAgent, VerifierAgent, NotebookSynthesizerAgent, MemoryAgent.
- [x] **How do agents communicate?** Via `A2ABus` using topic-based publish/subscribe routing, thread locks, and memory persistence.
- [x] **What is the MCP tool registry?** `ToolRegistry` exposing `file`, `dataset`, `model`, `notebook`, `job`, `memory`, `feature`.
- [x] **What ML models are supported?** RF, LogReg, GBM (classification); RF, Ridge, SVR (regression).
- [x] **How is hyperparameter tuning performed?** GridSearchCV or RandomizedSearchCV over StratifiedKFold/KFold CV on $X_{\text{train}}$.
- [x] **How is inference served?** Via FastAPI endpoints (`POST /predict`, `GET /health`, `POST /train`, `GET /model-info`) and Streamlit.
- [x] **How is data leakage prevented?** Preprocessing, outlier clipping, feature creation, and feature selection are fitted exclusively on training splits/folds.
- [x] **How is Gemini used?** Embedded in `NotebookTools` and Streamlit using `gemini-2.0-flash` to generate insights for the Jupyter report.
- [x] **What is the testing status?** 32 tests passing across 5 test files in `tests/`.

---

## 39. Source-of-Truth Rules

1. **Repository Code is Absolute Truth**: Code implementation takes precedence over comments or documentation if discrepancies exist.
2. **No Hallucinated Features**: Components not found in the codebase (such as vector databases, RAG chunkers, or Dockerfiles) are explicitly reported as absent.
3. **Strict Secret Hygiene**: Real API keys, tokens, or credentials are never included in code or documentation; safe placeholders are used.
4. **Architectural Distinctions**: Explicitly distinguishes between deterministic ML algorithms, rule-based heuristics, and optional LLM generation.

---

## 40. Final Project Summary

The **Multi-Agent AutoML Data Analyst** is a modular, production-tested data science platform. It bridges the gap between automated data exploration and production-ready machine learning by combining a collaborative multi-agent architecture with a leak-free scikit-learn AutoML pipeline. 

By structuring the workflow into specialized agents (Profiler, EDA, Model, Verifier, Notebook Synthesizer) coordinated over a thread-safe message bus and exposing atomic capabilities via an MCP tool registry, the platform guarantees analytical rigor, modularity, and transparency. 

Its automated feature engineering layer handles outlier clipping, interaction generation, missing-value imputation, categorical encoding, and feature selection strictly on training data. Hyperparameters are tuned over cross-validation folds, and the winning model is packaged into a self-contained `FullAutoMLPipeline` artifact. 

With dual user surfaces—an interactive multi-page Streamlit analytical studio and a high-performance FastAPI REST API—alongside 32 passing tests, the project serves as a comprehensive reference implementation for automated tabular data science and multi-agent system design.
