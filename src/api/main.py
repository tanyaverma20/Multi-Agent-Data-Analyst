# src/api/main.py
import os
import sys
import pickle
import pandas as pd
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure src is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == "src" else BASE_DIR
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for p in [PROJECT_ROOT, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    TrainRequest,
    TrainResponse,
    ModelInfoResponse,
)
from tools.model_tools import ModelTools, FullAutoMLPipeline
from tools.memory_tools import MemoryTools

app = FastAPI(
    title="Multi-Agent AutoML Data Analyst API",
    description="Production REST API for automated feature engineering, hyperparameter tuning, model training, and inference.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_latest_model() -> Optional[FullAutoMLPipeline]:
    """Helper to locate and load the latest trained FullAutoMLPipeline artifact."""
    memory = MemoryTools()
    model_output = memory.load("model_output")

    model_path = None
    if isinstance(model_output, dict) and "model_path" in model_output:
        cand_path = model_output["model_path"]
        if os.path.exists(cand_path):
            model_path = cand_path

    # Fallback to scanning models directory
    if not model_path:
        models_dir = os.path.join(PROJECT_ROOT, "models")
        if os.path.exists(models_dir):
            pkl_files = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith(".pkl")]
            if pkl_files:
                # pick most recently modified
                pkl_files.sort(key=os.path.getmtime, reverse=True)
                model_path = pkl_files[0]

    if model_path and os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception:
            return None
    return None


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Service health and liveness probe."""
    return HealthResponse(
        status="healthy",
        service="multi-agent-automl-data-analyst",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(request: PredictionRequest):
    """
    Generate predictions on raw feature data using the trained FullAutoMLPipeline.
    Automatic preprocessing, scaling, encoding, outlier clipping, and feature creation
    are executed internally by the pipeline.
    """
    model = _load_latest_model()
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No trained model found. Please train a model via /train first.",
        )

    # Convert request data to DataFrame
    try:
        if isinstance(request.data, dict):
            df = pd.DataFrame([request.data])
        elif isinstance(request.data, list):
            if len(request.data) == 0:
                raise ValueError("Prediction data list cannot be empty.")
            df = pd.DataFrame(request.data)
        else:
            raise ValueError("Input data must be a dictionary or list of dictionaries.")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid prediction input format: {str(e)}",
        )

    try:
        raw_predictions = model.predict(df)
        predictions_list = (
            raw_predictions.tolist()
            if hasattr(raw_predictions, "tolist")
            else list(raw_predictions)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )

    # Retrieve model metadata if available
    memory = MemoryTools()
    model_output = memory.load("model_output")
    model_name = "automl_model"
    task_type = "unknown"
    if isinstance(model_output, dict):
        model_name = model_output.get("best_model", model_output.get("model_name", "automl_model"))
        task_type = model_output.get("task_type", "unknown")

    return PredictionResponse(
        status="success",
        model_name=model_name,
        task_type=task_type,
        predictions=predictions_list,
    )


@app.post("/train", response_model=TrainResponse, tags=["Training"])
def train_model(request: TrainRequest):
    """
    Trigger AutoML training with feature engineering, cross-validation,
    and hyperparameter tuning.
    """
    df = None
    if request.data:
        try:
            df = pd.DataFrame(request.data)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not parse inline training data: {str(e)}",
            )
    elif request.csv_path:
        if not os.path.exists(request.csv_path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Specified csv_path does not exist: {request.csv_path}",
            )
        try:
            df = pd.read_csv(request.csv_path)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to load CSV file: {str(e)}",
            )
    else:
        # Check default sample.csv
        sample_path = os.path.join(PROJECT_ROOT, "sample.csv")
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No training data provided and default sample.csv not found.",
            )

    if request.target_col not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target column '{request.target_col}' not found in dataset.",
        )

    model_tools = ModelTools()
    result = model_tools.train(
        df=df,
        target_col=request.target_col,
        cv_folds=request.cv_folds,
        tune_hyperparameters=request.tune_hyperparameters,
        tuning_method=request.tuning_method,
    )

    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Training failed: {result.get('error')}",
        )

    # Save to memory
    MemoryTools().save("model_output", result)

    best_model = result.get("best_model", "unknown")
    best_cv_score = 0.0
    cv_info = result.get("cross_validation", {}).get("models", {}).get(best_model, {})
    if cv_info:
        best_cv_score = cv_info.get("mean_cv_score", 0.0)

    return TrainResponse(
        status="success",
        task_type=result.get("task_type", "unknown"),
        best_model=best_model,
        best_cv_score=best_cv_score,
        test_metrics=result.get("test_metrics", {}),
        hyperparameter_tuning=result.get("hyperparameter_tuning"),
        message="Model trained and hyperparameter-tuned successfully",
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Info"])
def get_model_info():
    """Retrieve metadata, parameters, and evaluation metrics for the active model."""
    memory = MemoryTools()
    model_output = memory.load("model_output")

    if not isinstance(model_output, dict) or model_output.get("status") != "success":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No trained model found. Please train a model first.",
        )

    best_name = model_output.get("best_model", model_output.get("model_name", "automl_model"))
    cv_info = model_output.get("cross_validation", {})
    best_cv_score = None
    if "models" in cv_info and best_name in cv_info["models"]:
        best_cv_score = cv_info["models"][best_name].get("mean_cv_score")

    fs = model_output.get("feature_selection", {})

    return ModelInfoResponse(
        status="success",
        model_name=best_name,
        task_type=model_output.get("task_type", "unknown"),
        model_path=model_output.get("model_path", ""),
        feature_count=fs.get("original_count", 0),
        selected_feature_count=fs.get("selected_count", 0),
        selected_features=fs.get("selected_features", []),
        cv_metric=cv_info.get("cv_metric", "score"),
        cv_score=best_cv_score,
        test_metrics=model_output.get("test_metrics", {}),
        hyperparameter_tuning=model_output.get("hyperparameter_tuning"),
    )
