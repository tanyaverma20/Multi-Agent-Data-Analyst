# src/api/schemas.py
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Service health status")
    service: str = Field(default="multi-agent-automl-data-analyst", description="Service name")


class PredictionRequest(BaseModel):
    data: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(
        ...,
        description="Raw feature record (dict) or list of feature records to predict on."
    )


class PredictionResponse(BaseModel):
    status: str = Field(default="success")
    model_name: str = Field(..., description="Name of model used for prediction")
    task_type: str = Field(..., description="Classification or regression")
    predictions: List[Any] = Field(..., description="Model predictions")


class TrainRequest(BaseModel):
    target_col: str = Field(..., description="Column name to predict")
    data: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional inline dataset rows as list of dicts."
    )
    csv_path: Optional[str] = Field(
        default=None,
        description="Optional file path to CSV dataset on disk."
    )
    cv_folds: int = Field(default=5, ge=2, le=10, description="Cross-validation fold count")
    tune_hyperparameters: bool = Field(default=True, description="Whether to tune hyperparameters")
    tuning_method: str = Field(default="grid", description="Tuning method: 'grid' or 'random'")


class TrainResponse(BaseModel):
    status: str = Field(default="success")
    task_type: str = Field(...)
    best_model: str = Field(...)
    best_cv_score: float = Field(...)
    test_metrics: Dict[str, float] = Field(default_factory=dict)
    hyperparameter_tuning: Optional[Dict[str, Any]] = None
    message: str = Field(default="Model trained successfully")


class ModelInfoResponse(BaseModel):
    status: str = Field(default="success")
    model_name: str = Field(...)
    task_type: str = Field(...)
    model_path: str = Field(...)
    feature_count: int = Field(default=0)
    selected_feature_count: int = Field(default=0)
    selected_features: List[str] = Field(default_factory=list)
    cv_metric: str = Field(...)
    cv_score: Optional[float] = None
    test_metrics: Dict[str, float] = Field(default_factory=dict)
    hyperparameter_tuning: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
