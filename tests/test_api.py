# tests/test_api.py
import pytest
import os
import sys
import pandas as pd
from fastapi.testclient import TestClient

# Ensure src is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from api.main import app
from tools.model_tools import ModelTools
from tools.memory_tools import MemoryTools


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def prepare_test_model():
    """Ensure a trained model exists for test client inference."""
    df = pd.DataFrame({
        "age": [25, 38, 45, 22, 60, 33, 29, 52, 41, 19] * 4,
        "income": [45000, 82000, 120000, 28000, 95000, 54000, 62000, 110000, 75000, 18000] * 4,
        "employment_type": ["employed", "employed", "self_employed", "unemployed", "retired"] * 8,
        "default": [0, 0, 0, 1, 0, 1, 0, 0, 0, 1] * 4,
    })
    model_tools = ModelTools(output_dir="models")
    result = model_tools.train(df, target_col="default", cv_folds=2, tune_hyperparameters=False)
    MemoryTools().save("model_output", result)
    yield
    # Cleanup memory if needed


def test_get_health(client):
    """Test GET /health returns 200 and healthy payload."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "multi-agent-automl-data-analyst"


def test_get_model_info(client):
    """Test GET /model-info returns metadata for the active model."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "model_name" in data
    assert "task_type" in data
    assert "feature_count" in data
    assert "test_metrics" in data


def test_post_predict_single_record(client):
    """Test POST /predict with a single feature dictionary."""
    payload = {
        "data": {
            "age": 30,
            "income": 50000,
            "employment_type": "employed"
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "predictions" in data
    assert len(data["predictions"]) == 1


def test_post_predict_batch_records(client):
    """Test POST /predict with a list of feature dictionaries."""
    payload = {
        "data": [
            {"age": 25, "income": 30000, "employment_type": "unemployed"},
            {"age": 55, "income": 90000, "employment_type": "retired"},
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["predictions"]) == 2


def test_post_predict_invalid_data(client):
    """Test POST /predict with empty list returns 400 Bad Request."""
    payload = {"data": []}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "detail" in response.json()


def test_post_train_inline_data(client):
    """Test POST /train triggers AutoML training on inline records."""
    training_data = [
        {"feature_a": 1.0, "feature_b": 10.0, "target": 0},
        {"feature_a": 2.0, "feature_b": 20.0, "target": 1},
        {"feature_a": 3.0, "feature_b": 30.0, "target": 0},
        {"feature_a": 4.0, "feature_b": 40.0, "target": 1},
        {"feature_a": 5.0, "feature_b": 50.0, "target": 0},
        {"feature_a": 6.0, "feature_b": 60.0, "target": 1},
        {"feature_a": 7.0, "feature_b": 70.0, "target": 0},
        {"feature_a": 8.0, "feature_b": 80.0, "target": 1},
    ] * 3

    payload = {
        "target_col": "target",
        "data": training_data,
        "cv_folds": 2,
        "tune_hyperparameters": True,
        "tuning_method": "grid",
    }
    response = client.post("/train", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "best_model" in data
    assert "test_metrics" in data
    assert "hyperparameter_tuning" in data
    assert data["hyperparameter_tuning"]["enabled"] is True
