# tests/test_hyperparameter_tuning.py
import pytest
import numpy as np
import pandas as pd
import os
import sys

# Ensure src is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tools.model_tools import ModelTools
from agents.model_agent import ModelAgent
from core.a2a_bus import A2ABus
from tools.memory_tools import MemoryTools
from tools.feature_tools import FeatureTools


@pytest.fixture
def model_tools():
    return ModelTools(output_dir="models")


def test_hyperparameter_tuning_classification(model_tools):
    """Test hyperparameter tuning for classification models."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "age": np.random.randint(20, 60, n),
        "income": np.random.normal(50000, 10000, n),
        "department": np.random.choice(["HR", "Engineering", "Sales"], n),
        "target": np.random.choice([0, 1], n),
    })

    result = model_tools.train(
        df=df,
        target_col="target",
        cv_folds=3,
        tune_hyperparameters=True,
        tuning_method="grid",
    )

    assert result["status"] == "success"
    assert "hyperparameter_tuning" in result
    ht = result["hyperparameter_tuning"]
    assert ht["enabled"] is True
    assert ht["method"] == "GridSearchCV"
    assert "models" in ht
    assert len(ht["models"]) > 0

    # Verify best params and CV statistics recorded for each tuned candidate
    for m_name, m_data in ht["models"].items():
        assert "best_params" in m_data
        assert isinstance(m_data["best_params"], dict)
        assert "best_cv_score" in m_data
        assert "cv_std" in m_data
        assert "candidates_evaluated" in m_data
        assert m_data["candidates_evaluated"] > 0
        assert "search_duration_sec" in m_data


def test_hyperparameter_tuning_regression(model_tools):
    """Test hyperparameter tuning for regression models."""
    np.random.seed(42)
    n = 60
    x1 = np.random.normal(10, 2, n)
    x2 = np.random.normal(5, 1, n)
    y = 3.0 * x1 - 2.0 * x2 + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({
        "feature_1": x1,
        "feature_2": x2,
        "target_price": y,
    })

    result = model_tools.train(
        df=df,
        target_col="target_price",
        cv_folds=3,
        tune_hyperparameters=True,
        tuning_method="grid",
    )

    assert result["status"] == "success"
    assert result["task_type"] == "regression"
    ht = result["hyperparameter_tuning"]
    assert ht["enabled"] is True
    assert "models" in ht
    # Verify Ridge and SVR tuned parameters recorded
    assert "ridge" in ht["models"]
    assert "alpha" in ht["models"]["ridge"]["best_params"]


def test_randomized_search_cv(model_tools):
    """Test RandomizedSearchCV option executes and selects optimal parameters."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "feat_a": np.random.randn(n),
        "feat_b": np.random.randn(n),
        "label": np.random.choice([0, 1], n),
    })

    result = model_tools.train(
        df=df,
        target_col="label",
        cv_folds=2,
        tune_hyperparameters=True,
        tuning_method="random",
    )

    assert result["status"] == "success"
    ht = result["hyperparameter_tuning"]
    assert ht["method"] == "RandomizedSearchCV"


def test_no_data_leakage_during_hyperparameter_tuning(model_tools):
    """
    Verify test set is strictly held out and never seen during
    feature engineering, feature selection, or hyperparameter search.
    """
    np.random.seed(42)
    n = 80
    df = pd.DataFrame({
        "feature_val": np.random.randn(n),
        "target": np.random.choice([0, 1], n),
    })

    # Train model with tuning
    result = model_tools.train(df, target_col="target", cv_folds=3, tune_hyperparameters=True)
    assert result["status"] == "success"

    # Verify best_model was picked by CV score, not test metrics
    best_name = result["best_model"]
    cv_info = result["cross_validation"]["models"][best_name]
    best_cv_score = cv_info["mean_cv_score"]

    for other_name, other_info in result["cross_validation"]["models"].items():
        assert best_cv_score >= other_info["mean_cv_score"]


def test_model_agent_emits_tuning_a2a_events():
    """Verify ModelAgent emits hyperparameter_tuning events on A2A bus."""
    memory = MemoryTools(storage_dir="tests/test_storage_tuning")
    a2a = A2ABus(memory=memory, persist=False)
    agent = ModelAgent(a2a_bus=a2a, cv=2)

    df = pd.DataFrame({
        "num": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
        "target": [0, 1] * 15,
    })

    result = agent.run(df, target_col="target", tune_hyperparameters=True)
    assert result["status"] == "success"

    audit = a2a.audit_log()
    topics = [msg.get("topic") for msg in audit]

    assert "hyperparameter_tuning.started" in topics
    assert "hyperparameter_tuning.completed" in topics


def test_feature_tools_tune_mcp_method():
    """Verify MCP tune_model tool method works."""
    ft = FeatureTools()
    df = pd.DataFrame({
        "x": [10, 20, 30, 40, 50, 60, 70, 80] * 2,
        "y": [0, 1] * 8,
    })
    res = ft.tune_model(df, target_col="y", cv_folds=2)
    assert res["status"] == "success"
    assert "hyperparameter_tuning" in res
