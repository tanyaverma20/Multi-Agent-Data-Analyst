# tests/test_agents_integration.py
import pytest
import numpy as np
import pandas as pd
import os
import sys

# Ensure src is on path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.a2a_bus import A2ABus
from tools.memory_tools import MemoryTools
from tools.agent_tools import ToolRegistry
from agents.model_agent import ModelAgent
from agents.verifier_agent import VerifierAgent
from agents.notebook_synthesizer_agent import NotebookSynthesizerAgent
from agents.profiler_agent import ProfilerAgent
from agents.eda_agent import EDAAgent


def test_tool_registry_includes_feature_tool():
    """Verify ToolRegistry contains the feature tool."""
    reg = ToolRegistry()
    tools = reg.list_tools()
    assert "feature" in tools
    assert "model" in tools
    assert "file" in tools
    assert "dataset" in tools
    assert "memory" in tools


def test_model_agent_a2a_lifecycle_events():
    """Verify ModelAgent runs and emits all lifecycle events on A2A bus."""
    memory = MemoryTools(storage_dir="tests/test_storage")
    a2a = A2ABus(memory=memory, persist=False)

    model_agent = ModelAgent(a2a_bus=a2a, cv=2)

    df = pd.DataFrame({
        "x1": np.random.randn(40),
        "x2": np.random.randn(40),
        "target": np.random.choice([0, 1], 40),
    })

    result = model_agent.run(df, target_col="target")
    assert result["status"] == "success"

    # Audit log must contain the expected lifecycle events
    audit = a2a.audit_log()
    topics = [msg.get("topic") for msg in audit]

    assert "feature_engineering.started" in topics
    assert "feature_engineering.completed" in topics
    assert "cross_validation.completed" in topics
    assert "feature_selection.completed" in topics
    assert "feature_importance.completed" in topics
    assert "model.trained" in topics


def test_verifier_agent_evaluates_automl_output():
    """Verify VerifierAgent accepts upgraded AutoML output and produces quality rating."""
    memory = MemoryTools(storage_dir="tests/test_storage")
    a2a = A2ABus(memory=memory, persist=False)
    verifier = VerifierAgent(a2a_bus=a2a)

    mock_model_output = {
        "status": "success",
        "task_type": "classification",
        "best_model": "rf",
        "metrics": {"accuracy": 0.88, "f1_score": 0.87, "f1": 0.87},
        "cross_validation": {
            "folds": 5,
            "models": {
                "rf": {"mean_cv_score": 0.86, "std_cv_score": 0.03}
            }
        },
        "outliers": {"total_outliers": 5},
        "feature_selection": {"selected_count": 8, "original_count": 12},
    }

    res = verifier.run(mock_model_output)
    assert res["status"] == "success"
    assert res["quality"] in ["Good", "Acceptable", "Weak"]
    assert any("CV variance is low" in note for note in res["notes"])
    assert any("Outliers safely handled" in note for note in res["notes"])


def test_notebook_synthesizer_with_rich_automl():
    """Verify NotebookSynthesizerAgent generates notebook containing new sections."""
    nb_agent = NotebookSynthesizerAgent()

    profiler_out = {"status": "success", "rows": 50, "cols": 4}
    eda_out = {"status": "success", "numeric_columns": ["x1", "x2"]}
    model_out = {
        "status": "success",
        "task_type": "classification",
        "best_model": "rf",
        "metrics": {"accuracy": 0.9, "f1": 0.9},
        "feature_engineering": {
            "created_features": ["ratio_x1_x2"],
            "transformed_features": [],
            "transformations_applied": ["IQR Outlier Clipping", "Ratios"],
        },
        "outliers": {"method": "IQR", "total_outliers": 2, "percentage_affected": 4.0},
        "feature_selection": {"original_count": 5, "selected_count": 4, "method": "k_best"},
        "cross_validation": {
            "folds": 5,
            "models": {
                "rf": {"mean_cv_score": 0.89, "std_cv_score": 0.02, "fold_scores": [0.88, 0.9, 0.89]}
            }
        },
        "feature_importance": [
            {"rank": 1, "feature": "x1", "importance": 0.6, "direction": "positive"}
        ],
    }
    verifier_out = {"status": "success", "quality": "Good"}

    nb_res = nb_agent.run(profiler_out, eda_out, model_out, verifier_out)
    assert nb_res["status"] == "success"
    assert os.path.exists(nb_res["notebook_path"])
