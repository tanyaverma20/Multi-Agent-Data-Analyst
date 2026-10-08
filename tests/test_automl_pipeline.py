# tests/test_automl_pipeline.py
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

from tools.model_tools import ModelTools, FullAutoMLPipeline


@pytest.fixture
def model_tools():
    return ModelTools(output_dir="models")


def test_classification_end_to_end(model_tools):
    """Test full classification pipeline with numeric + categorical columns."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "age": np.random.randint(20, 65, n),
        "income": np.random.normal(50000, 15000, n),
        "department": np.random.choice(["Sales", "Engineering", "Marketing"], n),
        "satisfaction_score": np.random.uniform(1, 10, n),
        "churn": np.random.choice([0, 1], n),
    })

    result = model_tools.train(df, target_col="churn", cv_folds=3)

    assert result["status"] == "success"
    assert result["task_type"] == "classification"
    assert "best_model" in result
    assert os.path.exists(result["model_path"])

    # Verify metrics
    metrics = result["metrics"]
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "f1" in metrics  # backward compatibility

    # Verify cross validation
    cv = result["cross_validation"]
    assert cv["folds"] >= 2
    assert "rf" in cv["models"]
    assert "mean_cv_score" in cv["models"]["rf"]
    assert "std_cv_score" in cv["models"]["rf"]

    # Verify feature importance
    fi = result["feature_importance"]
    assert len(fi) > 0
    assert "rank" in fi[0]
    assert "feature" in fi[0]
    assert "importance" in fi[0]

    # Verify feature engineering summary
    fe = result["feature_engineering"]
    assert "numeric_features" in fe
    assert "categorical_features" in fe
    assert "transformations_applied" in fe


def test_regression_end_to_end(model_tools):
    """Test full regression pipeline with numeric columns."""
    np.random.seed(42)
    n = 70
    x1 = np.random.normal(10, 2, n)
    x2 = np.random.normal(20, 5, n)
    target = 2.5 * x1 + 1.2 * x2 + np.random.normal(0, 1, n)

    df = pd.DataFrame({
        "feature1": x1,
        "feature2": x2,
        "price": target,
    })

    result = model_tools.train(df, target_col="price", cv_folds=3)

    assert result["status"] == "success"
    assert result["task_type"] == "regression"
    assert "r2" in result["metrics"]
    assert "mse" in result["metrics"]
    assert "rmse" in result["metrics"]
    assert "mae" in result["metrics"]

    # Cross validation must record R2 scores
    assert result["cross_validation"]["cv_metric"] == "r2"


def test_missing_values_handling(model_tools):
    """Verify dataset with missing values in both numeric and categorical columns trains without error."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "num_with_nan": [np.nan if i % 5 == 0 else float(i) for i in range(n)],
        "cat_with_nan": [None if i % 7 == 0 else f"cat_{i % 3}" for i in range(n)],
        "normal_num": np.random.randn(n),
        "target": np.random.choice([0, 1], n),
    })

    result = model_tools.train(df, target_col="target", cv_folds=3)
    assert result["status"] == "success"
    assert "metrics" in result


def test_unseen_categorical_values(model_tools):
    """Verify saved pipeline safely predicts on unseen categorical levels."""
    train_df = pd.DataFrame({
        "city": ["New York", "London", "Paris", "Tokyo"] * 10,
        "age": [25, 30, 35, 40] * 10,
        "salary": [50000, 60000, 70000, 80000] * 10,
        "approved": [0, 1, 0, 1] * 10,
    })

    result = model_tools.train(train_df, target_col="approved", cv_folds=2)
    assert result["status"] == "success"

    import pickle
    with open(result["model_path"], "rb") as f:
        loaded_pipeline = pickle.load(f)

    # Test data contains completely unseen city 'Berlin' and 'Sydney'
    test_df = pd.DataFrame({
        "city": ["Berlin", "Sydney", "New York"],
        "age": [28, 33, 45],
        "salary": [55000, 65000, 75000],
    })

    preds = loaded_pipeline.predict(test_df)
    assert len(preds) == 3


def test_outlier_handling_in_pipeline(model_tools):
    """Verify extreme outliers are captured and reported in the pipeline output."""
    np.random.seed(42)
    n = 60
    vals = list(np.random.normal(100, 10, n))
    vals[0] = 9999.0  # extreme outlier
    vals[1] = -8888.0  # extreme outlier

    df = pd.DataFrame({
        "metric": vals,
        "target": np.random.choice([0, 1], n),
    })

    result = model_tools.train(df, target_col="target", cv_folds=3, outlier_strategy="clip")
    assert result["status"] == "success"
    outliers = result["outliers"]
    assert outliers["total_outliers"] > 0
    assert "metric" in outliers["affected_columns"]
    assert outliers["handling_strategy"] == "clip"


def test_constant_column_dataset(model_tools):
    """Verify dataset with a constant column is handled without crashing."""
    n = 40
    df = pd.DataFrame({
        "constant_col": [42.0] * n,
        "varying_col": list(range(n)),
        "target": [0, 1] * (n // 2),
    })

    result = model_tools.train(df, target_col="target", cv_folds=2)
    assert result["status"] == "success"


def test_small_dataset(model_tools):
    """Verify small dataset (e.g. 20 samples) trains successfully with adaptive folds."""
    n = 20
    df = pd.DataFrame({
        "f1": list(range(n)),
        "f2": [x * 2 for x in range(n)],
        "target": [0, 1] * (n // 2),
    })

    result = model_tools.train(df, target_col="target", cv_folds=3)
    assert result["status"] == "success"
    assert result["cross_validation"]["folds"] >= 2


def test_feature_importance_mapping(model_tools):
    """Verify feature names in feature importance match readable feature names."""
    df = pd.DataFrame({
        "feature_alpha": np.random.randn(50),
        "feature_beta": np.random.randn(50),
        "target": np.random.choice([0, 1], 50),
    })

    result = model_tools.train(df, target_col="target", cv_folds=2)
    assert result["status"] == "success"
    fi = result["feature_importance"]
    assert len(fi) > 0
    fi_names = [item["feature"] for item in fi]
    # Check that feature names do not have ugly prefixes like num__
    for name in fi_names:
        assert not name.startswith("num__")
        assert not name.startswith("cat__")
