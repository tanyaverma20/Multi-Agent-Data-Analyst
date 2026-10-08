# tests/test_feature_engineering.py
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

from core.feature_engineering import (
    OutlierHandler,
    FeatureCreationTransformer,
    AutomatedFeatureSelector,
)
from tools.feature_tools import FeatureTools


def test_outlier_detection_and_clipping():
    """Verify IQR outlier detection, bound calculation, and clipping strategy."""
    np.random.seed(42)
    # 50 normal observations and 2 extreme outliers
    data = {
        "val": list(np.random.normal(50, 5, 50)) + [1000.0, -500.0],
        "category": ["A"] * 26 + ["B"] * 26,
    }
    df = pd.DataFrame(data)

    handler = OutlierHandler(method="iqr", strategy="clip", factor=1.5)
    handler.fit(df)
    summary = handler.get_summary()

    assert summary["total_outliers"] >= 2
    assert "val" in summary["affected_columns"]
    assert summary["affected_columns"]["val"]["lower_bound"] < 50
    assert summary["affected_columns"]["val"]["upper_bound"] > 50

    # Test transform: outliers must be clipped to learned bounds
    df_transformed = handler.transform(df)
    upper = summary["affected_columns"]["val"]["upper_bound"]
    lower = summary["affected_columns"]["val"]["lower_bound"]

    assert df_transformed["val"].max() <= upper + 1e-5
    assert df_transformed["val"].min() >= lower - 1e-5


def test_outlier_handler_no_data_leakage():
    """Verify outlier bounds are learned strictly on training data and not modified by test data."""
    train_df = pd.DataFrame({"x": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]})
    test_df = pd.DataFrame({"x": [100.0, 200.0]})

    handler = OutlierHandler(method="iqr", strategy="clip")
    handler.fit(train_df)
    initial_upper = handler.bounds_["x"][1]

    # Transform test set
    test_transformed = handler.transform(test_df)

    # Bounds must remain identical to train bounds
    assert handler.bounds_["x"][1] == initial_upper
    assert (test_transformed["x"] <= initial_upper).all()


def test_feature_creation_transformer():
    """Verify feature creation: ratios, differences, interactions, and skewness."""
    df = pd.DataFrame({
        "a": [10.0, 20.0, 30.0, 40.0, 50.0],
        "b": [2.0, 4.0, 5.0, 8.0, 10.0],
        "skewed": [1.0, 2.0, 3.0, 10.0, 100.0],  # Right-skewed
    })

    transformer = FeatureCreationTransformer(
        create_ratios=True,
        create_differences=True,
        create_interactions=True,
        transform_skewed=True,
        max_created_features=10,
    )
    transformer.fit(df)
    df_out = transformer.transform(df)

    summary = transformer.get_summary()
    assert summary["total_created"] > 0
    # Must have created features
    created_cols = [c for c in df_out.columns if "ratio_" in c or "diff_" in c or "inter_" in c]
    assert len(created_cols) > 0

    # Verify no infinite or NaN values created
    for col in created_cols:
        assert not np.isinf(df_out[col]).any()


def test_feature_creation_safeguards_division_by_zero():
    """Verify ratio creation safely handles zeroes and avoids infinite values."""
    df = pd.DataFrame({
        "num": [10.0, 20.0, 30.0],
        "denom_with_zero": [0.0, 5.0, 0.0],
    })

    transformer = FeatureCreationTransformer(create_ratios=True, max_created_features=5)
    transformer.fit(df)
    df_out = transformer.transform(df)

    # Check that any generated ratio is not infinite
    for col in df_out.columns:
        assert not np.isinf(df_out[col].fillna(0)).any()


def test_feature_creation_constant_column():
    """Verify constant columns are handled safely without crash."""
    df = pd.DataFrame({
        "constant": [1.0, 1.0, 1.0, 1.0],
        "variable": [2.0, 4.0, 6.0, 8.0],
    })

    transformer = FeatureCreationTransformer(max_created_features=5)
    transformer.fit(df)
    df_out = transformer.transform(df)

    assert df_out.shape[0] == 4


def test_automated_feature_selector_classification():
    """Verify feature selection on classification tasks."""
    X = pd.DataFrame({
        f"feat_{i}": np.random.randn(50) for i in range(15)
    })
    # Make feat_0 informative for y
    y = pd.Series((X["feat_0"] > 0).astype(int))

    selector = AutomatedFeatureSelector(task_type="classification", method="k_best", k=5)
    selector.fit(X, y)
    X_sel = selector.transform(X)

    summary = selector.get_summary()
    assert summary["original_count"] == 15
    assert summary["selected_count"] == 5
    assert len(summary["selected_features"]) == 5
    assert X_sel.shape[1] == 5


def test_automated_feature_selector_regression():
    """Verify feature selection on regression tasks."""
    X = pd.DataFrame({
        f"feat_{i}": np.random.randn(50) for i in range(15)
    })
    y = pd.Series(X["feat_0"] * 3.5 + X["feat_1"] * 2.0 + np.random.randn(50) * 0.1)

    selector = AutomatedFeatureSelector(task_type="regression", method="k_best", k=4)
    selector.fit(X, y)
    X_sel = selector.transform(X)

    summary = selector.get_summary()
    assert summary["selected_count"] == 4
    assert X_sel.shape[1] == 4


def test_feature_tools_mcp_standalone():
    """Verify FeatureTools MCP methods work as standalone calls."""
    ft = FeatureTools()
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 50.0],
        "num2": [10.0, 20.0, 30.0, 40.0, 50.0],
        "cat": ["a", "b", "a", "b", "a"],
        "target": [0, 1, 0, 1, 0],
    })

    outlier_res = ft.detect_outliers(df)
    assert outlier_res["status"] == "success"
    assert "outliers" in outlier_res

    eng_res = ft.engineer_features(df, target_col="target")
    assert eng_res["status"] == "success"
    assert "created_features" in eng_res

    sel_res = ft.select_features(df, target_col="target", k=2)
    assert sel_res["status"] == "success"
    assert "selection_summary" in sel_res
