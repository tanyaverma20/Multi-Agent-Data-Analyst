# src/tools/feature_tools.py
"""
MCP-compliant Feature Engineering Tools.
Exposes feature engineering, outlier detection, and feature selection as standalone tools.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union

from core.feature_engineering import (
    OutlierHandler,
    FeatureCreationTransformer,
    AutomatedFeatureSelector,
)


class FeatureTools:
    """
    Standalone toolset for feature engineering operations.
    """

    def detect_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
        strategy: str = "clip",
        factor: float = 1.5,
    ) -> Dict[str, Any]:
        """
        Detect outliers in dataframe numerical columns.
        """
        try:
            handler = OutlierHandler(method=method, strategy=strategy, factor=factor)
            handler.fit(df)
            summary = handler.get_summary()
            return {"status": "success", "outliers": summary}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def engineer_features(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        max_created_features: int = 15,
        create_ratios: bool = True,
        create_differences: bool = True,
        create_interactions: bool = True,
    ) -> Dict[str, Any]:
        """
        Create derived features and apply mathematical transformations.
        """
        try:
            X = df.drop(columns=[target_col]) if target_col and target_col in df.columns else df.copy()
            y = df[target_col] if target_col and target_col in df.columns else None

            creator = FeatureCreationTransformer(
                create_ratios=create_ratios,
                create_differences=create_differences,
                create_interactions=create_interactions,
                transform_skewed=True,
                max_created_features=max_created_features,
            )
            creator.fit(X, y)
            X_engineered = creator.transform(X)
            summary = creator.get_summary()

            return {
                "status": "success",
                "original_shape": list(X.shape),
                "engineered_shape": list(X_engineered.shape),
                "created_features": summary["created_features"],
                "transformed_features": summary["transformed_features"],
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def select_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        method: str = "k_best",
        k: Union[int, str] = "auto",
    ) -> Dict[str, Any]:
        """
        Select top features for a given target column.
        """
        try:
            if target_col not in df.columns:
                return {"status": "error", "error": f"Target column '{target_col}' not found."}

            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Detect task
            task_type = "regression" if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20 else "classification"

            # Preprocess numeric for selector
            numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
            if not numeric_cols:
                return {"status": "error", "error": "No numeric features available for selection."}

            X_num = X[numeric_cols].fillna(X[numeric_cols].median())

            selector = AutomatedFeatureSelector(task_type=task_type, method=method, k=k)
            selector.fit(X_num, y)
            summary = selector.get_summary()

            return {
                "status": "success",
                "task_type": task_type,
                "selection_summary": summary,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def tune_model(
        self,
        df: pd.DataFrame,
        target_col: str,
        cv_folds: int = 5,
        tuning_method: str = "grid",
    ) -> Dict[str, Any]:
        """
        MCP tool to run hyperparameter tuning with cross-validation.
        """
        try:
            from tools.model_tools import ModelTools
            tool = ModelTools()
            return tool.tune(df, target_col=target_col, cv_folds=cv_folds, tuning_method=tuning_method)
        except Exception as e:
            return {"status": "error", "error": str(e)}
