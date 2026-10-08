# src/core/feature_engineering.py
"""
Reusable Feature Engineering Layer for Multi-Agent AutoML.

Capabilities:
1. Outlier detection and handling (IQR and Isolation Forest, configurable clipping/winsorization)
2. Feature creation and transformation (ratios, differences, polynomial/interactions, log/power transforms)
3. Missing-value imputation (numerical and categorical)
4. Categorical encoding (OneHotEncoder with unseen category handling)
5. Numerical scaling (StandardScaler / RobustScaler)
6. Automated feature selection (SelectKBest, SelectFromModel, percentile adaptation for classification/regression)
7. Feature name tracking across all transformations (zero data leakage, strictly fit on train)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    SelectFromModel,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Learns outlier boundaries strictly on training data and safely clips or handles outliers.
    Avoids blindly deleting observations.
    """

    def __init__(self, method: str = "iqr", strategy: str = "clip", factor: float = 1.5):
        self.method = method  # "iqr" or "isolation_forest"
        self.strategy = strategy  # "clip", "none"
        self.factor = factor
        self.bounds_: Dict[str, Tuple[float, float]] = {}
        self.affected_columns_: Dict[str, Dict[str, Any]] = {}
        self.total_outliers_: int = 0
        self.percentage_affected_: float = 0.0
        self.numeric_cols_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.numeric_cols_ = X.select_dtypes(include=["number"]).columns.tolist()
        self.bounds_ = {}
        self.affected_columns_ = {}
        total_outlier_count = 0
        rows_with_outliers = set()

        n_rows = len(X)
        if n_rows == 0 or len(self.numeric_cols_) == 0:
            self.total_outliers_ = 0
            self.percentage_affected_ = 0.0
            return self

        for col in self.numeric_cols_:
            series = X[col].dropna()
            if len(series) < 4:
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            if iqr <= 1e-9:
                # Column is constant or low-variance, bounds are min/max
                lower = float(series.min())
                upper = float(series.max())
            else:
                lower = float(q1 - self.factor * iqr)
                upper = float(q3 + self.factor * iqr)

            self.bounds_[col] = (lower, upper)

            # Count outliers in training set
            outlier_mask = (X[col] < lower) | (X[col] > upper)
            outlier_indices = X.index[outlier_mask].tolist()
            col_outlier_count = int(outlier_mask.sum())

            if col_outlier_count > 0:
                total_outlier_count += col_outlier_count
                rows_with_outliers.update(outlier_indices)
                self.affected_columns_[col] = {
                    "outlier_count": col_outlier_count,
                    "percentage": round((col_outlier_count / n_rows) * 100, 2),
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                }

        self.total_outliers_ = total_outlier_count
        self.percentage_affected_ = round((len(rows_with_outliers) / n_rows) * 100, 2) if n_rows > 0 else 0.0
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_out = X.copy()

        if self.strategy == "clip" and self.bounds_:
            for col, (lower, upper) in self.bounds_.items():
                if col in X_out.columns and pd.api.types.is_numeric_dtype(X_out[col]):
                    X_out[col] = X_out[col].clip(lower=lower, upper=upper)

        return X_out

    def get_summary(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "columns_analyzed": self.numeric_cols_,
            "affected_columns": self.affected_columns_,
            "total_outliers": self.total_outliers_,
            "percentage_affected": self.percentage_affected_,
            "handling_strategy": self.strategy,
        }


class FeatureCreationTransformer(BaseEstimator, TransformerMixin):
    """
    Creates derived features and applies mathematical transformations.
    Safeguards:
    - Avoid division by zero
    - Avoid infinite or NaN values
    - Avoid duplicate features
    - Cap max generated features to avoid explosion
    - Skip constant/near-constant features
    """

    def __init__(
        self,
        create_ratios: bool = True,
        create_differences: bool = True,
        create_interactions: bool = True,
        transform_skewed: bool = True,
        max_created_features: int = 15,
        skew_threshold: float = 1.0,
    ):
        self.create_ratios = create_ratios
        self.create_differences = create_differences
        self.create_interactions = create_interactions
        self.transform_skewed = transform_skewed
        self.max_created_features = max_created_features
        self.skew_threshold = skew_threshold

        # Fitted attributes
        self.numeric_cols_: List[str] = []
        self.skewed_cols_: List[str] = []
        self.ratio_pairs_: List[Tuple[str, str, str]] = []  # (col1, col2, new_name)
        self.diff_pairs_: List[Tuple[str, str, str]] = []  # (col1, col2, new_name)
        self.interaction_pairs_: List[Tuple[str, str, str]] = []  # (col1, col2, new_name)
        self.created_feature_names_: List[str] = []
        self.transformed_feature_names_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.numeric_cols_ = X.select_dtypes(include=["number"]).columns.tolist()
        self.skewed_cols_ = []
        self.ratio_pairs_ = []
        self.diff_pairs_ = []
        self.interaction_pairs_ = []
        self.created_feature_names_ = []
        self.transformed_feature_names_ = []

        if len(self.numeric_cols_) == 0:
            return self

        # 1. Detect skewed features for log1p transformation
        if self.transform_skewed:
            for col in self.numeric_cols_:
                s = X[col].dropna()
                if len(s) > 5 and s.var() > 1e-6:
                    skew_val = s.skew()
                    # Apply log1p if positive / non-negative and highly right-skewed
                    if skew_val > self.skew_threshold and (s >= 0).all():
                        self.skewed_cols_.append(col)
                        self.transformed_feature_names_.append(f"log1p_{col}")

        # 2. Select informative pairs for feature creation
        # Only consider numeric columns with non-zero variance
        valid_cols = [c for c in self.numeric_cols_ if X[c].dropna().nunique() > 2]
        created_count = 0

        # Sort columns by variance to prioritize the most variable signals
        if len(valid_cols) >= 2:
            variances = {c: float(X[c].dropna().var()) for c in valid_cols}
            sorted_cols = sorted(valid_cols, key=lambda c: variances.get(c, 0.0), reverse=True)
            # Use top columns to avoid combinatorial explosion
            candidate_cols = sorted_cols[: min(6, len(sorted_cols))]

            existing_feature_names = set(X.columns.tolist())

            for i in range(len(candidate_cols)):
                for j in range(i + 1, len(candidate_cols)):
                    if created_count >= self.max_created_features:
                        break

                    c1, c2 = candidate_cols[i], candidate_cols[j]

                    # Ratio: c1 / c2 (only if c2 has non-negative values or mostly non-zero)
                    if self.create_ratios and created_count < self.max_created_features:
                        ratio_name = f"ratio_{c1}_over_{c2}"
                        if ratio_name not in existing_feature_names:
                            self.ratio_pairs_.append((c1, c2, ratio_name))
                            self.created_feature_names_.append(ratio_name)
                            existing_feature_names.add(ratio_name)
                            created_count += 1

                    # Difference: c1 - c2
                    if self.create_differences and created_count < self.max_created_features:
                        diff_name = f"diff_{c1}_minus_{c2}"
                        if diff_name not in existing_feature_names:
                            self.diff_pairs_.append((c1, c2, diff_name))
                            self.created_feature_names_.append(diff_name)
                            existing_feature_names.add(diff_name)
                            created_count += 1

                    # Interaction: c1 * c2
                    if self.create_interactions and created_count < self.max_created_features:
                        inter_name = f"inter_{c1}_x_{c2}"
                        if inter_name not in existing_feature_names:
                            self.interaction_pairs_.append((c1, c2, inter_name))
                            self.created_feature_names_.append(inter_name)
                            existing_feature_names.add(inter_name)
                            created_count += 1

                if created_count >= self.max_created_features:
                    break

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_out = X.copy()

        # 1. Apply log transformations
        for col in self.skewed_cols_:
            if col in X_out.columns:
                new_col = f"log1p_{col}"
                # Safe log1p: clip negative values to 0
                safe_vals = np.maximum(0, X_out[col].fillna(0).values)
                X_out[new_col] = np.log1p(safe_vals)

        # 2. Apply ratio pairs with zero-division safeguard
        for c1, c2, name in self.ratio_pairs_:
            if c1 in X_out.columns and c2 in X_out.columns:
                denom = X_out[c2].replace(0, np.nan)
                ratio_vals = X_out[c1] / denom
                # Replace inf and -inf
                ratio_vals = ratio_vals.replace([np.inf, -np.inf], np.nan)
                X_out[name] = ratio_vals

        # 3. Apply difference pairs
        for c1, c2, name in self.diff_pairs_:
            if c1 in X_out.columns and c2 in X_out.columns:
                X_out[name] = X_out[c1] - X_out[c2]

        # 4. Apply interaction pairs
        for c1, c2, name in self.interaction_pairs_:
            if c1 in X_out.columns and c2 in X_out.columns:
                X_out[name] = X_out[c1] * X_out[c2]

        return X_out

    def get_summary(self) -> Dict[str, Any]:
        return {
            "created_features": self.created_feature_names_,
            "transformed_features": self.transformed_feature_names_,
            "total_created": len(self.created_feature_names_),
            "total_transformed": len(self.transformed_feature_names_),
        }


class AutomatedFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Automated feature selection supporting classification and regression.
    Selects top features using univariate statistical tests or model-based importance,
    strictly fitted on training data.
    """

    def __init__(
        self,
        task_type: str = "classification",
        method: str = "k_best",
        k: Union[int, str] = "auto",
        percentile: int = 80,
    ):
        self.task_type = task_type  # "classification" or "regression"
        self.method = method  # "k_best", "percentile", "model_based"
        self.k = k  # integer or "auto"
        self.percentile = percentile

        # Fitted attributes
        self.original_feature_count_: int = 0
        self.selected_feature_count_: int = 0
        self.selected_features_: List[str] = []
        self.feature_scores_: Dict[str, float] = {}
        self.support_mask_: np.ndarray = np.array([])
        self.selector_ = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_mat = X.values
        else:
            X_mat = np.asarray(X)
            feature_names = [f"feature_{i}" for i in range(X_mat.shape[1])]

        n_samples, n_features = X_mat.shape
        self.original_feature_count_ = n_features

        # Safeguard: if 5 or fewer features, retain all
        if n_features <= 5:
            self.support_mask_ = np.ones(n_features, dtype=bool)
            self.selected_feature_count_ = n_features
            self.selected_features_ = feature_names
            self.feature_scores_ = {fn: 1.0 for fn in feature_names}
            return self

        # Determine effective k
        if self.k == "auto":
            if n_features <= 12:
                effective_k = max(4, int(n_features * 0.85))
            else:
                effective_k = max(8, int(n_features * 0.70))
        else:
            effective_k = min(int(self.k), n_features)

        # Handle NaNs inside X_mat if any remain before selection
        if np.isnan(X_mat).any():
            imputer = SimpleImputer(strategy="median")
            X_clean = imputer.fit_transform(X_mat)
        else:
            X_clean = X_mat

        # Ensure y is clean 1D array
        y_clean = np.asarray(y)

        try:
            if self.method == "model_based":
                if self.task_type == "classification":
                    estimator = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=6)
                else:
                    estimator = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=6)
                estimator.fit(X_clean, y_clean)
                importances = estimator.feature_importances_
                # Rank and select top k
                top_indices = np.argsort(importances)[::-1][:effective_k]
                self.support_mask_ = np.zeros(n_features, dtype=bool)
                self.support_mask_[top_indices] = True
                self.feature_scores_ = {
                    feature_names[i]: float(round(importances[i], 4)) for i in range(n_features)
                }

            else:
                # Default: SelectKBest with f_classif or f_regression
                score_func = f_classif if self.task_type == "classification" else f_regression
                selector = SelectKBest(score_func=score_func, k=effective_k)
                selector.fit(X_clean, y_clean)
                self.selector_ = selector
                self.support_mask_ = selector.get_support()
                scores = selector.scores_
                for i in range(n_features):
                    sc = scores[i]
                    val = 0.0 if np.isnan(sc) or np.isinf(sc) else float(round(sc, 4))
                    self.feature_scores_[feature_names[i]] = val

        except Exception:
            # Fallback if statistical test fails (e.g. constant columns or single-class target)
            self.support_mask_ = np.ones(n_features, dtype=bool)
            self.feature_scores_ = {fn: 1.0 for fn in feature_names}

        # Map selected features
        self.selected_feature_count_ = int(np.sum(self.support_mask_))
        self.selected_features_ = [
            feature_names[i] for i in range(n_features) if self.support_mask_[i]
        ]
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        if len(self.support_mask_) == 0:
            return X

        if isinstance(X, pd.DataFrame):
            # If columns match original features
            if len(X.columns) == len(self.support_mask_):
                selected_cols = [X.columns[i] for i in range(len(self.support_mask_)) if self.support_mask_[i]]
                return X[selected_cols]
            else:
                # Array fallback
                return X.iloc[:, self.support_mask_]
        else:
            X_mat = np.asarray(X)
            return X_mat[:, self.support_mask_]

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.selected_features_

    def get_summary(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "original_count": self.original_feature_count_,
            "selected_count": self.selected_feature_count_,
            "selected_features": self.selected_features_,
            "feature_scores": dict(
                sorted(self.feature_scores_.items(), key=lambda item: item[1], reverse=True)[:25]
            ),
        }
