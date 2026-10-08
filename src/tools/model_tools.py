# src/tools/model_tools.py
import os
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from core.feature_engineering import (
    OutlierHandler,
    FeatureCreationTransformer,
    AutomatedFeatureSelector,
)


class FullAutoMLPipeline(BaseEstimator):
    """
    End-to-end composite pipeline chaining:
    1. OutlierHandler (learned on train)
    2. FeatureCreationTransformer (learned on train)
    3. ColumnTransformer (imputation + scaling + one-hot encoding)
    4. AutomatedFeatureSelector (learned on train)
    5. Final Estimator
    Ensures safe inference and zero data leakage.
    """

    def __init__(
        self,
        outlier_handler: OutlierHandler,
        feature_creator: FeatureCreationTransformer,
        preprocessor: ColumnTransformer,
        feature_selector: AutomatedFeatureSelector,
        estimator: BaseEstimator,
        feature_names_in: Optional[List[str]] = None,
        final_feature_names: Optional[List[str]] = None,
        feature_names_in_: Optional[List[str]] = None,
        final_feature_names_: Optional[List[str]] = None,
    ):
        self.outlier_handler = outlier_handler
        self.feature_creator = feature_creator
        self.preprocessor = preprocessor
        self.feature_selector = feature_selector
        self.estimator = estimator
        self.feature_names_in_ = feature_names_in if feature_names_in is not None else (feature_names_in_ or [])
        self.final_feature_names_ = final_feature_names if final_feature_names is not None else (final_feature_names_ or [])

    def transform_features(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        X_out = self.outlier_handler.transform(X)
        X_out = self.feature_creator.transform(X_out)
        X_prep = self.preprocessor.transform(X_out)
        X_sel = self.feature_selector.transform(X_prep)
        return X_sel

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_trans = self.transform_features(X)
        return self.estimator.predict(X_trans)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            X_trans = self.transform_features(X)
            return self.estimator.predict_proba(X_trans)
        raise AttributeError("Underlying estimator does not support predict_proba")


class ModelTools:
    """
    Production-quality AutoML tool supporting:
    1. Automated task detection (Classification / Regression)
    2. Outlier detection & handling
    3. Feature creation & transformation
    4. Automated feature selection
    5. Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
    6. Cross-validation (StratifiedKFold / KFold)
    7. Feature importance analysis (Tree, Linear, Permutation)
    8. Evaluation on held-out test data (Zero Data Leakage)
    """

    def __init__(self, output_dir="models"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ----------------------------------------------------
    # TASK DETECTION
    # ----------------------------------------------------
    def _detect_task(self, y: pd.Series) -> str:
        """
        Detects classification or regression.
        Maintains backward compatibility while handling binary, numeric, and categorical targets.
        """
        if pd.api.types.is_numeric_dtype(y):
            if y.nunique() > 20:
                return "regression"
            if pd.api.types.is_float_dtype(y) and y.nunique() > 10:
                return "regression"
            return "classification"
        return "classification"

    # ----------------------------------------------------
    # BUILD PREPROCESSOR FOR TRANSFORMED DATA
    # ----------------------------------------------------
    def _build_preprocessor(self, X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
        """
        Builds ColumnTransformer handling numeric and categorical features.
        Uses SimpleImputer, StandardScaler, and OneHotEncoder with unseen category handling.
        """
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        transformers = []
        if numeric_cols:
            transformers.append(("num", numeric_pipeline, numeric_cols))
        if cat_cols:
            transformers.append(("cat", categorical_pipeline, cat_cols))

        preprocessor = ColumnTransformer(transformers=transformers)
        return preprocessor, numeric_cols, cat_cols

    # ----------------------------------------------------
    # METRICS CALCULATION
    # ----------------------------------------------------
    def _score(self, task: str, y_true, y_pred, y_prob=None) -> Dict[str, float]:
        """
        Calculate metrics.
        Returns accuracy, precision, recall, f1_score for classification.
        Returns mse, rmse, mae, r2 for regression.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if task == "classification":
            metrics = {
                "accuracy": float(round(accuracy_score(y_true, y_pred), 4)),
                "f1_score": float(round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4)),
                "precision": float(round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4)),
                "recall": float(round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4)),
            }
            metrics["f1"] = metrics["f1_score"]

            if y_prob is not None:
                try:
                    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                        metrics["roc_auc"] = float(round(roc_auc_score(y_true, y_prob[:, 1]), 4))
                    elif y_prob.ndim == 2 and y_prob.shape[1] > 2:
                        metrics["roc_auc"] = float(round(roc_auc_score(y_true, y_prob, multi_class="ovr"), 4))
                except Exception:
                    pass
            return metrics
        else:
            mse_val = float(mean_squared_error(y_true, y_pred))
            mae_val = float(mean_absolute_error(y_true, y_pred))
            r2_val = float(r2_score(y_true, y_pred))
            return {
                "mse": float(round(mse_val, 4)),
                "rmse": float(round(np.sqrt(max(0.0, mse_val)), 4)),
                "mae": float(round(mae_val, 4)),
                "r2": float(round(r2_val, 4)),
            }

    # ----------------------------------------------------
    # CANDIDATE MODEL FACTORY
    # ----------------------------------------------------
    def _get_candidate_models(self, task: str, random_state: int = 42) -> Dict[str, BaseEstimator]:
        if task == "classification":
            return {
                "rf": RandomForestClassifier(n_estimators=100, random_state=random_state),
                "logreg": LogisticRegression(max_iter=1000, random_state=random_state),
                "gbm": GradientBoostingClassifier(n_estimators=80, random_state=random_state),
            }
        else:
            return {
                "rf": RandomForestRegressor(n_estimators=100, random_state=random_state),
                "ridge": Ridge(),
                "svr": SVR(),
            }

    # ----------------------------------------------------
    # TASK-SPECIFIC HYPERPARAMETER GRIDS
    # ----------------------------------------------------
    def _get_hyperparameter_grids(self, task: str) -> Dict[str, Dict[str, List[Any]]]:
        """
        Sensible, computationally manageable parameter grids for candidate models.
        """
        if task == "classification":
            return {
                "rf": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                },
                "logreg": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["lbfgs"],
                },
                "gbm": {
                    "n_estimators": [50, 80],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5],
                    "min_samples_split": [2, 5],
                },
            }
        else:
            return {
                "rf": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                },
                "ridge": {
                    "alpha": [0.1, 1.0, 10.0, 50.0],
                },
                "svr": {
                    "C": [0.5, 1.0, 5.0],
                    "epsilon": [0.1, 0.2],
                    "kernel": ["rbf", "linear"],
                },
            }

    # ----------------------------------------------------
    # EXTRACT FEATURE NAMES AFTER PREPROCESSING
    # ----------------------------------------------------
    def _get_preprocessed_feature_names(
        self, preprocessor: ColumnTransformer, numeric_cols: List[str], cat_cols: List[str]
    ) -> List[str]:
        feature_names = []
        try:
            raw_names = preprocessor.get_feature_names_out()
            for name in raw_names:
                clean_name = name.replace("num__", "").replace("cat__", "")
                feature_names.append(clean_name)
        except Exception:
            feature_names.extend(numeric_cols)
            for transformer_name, transformer, cols in preprocessor.transformers_:
                if transformer_name == "cat" and hasattr(transformer.named_steps["encode"], "get_feature_names_out"):
                    cat_enc_names = transformer.named_steps["encode"].get_feature_names_out(cols)
                    feature_names.extend(cat_enc_names.tolist())
        return feature_names

    # ----------------------------------------------------
    # FEATURE IMPORTANCE ANALYSIS
    # ----------------------------------------------------
    def _compute_feature_importance(
        self,
        estimator: BaseEstimator,
        feature_names: List[str],
        X_val: np.ndarray,
        y_val: np.ndarray,
        task: str,
        model_name: str,
    ) -> List[Dict[str, Any]]:
        importances = None
        directions = ["N/A"] * len(feature_names)

        # 1. Tree-based models
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_

        # 2. Linear models
        elif hasattr(estimator, "coef_"):
            coef = estimator.coef_
            if coef.ndim == 1:
                importances = np.abs(coef)
                directions = ["positive" if c > 0 else "negative" for c in coef]
            else:
                importances = np.mean(np.abs(coef), axis=0)
                if coef.shape[0] == 1:
                    directions = ["positive" if c > 0 else "negative" for c in coef[0]]
                else:
                    directions = ["multiclass" for _ in range(len(feature_names))]

        # 3. Fallback: Permutation importance
        if importances is None or len(importances) != len(feature_names):
            try:
                perm = permutation_importance(
                    estimator, X_val, y_val, n_repeats=5, random_state=42, n_jobs=1
                )
                importances = perm.importances_mean
                directions = ["N/A"] * len(feature_names)
            except Exception:
                importances = np.ones(len(feature_names)) / max(1, len(feature_names))

        total = np.sum(np.maximum(0, importances))
        if total > 0:
            norm_importances = np.maximum(0, importances) / total
        else:
            norm_importances = np.asarray(importances)

        ranked_indices = np.argsort(norm_importances)[::-1]
        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            feat_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            results.append({
                "rank": rank,
                "feature": feat_name,
                "importance": float(round(norm_importances[idx], 4)),
                "direction": directions[idx] if idx < len(directions) else "N/A",
            })

        return results

    # ----------------------------------------------------
    # MAIN TRAIN FUNCTION
    # ----------------------------------------------------
    def train(
        self,
        df: pd.DataFrame,
        target_col: str,
        cv_folds: int = 5,
        tune_hyperparameters: bool = True,
        tuning_method: str = "grid",
        param_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
        outlier_strategy: str = "clip",
        feature_selection_k: Union[int, str] = "auto",
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Production-grade AutoML pipeline:
        1. Fit preprocessing strictly on training data (Zero Data Leakage).
        2. Detect & handle outliers.
        3. Create & transform features.
        4. Select top informative features adaptively.
        5. Hyperparameter tuning using GridSearchCV/RandomizedSearchCV over CV.
        6. Select best model based on mean CV score.
        7. Fit final pipeline on train split and evaluate on held-out test split.
        8. Compute ranked feature importance table with name recovery.
        9. Save best model and return rich structured contract.
        """
        try:
            if target_col not in df.columns:
                return {"status": "error", "error": f"Target column '{target_col}' not found in dataframe."}

            X = df.drop(columns=[target_col])
            y = df[target_col]

            if X.shape[1] == 0:
                return {"status": "error", "error": "No feature columns found in dataset."}

            task = self._detect_task(y)

            # Split into train and held-out test set (80/20) - test set NEVER touched during tuning
            stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state, stratify=stratify
            )

            X_train = X_train.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

            # ------------------------------------------------
            # 1. FIT FEATURE ENGINEERING ON X_train
            # ------------------------------------------------
            outlier_handler = OutlierHandler(method="iqr", strategy=outlier_strategy)
            outlier_handler.fit(X_train)
            X_train_clean = outlier_handler.transform(X_train)

            feature_creator = FeatureCreationTransformer(
                create_ratios=True,
                create_differences=True,
                create_interactions=True,
                transform_skewed=True,
                max_created_features=15,
            )
            feature_creator.fit(X_train_clean, y_train)
            X_train_engineered = feature_creator.transform(X_train_clean)

            preprocessor, num_cols, cat_cols = self._build_preprocessor(X_train_engineered)
            X_train_preprocessed = preprocessor.fit_transform(X_train_engineered)

            intermediate_feature_names = self._get_preprocessed_feature_names(
                preprocessor, num_cols, cat_cols
            )

            feature_selector = AutomatedFeatureSelector(
                task_type=task,
                method="k_best",
                k=feature_selection_k,
            )
            feature_selector.fit(X_train_preprocessed, y_train)
            X_train_selected = feature_selector.transform(X_train_preprocessed)

            if len(feature_selector.support_mask_) == len(intermediate_feature_names):
                selected_feature_names = [
                    intermediate_feature_names[i]
                    for i in range(len(intermediate_feature_names))
                    if feature_selector.support_mask_[i]
                ]
            else:
                selected_feature_names = feature_selector.selected_features_

            outlier_summary = outlier_handler.get_summary()
            creation_summary = feature_creator.get_summary()
            selection_summary = feature_selector.get_summary()
            selection_summary["selected_features"] = selected_feature_names

            feature_engineering_summary = {
                "created_features": creation_summary["created_features"],
                "transformed_features": creation_summary["transformed_features"],
                "numeric_features": num_cols,
                "categorical_features": cat_cols,
                "transformations_applied": [
                    "IQR Outlier Clipping" if outlier_strategy == "clip" else "Outlier Detection",
                    "Log1p Skew Transform",
                    "Pairwise Ratios & Differences",
                    "Interaction Products",
                    "Median Imputation (Numeric)",
                    "Standard Scaling (Numeric)",
                    "One-Hot Encoding (Categorical)",
                    f"Automated Feature Selection ({selection_summary['method']})",
                ],
            }

            # ------------------------------------------------
            # 2. CROSS-VALIDATION + HYPERPARAMETER TUNING
            # ------------------------------------------------
            effective_folds = min(cv_folds, len(X_train))
            if task == "classification":
                min_class_count = y_train.value_counts().min()
                if min_class_count >= effective_folds and effective_folds >= 2:
                    cv_splitter = StratifiedKFold(
                        n_splits=effective_folds, shuffle=True, random_state=random_state
                    )
                else:
                    cv_folds_fallback = max(2, min(effective_folds, min_class_count))
                    cv_splitter = KFold(n_splits=cv_folds_fallback, shuffle=True, random_state=random_state)
            else:
                effective_folds = max(2, min(effective_folds, 5))
                cv_splitter = KFold(n_splits=effective_folds, shuffle=True, random_state=random_state)

            candidate_models = self._get_candidate_models(task, random_state=random_state)
            default_param_grids = self._get_hyperparameter_grids(task)
            active_grids = param_grids if param_grids is not None else default_param_grids

            cv_results = {}
            tuning_models_info = {}
            fitted_candidate_estimators = {}
            best_model_name = None
            best_cv_score = -float("inf")

            scoring_metric = "f1_weighted" if task == "classification" else "r2"

            for name, base_model in candidate_models.items():
                grid = active_grids.get(name, {})

                if tune_hyperparameters and grid:
                    t_start = time.time()
                    total_combinations = int(np.prod([len(v) for v in grid.values()])) if grid else 1

                    if tuning_method == "random" and total_combinations > 8:
                        search_method_name = "RandomizedSearchCV"
                        searcher = RandomizedSearchCV(
                            base_model,
                            param_distributions=grid,
                            n_iter=min(8, total_combinations),
                            cv=cv_splitter,
                            scoring=scoring_metric,
                            random_state=random_state,
                            n_jobs=1,
                        )
                    else:
                        search_method_name = "GridSearchCV"
                        searcher = GridSearchCV(
                            base_model,
                            param_grid=grid,
                            cv=cv_splitter,
                            scoring=scoring_metric,
                            n_jobs=1,
                        )

                    searcher.fit(X_train_selected, y_train)
                    duration_sec = round(time.time() - t_start, 3)

                    best_est = searcher.best_estimator_
                    best_params = searcher.best_params_
                    b_idx = searcher.best_index_
                    mean_score = float(round(searcher.cv_results_["mean_test_score"][b_idx], 4))
                    std_score = float(round(searcher.cv_results_["std_test_score"][b_idx], 4))

                    fold_scores = []
                    for i in range(effective_folds):
                        key = f"split{i}_test_score"
                        if key in searcher.cv_results_:
                            fold_scores.append(float(round(searcher.cv_results_[key][b_idx], 4)))

                    candidates_count = len(searcher.cv_results_["params"])

                    tuning_models_info[name] = {
                        "parameter_search_method": search_method_name,
                        "parameter_space": grid,
                        "best_params": best_params,
                        "best_cv_score": mean_score,
                        "cv_std": std_score,
                        "candidates_evaluated": candidates_count,
                        "search_duration_sec": duration_sec,
                        "fold_scores": fold_scores,
                    }

                    # Track CV metrics for candidate
                    train_preds = best_est.predict(X_train_selected)
                    train_prob = best_est.predict_proba(X_train_selected) if task == "classification" and hasattr(best_est, "predict_proba") else None
                    candidate_metrics = self._score(task, y_train, train_preds, train_prob)

                    cv_results[name] = {
                        "mean_cv_score": mean_score,
                        "std_cv_score": std_score,
                        "fold_scores": fold_scores,
                        "best_params": best_params,
                        "metrics": candidate_metrics,
                    }
                    fitted_candidate_estimators[name] = best_est

                else:
                    # Non-tuning path: evaluate base model directly over folds
                    fold_scores = []
                    fold_metrics_list = []
                    from sklearn.base import clone

                    for train_idx, val_idx in cv_splitter.split(X_train_selected, y_train):
                        X_f_tr = X_train_selected[train_idx] if isinstance(X_train_selected, np.ndarray) else X_train_selected.iloc[train_idx]
                        y_f_tr = y_train.iloc[train_idx]
                        X_f_va = X_train_selected[val_idx] if isinstance(X_train_selected, np.ndarray) else X_train_selected.iloc[val_idx]
                        y_f_va = y_train.iloc[val_idx]

                        fold_model = clone(base_model)
                        fold_model.fit(X_f_tr, y_f_tr)
                        preds = fold_model.predict(X_f_va)
                        prob = fold_model.predict_proba(X_f_va) if task == "classification" and hasattr(fold_model, "predict_proba") else None
                        fold_m = self._score(task, y_f_va, preds, prob)
                        fold_metrics_list.append(fold_m)
                        score = fold_m["f1_score"] if task == "classification" else fold_m["r2"]
                        fold_scores.append(score)

                    mean_score = float(round(np.mean(fold_scores), 4))
                    std_score = float(round(np.std(fold_scores), 4))
                    avg_metrics = {k: float(round(np.mean([m[k] for m in fold_metrics_list]), 4)) for k in fold_metrics_list[0].keys()}

                    cv_results[name] = {
                        "mean_cv_score": mean_score,
                        "std_cv_score": std_score,
                        "fold_scores": [float(round(s, 4)) for s in fold_scores],
                        "metrics": avg_metrics,
                    }
                    fitted_base = clone(base_model)
                    fitted_base.fit(X_train_selected, y_train)
                    fitted_candidate_estimators[name] = fitted_base

                if mean_score > best_cv_score:
                    best_cv_score = mean_score
                    best_model_name = name

            if best_model_name is None:
                best_model_name = list(candidate_models.keys())[0]

            winning_estimator = fitted_candidate_estimators[best_model_name]

            # ------------------------------------------------
            # 3. BUILD FULL DEPLOYABLE PIPELINE
            # ------------------------------------------------
            final_pipeline = FullAutoMLPipeline(
                outlier_handler=outlier_handler,
                feature_creator=feature_creator,
                preprocessor=preprocessor,
                feature_selector=feature_selector,
                estimator=winning_estimator,
                feature_names_in=X.columns.tolist(),
                final_feature_names=selected_feature_names,
            )

            # ------------------------------------------------
            # 4. EVALUATE ON HELD-OUT TEST SET
            # ------------------------------------------------
            test_preds = final_pipeline.predict(X_test)
            test_probs = None
            if task == "classification":
                try:
                    test_probs = final_pipeline.predict_proba(X_test)
                except Exception:
                    pass

            test_metrics = self._score(task, y_test, test_preds, test_probs)

            # ------------------------------------------------
            # 5. COMPUTE FEATURE IMPORTANCE
            # ------------------------------------------------
            X_test_clean = outlier_handler.transform(X_test)
            X_test_eng = feature_creator.transform(X_test_clean)
            X_test_prep = preprocessor.transform(X_test_eng)
            X_test_sel = feature_selector.transform(X_test_prep)

            feature_importance_table = self._compute_feature_importance(
                estimator=winning_estimator,
                feature_names=selected_feature_names,
                X_val=X_test_sel,
                y_val=np.asarray(y_test),
                task=task,
                model_name=best_model_name,
            )

            # ------------------------------------------------
            # 6. SAVE MODEL ARTIFACT
            # ------------------------------------------------
            model_path = os.path.join(self.output_dir, f"{best_model_name}_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(final_pipeline, f)

            sample_preds = test_preds[:10].tolist() if hasattr(test_preds, "tolist") else list(test_preds[:10])

            # Hyperparameter tuning summary
            tuning_summary = {
                "enabled": tune_hyperparameters,
                "method": "GridSearchCV" if tuning_method == "grid" else "RandomizedSearchCV",
                "cv_folds": effective_folds,
                "models": tuning_models_info,
                "best_tuned_model": best_model_name,
                "best_overall_params": tuning_models_info.get(best_model_name, {}).get("best_params", {}),
            }

            # ------------------------------------------------
            # 7. RETURN RICH CONTRACT (FULL BACKWARD COMPATIBILITY)
            # ------------------------------------------------
            return {
                "status": "success",
                "task_type": task,
                "model_name": best_model_name,
                "best_model": best_model_name,
                "model_path": model_path,
                "feature_engineering": feature_engineering_summary,
                "outliers": outlier_summary,
                "feature_selection": selection_summary,
                "hyperparameter_tuning": tuning_summary,
                "cross_validation": {
                    "folds": effective_folds,
                    "cv_metric": "f1_score" if task == "classification" else "r2",
                    "models": cv_results,
                },
                "feature_importance": feature_importance_table,
                "metrics": test_metrics,
                "test_metrics": test_metrics,
                "sample_predictions": sample_preds,
            }

        except Exception as e:
            import traceback
            return {"status": "error", "error": f"{str(e)}: {traceback.format_exc()}"}

    def tune(
        self,
        df: pd.DataFrame,
        target_col: str,
        cv_folds: int = 5,
        tuning_method: str = "grid",
        param_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Explicit tune method exposed as MCP tool (model.tune).
        """
        return self.train(
            df=df,
            target_col=target_col,
            cv_folds=cv_folds,
            tune_hyperparameters=True,
            tuning_method=tuning_method,
            param_grids=param_grids,
            random_state=random_state,
        )
