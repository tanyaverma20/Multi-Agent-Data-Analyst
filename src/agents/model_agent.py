# src/agents/model_agent.py
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from tools.file_tools import FileTools
from tools.memory_tools import MemoryTools
from tools.model_tools import ModelTools

try:
    from core.a2a_bus import A2ABus
except Exception:
    A2ABus = None


class ModelAgent:
    """
    ModelAgent (AutoML Agent).
    Orchestrates:
    - Feature Engineering
    - Outlier Detection & Handling
    - Feature Creation & Transformation
    - Automated Feature Selection
    - Hyperparameter Tuning (GridSearchCV / RandomizedSearchCV)
    - K-Fold Cross-Validation & Model Selection
    - Feature Importance Analysis
    - Publishes fine-grained A2A lifecycle events and registers model with memory & bus.
    """

    def __init__(
        self,
        a2a_bus: Optional[Any] = None,
        output_dir: str = "models",
        random_state: int = 42,
        n_iter_search: int = 20,
        cv: int = 5,
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.file_tool = FileTools()
        self.memory = MemoryTools()
        self.random_state = random_state
        self.n_iter_search = n_iter_search
        self.cv = cv
        self.a2a_bus = a2a_bus
        self.model_tools = ModelTools(output_dir=self.output_dir)

        if self.a2a_bus and hasattr(self.a2a_bus, "register_agent"):
            self.a2a_bus.register_agent("model")

    def _publish_event(self, topic: str, to: str, payload: Dict[str, Any]):
        """Helper to publish event to A2A bus if available."""
        if self.a2a_bus and hasattr(self.a2a_bus, "publish"):
            try:
                self.a2a_bus.publish(
                    from_agent="model",
                    to=to,
                    topic=topic,
                    payload=payload,
                )
            except Exception:
                pass

    def _train_and_evaluate(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        cv_folds: Optional[int] = None,
        tune_hyperparameters: bool = True,
        tuning_method: str = "grid",
        param_grids: Optional[Dict[str, Dict[str, Any]]] = None,
        outlier_strategy: str = "clip",
        feature_selection_k: Any = "auto",
    ) -> Dict[str, Any]:
        """
        Executes end-to-end AutoML pipeline via ModelTools with event publication.
        """
        seed = random_state if random_state is not None else self.random_state
        folds = cv_folds if cv_folds is not None else self.cv

        # Determine target column if not supplied
        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]

        # 1. Publish feature engineering started
        self._publish_event(
            topic="feature_engineering.started",
            to="broadcast",
            payload={"target_col": target_col, "rows": len(df), "cols": len(df.columns)},
        )

        # 2. Publish hyperparameter tuning started if enabled
        if tune_hyperparameters:
            self._publish_event(
                topic="hyperparameter_tuning.started",
                to="broadcast",
                payload={"target_col": target_col, "cv_folds": folds, "tuning_method": tuning_method},
            )

        # 3. Run training in ModelTools
        result = self.model_tools.train(
            df=df,
            target_col=target_col,
            cv_folds=folds,
            tune_hyperparameters=tune_hyperparameters,
            tuning_method=tuning_method,
            param_grids=param_grids,
            outlier_strategy=outlier_strategy,
            feature_selection_k=feature_selection_k,
            random_state=seed,
        )

        if result.get("status") == "success":
            # 4. Publish feature engineering completed
            self._publish_event(
                topic="feature_engineering.completed",
                to="broadcast",
                payload={
                    "feature_engineering": result.get("feature_engineering", {}),
                    "outliers": result.get("outliers", {}),
                },
            )

            # 5. Publish hyperparameter tuning completed if enabled
            if tune_hyperparameters and "hyperparameter_tuning" in result:
                self._publish_event(
                    topic="hyperparameter_tuning.completed",
                    to="broadcast",
                    payload={"hyperparameter_tuning": result.get("hyperparameter_tuning", {})},
                )

            # 6. Publish cross validation completed
            self._publish_event(
                topic="cross_validation.completed",
                to="broadcast",
                payload={
                    "cross_validation": result.get("cross_validation", {}),
                    "best_model": result.get("best_model"),
                },
            )

            # 7. Publish feature selection completed
            self._publish_event(
                topic="feature_selection.completed",
                to="broadcast",
                payload={"feature_selection": result.get("feature_selection", {})},
            )

            # 8. Publish feature importance completed
            self._publish_event(
                topic="feature_importance.completed",
                to="broadcast",
                payload={"feature_importance": result.get("feature_importance", [])},
            )

        return result

    def poll_messages_and_run(self, df: pd.DataFrame, target_col: Optional[str] = None):
        """
        Check for A2A messages sent to the model agent
        and auto-run training if EDA is completed.
        """
        if not self.a2a_bus:
            return None

        messages = self.a2a_bus.fetch("model", consume=True)
        trained_result = None

        for msg in messages:
            if msg.get("topic") == "eda.completed":
                payload = msg.get("payload", {})
                self.memory.save("eda_output", payload)

                chosen_target = target_col
                if not chosen_target:
                    num_cols = payload.get("numeric_columns", [])
                    if num_cols:
                        chosen_target = num_cols[-1]
                    else:
                        chosen_target = df.columns[-1]

                trained_result = self.run(df, target_col=chosen_target)

        return trained_result

    def run(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        cv_folds: Optional[int] = None,
        tune_hyperparameters: bool = True,
        tuning_method: str = "grid",
        param_grids: Optional[Dict[str, Dict[str, Any]]] = None,
        outlier_strategy: str = "clip",
        feature_selection_k: Any = "auto",
    ) -> Dict[str, Any]:
        """
        Full AutoML entrypoint. Runs pipeline, updates memory, and alerts VerifierAgent via A2A.
        """
        result = self._train_and_evaluate(
            df=df,
            target_col=target_col,
            test_size=test_size,
            random_state=random_state,
            cv_folds=cv_folds,
            tune_hyperparameters=tune_hyperparameters,
            tuning_method=tuning_method,
            param_grids=param_grids,
            outlier_strategy=outlier_strategy,
            feature_selection_k=feature_selection_k,
        )

        self.memory.save("model_output", result)

        if result.get("status") == "success":
            self._publish_event(
                topic="model.trained",
                to="verifier",
                payload={"model_output": result},
            )

        return result

    def tune(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        cv_folds: Optional[int] = None,
        tuning_method: str = "grid",
        param_grids: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Explicit hyperparameter tuning entrypoint."""
        return self.run(
            df=df,
            target_col=target_col,
            cv_folds=cv_folds,
            tune_hyperparameters=True,
            tuning_method=tuning_method,
            param_grids=param_grids,
        )
