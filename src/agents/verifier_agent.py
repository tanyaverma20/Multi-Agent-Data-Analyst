# File: src/agents/verifier_agent.py
# Updated VerifierAgent to register to bus and verify model reliability, CV stability, and feature engineering.
import os
from tools.memory_tools import MemoryTools

try:
    from core.a2a_bus import A2ABus
except Exception:
    A2ABus = None


class VerifierAgent:
    def __init__(self, a2a_bus: A2ABus = None):
        self.memory = MemoryTools()
        self.a2a_bus = a2a_bus
        if self.a2a_bus and hasattr(self.a2a_bus, "register_agent"):
            self.a2a_bus.register_agent("verifier")

    def run(self, model_result):
        try:
            # simple sanity checks
            if not model_result or model_result.get("status") != "success":
                return {"status": "error", "error": "No trained model provided."}

            metrics = model_result.get("metrics", {})
            notes = ["Basic verification completed"]

            # Evaluate quality based on performance
            quality = "Unknown"
            if "r2" in metrics:
                r2 = metrics.get("r2")
                if r2 is None:
                    quality = "Unreliable"
                elif r2 > 0.7:
                    quality = "Good"
                elif r2 > 0.4:
                    quality = "Acceptable"
                else:
                    quality = "Weak"
            elif "f1" in metrics or "f1_score" in metrics:
                f1 = metrics.get("f1") or metrics.get("f1_score")
                if f1 is None:
                    quality = "Unreliable"
                elif f1 > 0.75:
                    quality = "Good"
                elif f1 > 0.5:
                    quality = "Acceptable"
                else:
                    quality = "Weak"

            # Check Cross-Validation stability
            cv_info = model_result.get("cross_validation", {})
            best_model_name = model_result.get("best_model") or model_result.get("model_name")
            if cv_info and "models" in cv_info and best_model_name in cv_info["models"]:
                best_cv = cv_info["models"][best_model_name]
                std_score = best_cv.get("std_cv_score", 0.0)
                mean_score = best_cv.get("mean_cv_score", 0.0)
                if std_score <= 0.08:
                    notes.append(f"CV variance is low (std: {std_score}), indicating stable generalization.")
                else:
                    notes.append(f"CV variance is moderate (std: {std_score}). Consider collecting more data.")

            # Check Outlier handling
            outliers = model_result.get("outliers", {})
            if outliers and outliers.get("total_outliers", 0) > 0:
                notes.append(
                    f"Outliers safely handled ({outliers.get('total_outliers')} values clipped to training IQR bounds)."
                )

            # Check Feature Selection
            fs = model_result.get("feature_selection", {})
            if fs and fs.get("selected_count", 0) > 0:
                notes.append(
                    f"Automated feature selection retained top {fs.get('selected_count')} of {fs.get('original_count')} features."
                )

            # Check Hyperparameter Tuning
            ht = model_result.get("hyperparameter_tuning", {})
            if ht and ht.get("enabled"):
                best_params = ht.get("best_overall_params", {})
                method = ht.get("method", "GridSearchCV")
                notes.append(
                    f"Hyperparameter tuning applied ({method}); optimal hyperparameters for {best_model_name}: {best_params}."
                )

            result = {
                "status": "success",
                "quality": quality,
                "metrics": metrics,
                "notes": notes,
            }

            # save to memory
            self.memory.save("verifier_output", result)

            # publish audit message
            if self.a2a_bus:
                self.a2a_bus.publish(
                    from_agent="verifier",
                    to="notebook",
                    topic="verifier.completed",
                    payload={"verifier_output": result},
                )

            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def poll_messages_and_run(self):
        """
        If there are messages from 'model' indicating training finished, fetch and run verification.
        """
        if not self.a2a_bus:
            return None
        msgs = self.a2a_bus.fetch("verifier", consume=True)
        ran = None
        for m in msgs:
            if m.get("topic") == "model.trained":
                payload = m.get("payload", {})
                model_out = payload.get("model_output")
                res = self.run(model_out)
                ran = res
        return ran
