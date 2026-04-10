from __future__ import annotations

import logging

from src.drift.detector import detect_data_drift
from src.drift.performance_monitor import evaluate_live_performance
from src.training.model_registry import load_registry, rollback_to_version
from src.training.train import train_model
from src.utils.alerts import send_alert
from src.utils.settings import get_settings

logger = logging.getLogger(__name__)


def should_trigger_retraining() -> dict:
    drift_result = detect_data_drift()
    perf_result = evaluate_live_performance()

    retrain_reason = []
    if drift_result["drift_detected"]:
        retrain_reason.append("data_drift")
    if perf_result.get("performance_drop"):
        retrain_reason.append("performance_drop")

    return {
        "trigger_retraining": len(retrain_reason) > 0,
        "reasons": retrain_reason,
        "drift": drift_result,
        "performance": perf_result,
    }


def retrain_and_replace_if_better() -> dict:
    settings = get_settings()
    decision = should_trigger_retraining()

    if not decision["trigger_retraining"]:
        logger.info("No retraining required")
        return {"action": "skip", "decision": decision}

    send_alert(
        "Self-healing trigger activated due to: " + ", ".join(decision["reasons"])
    )

    old_registry = load_registry()
    old_version = old_registry.get("current_version")
    old_model = None
    if old_version is not None:
        old_model = next((h for h in old_registry.get("history", []) if h["version"] == old_version), None)

    train_output = train_model(run_name="self_healing_retrain")

    new_accuracy = train_output["accuracy"]
    old_accuracy = old_model["accuracy"] if old_model else None

    if old_accuracy is not None:
        required = old_accuracy + settings.min_acceptable_improvement
        if new_accuracy < required:
            send_alert(
                f"New model underperformed. old={old_accuracy:.4f}, new={new_accuracy:.4f}. Keep reviewing manually."
            )
            if old_version is not None:
                rollback_to_version(old_version)
                return {
                    "action": "rolled_back",
                    "rolled_back_to_version": old_version,
                    "candidate_version": train_output["current_version"],
                    "new_accuracy": new_accuracy,
                    "old_accuracy": old_accuracy,
                    "decision": decision,
                }

    return {
        "action": "retrained",
        "previous_version": old_version,
        "new_version": train_output["current_version"],
        "new_accuracy": new_accuracy,
        "old_accuracy": old_accuracy,
        "decision": decision,
    }


def run_self_healing_cycle() -> dict:
    return retrain_and_replace_if_better()


if __name__ == "__main__":
    from src.utils.logging_utils import configure_logging

    configure_logging()
    logger.info("Self-heal output: %s", run_self_healing_cycle())
