"""Agentic actions and the feedback loop: log what was recommended/taken, and later
whether the predicted failure actually occurred. Kept deliberately simple — this is
the foundation for continuous learning, not a retraining pipeline.
"""

from __future__ import annotations

import uuid

from src import data_access


def log_action(machine_id: str, action_type: str, prediction_row: dict) -> str:
    action_id = str(uuid.uuid4())
    data_access.append_action_outcome(
        {
            "ACTION_ID": action_id,
            "MACHINE_ID": machine_id,
            "ACTION_TYPE": action_type,
            "RECOMMENDED_AT": data_access.now_iso(),
            "RUL_PREDICTION_AT_ACTION": prediction_row.get("RUL_PREDICTION"),
            "RISK_CLASS_AT_ACTION": prediction_row.get("RISK_CLASS"),
            "CONFIDENCE_AT_ACTION": prediction_row.get("CONFIDENCE_SCORE"),
            "TAKEN_FLAG": True,
            "TAKEN_AT": data_access.now_iso(),
            "FAILURE_OCCURRED_FLAG": None,
            "FAILURE_DATE": None,
            "NOTES": None,
        }
    )
    return action_id


def record_failure_result(action_id: str, failure_occurred: bool, failure_date: str | None = None) -> None:
    data_access.update_action_outcome_result(action_id, failure_occurred, failure_date)
