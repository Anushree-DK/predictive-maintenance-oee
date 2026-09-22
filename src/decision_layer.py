"""Answers the diagram's four questions per machine: why will it fail, what's
causing degradation, what should we do, how confident are we.

In snowflake mode this calls SNOWFLAKE.CORTEX.COMPLETE with the feature/prediction
context. In mock mode it uses a template so the dashboard is fully demoable
without burning Cortex credits or needing a warehouse.
"""

import config

_PROMPT_TEMPLATE = """You are a predictive-maintenance analyst. Given this machine's
condition, answer in 4 short bullet points: (1) why it will fail, (2) the likely
root cause of degradation, (3) the single most useful action to take now, (4) a one
line caveat about the confidence level.

Machine: {machine_id}
Predicted remaining useful life: {rul} cycles
Failure probability (next 30 cycles): {failure_probability}
Risk class: {risk_class}
Model confidence: {confidence}
Top drifting sensors (slope over recent cycles): {top_sensors}
"""

_ACTION_BY_RISK = {
    "CRITICAL": "Create a work order now and schedule repair within 24 hours.",
    "HIGH": "Schedule repair in the next maintenance window.",
    "MEDIUM": "Recommend ordering replacement parts and monitor closely.",
    "LOW": "No action needed; continue routine monitoring.",
}


def _top_drifting_sensors(feature_row, n=3) -> list[str]:
    slope_cols = [c for c in feature_row.index if c.endswith("_SLOPE")]
    ranked = sorted(slope_cols, key=lambda c: abs(feature_row[c]), reverse=True)[:n]
    return [c.replace("_SLOPE", "") for c in ranked]


def _mock_reasoning(machine_id, prediction_row, feature_row) -> dict:
    top_sensors = _top_drifting_sensors(feature_row)
    risk = prediction_row["RISK_CLASS"]
    return {
        "why": (
            f"Predicted RUL has dropped to {prediction_row['RUL_PREDICTION']:.0f} cycles "
            f"with a {prediction_row['FAILURE_PROBABILITY']:.0%} chance of failure in the next 30 cycles."
        ),
        "root_cause": (
            f"Sustained drift in {', '.join(top_sensors) if top_sensors else 'several sensors'}, "
            "consistent with progressive mechanical wear."
        ),
        "recommended_action": _ACTION_BY_RISK.get(risk, "Monitor."),
        "confidence_note": (
            f"Confidence score {prediction_row['CONFIDENCE_SCORE']:.0%} "
            f"({'high agreement across the model ensemble' if prediction_row['CONFIDENCE_SCORE'] > 0.7 else 'moderate agreement — treat as directional'})."
        ),
    }


def _cortex_reasoning(machine_id, prediction_row, feature_row) -> dict:
    from src.connection import get_session

    prompt = _PROMPT_TEMPLATE.format(
        machine_id=machine_id,
        rul=prediction_row["RUL_PREDICTION"],
        failure_probability=f"{prediction_row['FAILURE_PROBABILITY']:.0%}",
        risk_class=prediction_row["RISK_CLASS"],
        confidence=f"{prediction_row['CONFIDENCE_SCORE']:.0%}",
        top_sensors=", ".join(_top_drifting_sensors(feature_row)),
    )
    session = get_session()
    escaped_prompt = prompt.replace("'", "''")
    result = session.sql(
        f"SELECT SNOWFLAKE.CORTEX.COMPLETE('llama3.1-70b', '{escaped_prompt}') AS RESPONSE"
    ).collect()
    text = result[0]["RESPONSE"]
    return {"why": text, "root_cause": None, "recommended_action": None, "confidence_note": None}


def explain(machine_id, prediction_row, feature_row) -> dict:
    if config.SNOWFLAKE_MODE == "snowflake":
        try:
            return _cortex_reasoning(machine_id, prediction_row, feature_row)
        except Exception as e:
            # Cortex AI functions aren't available on every Snowflake account tier
            # (e.g. trial accounts) or region — fall back rather than break the
            # dashboard. The template reasoning still uses this machine's real
            # prediction and feature data, just without an LLM writing the prose.
            reasoning = _mock_reasoning(machine_id, prediction_row, feature_row)
            reasoning["why"] = f"[Cortex unavailable: {e}] {reasoning['why']}"
            return reasoning
    return _mock_reasoning(machine_id, prediction_row, feature_row)
