"""Per-engine diagnosis: Cortex Search over the NASA docs plus AI_COMPLETE, with a template fallback."""

import json

import config

LLM_MODEL = "claude-sonnet-4-5"
SEARCH_SERVICE = "PDM.PUBLIC.PDM_DOCS_SEARCH"

_RESPONSE_SCHEMA = {
    "type": "json",
    "schema": {
        "type": "object",
        "properties": {
            "why": {"type": "string", "description": "Why this engine is predicted to fail, in one or two sentences."},
            "root_cause": {"type": "string", "description": "Most likely degrading module and the sensor evidence for it."},
            "recommended_action": {"type": "string", "description": "The single most useful action to take now."},
            "confidence_note": {"type": "string", "description": "One-line caveat on how certain the prediction is."},
            "citations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Source passages used, as 'page N: short quote'.",
            },
        },
        "required": ["why", "root_cause", "recommended_action", "confidence_note", "citations"],
    },
}

_PROMPT_TEMPLATE = """You are a turbofan-engine reliability engineer. Using ONLY the engine data
and the NASA reference passages below, explain this engine's condition. Ground the
root cause in the reference material and cite the pages you used. If the passages
don't support a claim, say so rather than guessing.

ENGINE DATA
Engine: {machine_id} (fleet {fleet}: {operating_conditions} operating condition(s); fault modes in this fleet: {fault_modes})
Cycles flown: {cycles}
Predicted remaining useful life: {rul:.0f} cycles (90% prediction interval {rul_lower:.0f}-{rul_upper:.0f};
  predictions are capped at {rul_cap}, so {rul_cap} means "{rul_cap} or more")
Calibrated probability of failure within {horizon} cycles: {failure_probability:.0%}
Risk class: {risk_class}
Fastest-drifting sensors over the last 15 cycles (change in standard deviations per cycle,
relative to the engine's operating regime):
{drifting_sensors}

NASA REFERENCE PASSAGES (Saxena et al. 2008 and the C-MAPSS readme)
{passages}
"""

_ACTION_BY_RISK = {
    "CRITICAL": "Create a work order now and schedule repair within 24 hours.",
    "HIGH": "Schedule repair in the next maintenance window.",
    "MEDIUM": "Order replacement parts and monitor closely.",
    "LOW": "No action needed; continue routine monitoring.",
}


def top_drifting_sensors(feature_row, n=3) -> list[tuple[str, float]]:
    """(sensor, slope) for the sensors whose regime-normalized readings are changing fastest."""
    slope_cols = [c for c in feature_row.index if c.endswith("_SLOPE")]
    ranked = sorted(slope_cols, key=lambda c: abs(feature_row[c]), reverse=True)[:n]
    return [(c.replace("_SLOPE", ""), float(feature_row[c])) for c in ranked]


def describe_sensor(sensor: str) -> str:
    return f"{sensor} ({config.SENSOR_DESCRIPTIONS.get(sensor, 'unknown')})"


def _template_reasoning(machine_id, prediction_row, feature_row) -> dict:
    sensors = top_drifting_sensors(feature_row)
    width = prediction_row["RUL_UPPER"] - prediction_row["RUL_LOWER"]
    return {
        "why": (
            f"Predicted RUL has dropped to {prediction_row['RUL_PREDICTION']:.0f} cycles "
            f"with a {prediction_row['FAILURE_PROBABILITY']:.0%} chance of failure in the next "
            f"{config.FAILURE_HORIZON_CYCLES} cycles."
        ),
        "root_cause": (
            "Sustained drift in "
            + ", ".join(f"{describe_sensor(s)} at {slope:+.2f} σ/cycle" for s, slope in sensors)
            + f"; fleet fault modes: {prediction_row.get('FAULT_MODES', 'unknown')}."
        ),
        "recommended_action": _ACTION_BY_RISK.get(prediction_row["RISK_CLASS"], "Monitor."),
        "confidence_note": (
            f"90% interval: {prediction_row['RUL_LOWER']:.0f}-{prediction_row['RUL_UPPER']:.0f} cycles "
            f"({'narrow — act on it' if width <= 30 else 'wide — treat as directional and keep monitoring'})."
        ),
        "citations": [],
        "source": "template",
    }


def search_documentation(query: str, limit: int = 4) -> list[dict]:
    """Top Cortex Search passages from the NASA documentation."""
    from src.connection import get_session

    request = {"query": query, "columns": ["CHUNK", "DOC_NAME", "PAGE"], "limit": limit}
    raw = get_session().sql(
        "SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(?, ?)", params=[SEARCH_SERVICE, json.dumps(request)]
    ).collect()[0][0]
    return json.loads(raw)["results"]


def build_prompt(machine_id, prediction_row, feature_row, passages: list[dict]) -> str:
    sensors = top_drifting_sensors(feature_row)
    return _PROMPT_TEMPLATE.format(
        machine_id=machine_id,
        fleet=prediction_row.get("FLEET", "unknown"),
        operating_conditions=prediction_row.get("OPERATING_CONDITIONS", "unknown"),
        fault_modes=prediction_row.get("FAULT_MODES", "unknown"),
        cycles=prediction_row.get("CYCLES_OBSERVED", "unknown"),
        rul=prediction_row["RUL_PREDICTION"],
        rul_lower=prediction_row["RUL_LOWER"],
        rul_upper=prediction_row["RUL_UPPER"],
        rul_cap=config.RUL_CAP,
        horizon=config.FAILURE_HORIZON_CYCLES,
        failure_probability=prediction_row["FAILURE_PROBABILITY"],
        risk_class=prediction_row["RISK_CLASS"],
        drifting_sensors="\n".join(f"- {describe_sensor(s)}: {slope:+.3f}" for s, slope in sensors),
        passages="\n\n".join(f"[page {p['PAGE']}] {p['CHUNK']}" for p in passages),
    )


def _cortex_reasoning(machine_id, prediction_row, feature_row) -> dict:
    from src.connection import get_session

    sensors = top_drifting_sensors(feature_row)
    query = (
        f"degradation of {prediction_row.get('FAULT_MODES', 'engine modules')}: what drift in "
        + ", ".join(config.SENSOR_DESCRIPTIONS.get(s, s) for s, _ in sensors)
        + " indicates, and how failure is defined"
    )
    passages = search_documentation(query)
    raw = get_session().sql(
        "SELECT AI_COMPLETE(model => ?, prompt => ?, response_format => PARSE_JSON(?))",
        params=[LLM_MODEL, build_prompt(machine_id, prediction_row, feature_row, passages), json.dumps(_RESPONSE_SCHEMA)],
    ).collect()[0][0]
    reasoning = json.loads(raw)
    reasoning["source"] = f"Cortex AI_COMPLETE ({LLM_MODEL}) + Cortex Search"
    return reasoning


def explain(machine_id, prediction_row, feature_row) -> dict:
    if config.SNOWFLAKE_MODE == "snowflake":
        try:
            return _cortex_reasoning(machine_id, prediction_row, feature_row)
        except Exception as e:
            # fall back if Cortex isn't available
            reasoning = _template_reasoning(machine_id, prediction_row, feature_row)
            reasoning["source"] = f"template (Cortex unavailable: {e})"
            return reasoning
    return _template_reasoning(machine_id, prediction_row, feature_row)
