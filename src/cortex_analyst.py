"""Natural-language Q&A over the business tables (MACHINE_DATA,
MAINTENANCE_HISTORY, PRODUCTION_DATA, ACTION_OUTCOMES) via Cortex Analyst's
REST API, using the Semantic View defined in sql/003_semantic_model.yaml.

Built from Snowflake's official Cortex Analyst REST API docs, but not yet
tested against a live call — no Snowflake account with Cortex Analyst access
has been available while building this (see README "Switching to real
Snowflake"). Every failure mode here is caught and reported clearly rather
than crashing the dashboard, the same pattern src/decision_layer.py uses for
Cortex COMPLETE — so a wrong assumption here fails safely, visibly, and
doesn't take down the rest of the app.
"""

import requests

import config

SEMANTIC_VIEW = "PDM.PUBLIC.PM_SEMANTIC_VIEW"


def ask(question: str) -> dict:
    """Returns {"answer": str, "sql": str | None} or raises with a clear message."""
    if config.SNOWFLAKE_MODE != "snowflake":
        raise RuntimeError("Cortex Analyst needs SNOWFLAKE_MODE=snowflake.")

    from src.connection import get_session

    session = get_session()
    conn = session.connection
    account = config.SNOWFLAKE_CONNECTION_PARAMS["account"]
    url = f"https://{account}.snowflakecomputing.com/api/v2/cortex/analyst/message"

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {conn.rest.token}",
            "X-Snowflake-Authorization-Token-Type": "SNOWFLAKE_SESSION_TOKEN",
            "Content-Type": "application/json",
        },
        json={
            "messages": [{"role": "user", "content": [{"type": "text", "text": question}]}],
            "semantic_view": SEMANTIC_VIEW,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    answer_parts, sql = [], None
    for block in payload.get("message", {}).get("content", []):
        if block.get("type") == "text":
            answer_parts.append(block["text"])
        elif block.get("type") == "sql":
            sql = block.get("statement")

    return {"answer": " ".join(answer_parts) or "(no text response)", "sql": sql}
