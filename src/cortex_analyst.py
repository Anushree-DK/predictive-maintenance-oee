"""Natural-language Q&A over the fleet (engines, predictions, degradation signals,
OEE, failure history, work orders, action outcomes) via Cortex Analyst's REST API,
using the Semantic View defined in sql/003_semantic_model.yaml. Verified live; the
dashboard catches and shows failures rather than crashing.
"""

import requests

import config

SEMANTIC_VIEW = "PDM.PUBLIC.PM_SEMANTIC_VIEW"


def ask(question: str) -> dict:
    """Returns {"answer": str, "sql": str | None} or raises with a clear message."""
    if config.SNOWFLAKE_MODE != "snowflake":
        raise RuntimeError("Cortex Analyst needs SNOWFLAKE_MODE=snowflake.")

    from src.connection import get_session

    conn = get_session().connection
    response = requests.post(
        f"https://{conn.host}/api/v2/cortex/analyst/message",
        headers={
            # A Snowpark session token goes in the legacy "Snowflake Token" scheme;
            # Bearer is for OAuth / key-pair JWT / programmatic access tokens.
            "Authorization": f'Snowflake Token="{conn.rest.token}"',
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
