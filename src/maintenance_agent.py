"""Client for MAINTENANCE_AGENT, the Cortex Agent defined in
sql/006_maintenance_agent.sql, through the Cortex Agents Run API.

The agent plans its own tool calls — Cortex Analyst over the fleet's semantic view,
Cortex Search over the NASA documentation, and DRAFT_WORK_ORDER — and answers with
citations. ask() returns the answer plus a trace of what the agent did, so the
dashboard can show its work.
"""

import json
import re

import requests

import config

AGENT_PATH = "/api/v2/databases/{db}/schemas/{schema}/agents/MAINTENANCE_AGENT:run"
TIMEOUT_SECONDS = 180


def _user_message(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _assistant_message(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def parse_response(payload: dict) -> dict:
    """Final Run API payload -> answer text, tool trace, result tables, citations."""
    answer, steps, tables, citations = [], [], [], []
    for item in payload.get("content", []):
        kind = item.get("type")
        if kind == "text":
            answer.append(item.get("text", ""))
            for note in item.get("annotations") or []:
                title = note.get("doc_title")
                if title and title not in citations:
                    citations.append(title)
        elif kind == "tool_use":
            use = item["tool_use"]
            tool_input = use.get("input", {})
            if use.get("type") == "system_agentic_semantic_context":
                continue  # the agent reading the semantic model's schema, not a user-visible step
            steps.append({
                "tool": use.get("name"),
                "sql": tool_input.get("sql"),
                "query": tool_input.get("query"),
                "input": tool_input,
            })
        elif kind == "table":
            result = item["table"].get("result_set", {})
            columns = [col["name"] for col in result.get("resultSetMetaData", {}).get("rowType", [])]
            tables.append({"columns": columns, "rows": result.get("data", [])})
    text = re.sub(r"</?answer>", "", "\n".join(answer)).strip()  # the model sometimes wraps its reply
    return {"answer": text, "steps": steps, "tables": tables, "citations": citations}


def ask(question: str, history: list[dict] | None = None) -> dict:
    """history: earlier turns as [{"role": "user"|"assistant", "text": ...}]."""
    if config.SNOWFLAKE_MODE != "snowflake":
        raise RuntimeError("The maintenance agent runs in Snowflake — set SNOWFLAKE_MODE=snowflake.")
    from src.connection import get_session

    messages = [
        _user_message(turn["text"]) if turn["role"] == "user" else _assistant_message(turn["text"])
        for turn in history or []
    ] + [_user_message(question)]

    conn = get_session().connection
    path = AGENT_PATH.format(
        db=config.SNOWFLAKE_CONNECTION_PARAMS["database"], schema=config.SNOWFLAKE_CONNECTION_PARAMS["schema"]
    )
    response = requests.post(
        f"https://{conn.host}{path}",
        headers={
            "Authorization": f'Snowflake Token="{conn.rest.token}"',
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={"messages": messages, "stream": False},
        timeout=TIMEOUT_SECONDS,
    )
    if not response.ok:
        raise RuntimeError(f"Agent call failed ({response.status_code}): {response.text[:500]}")
    return parse_response(response.json())


if __name__ == "__main__":
    import sys

    result = ask(" ".join(sys.argv[1:]) or "Which engines are most likely to fail in the next 30 cycles?")
    print(result["answer"])
    print(json.dumps(result["steps"], indent=1)[:2000])
