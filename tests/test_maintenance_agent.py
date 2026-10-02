from src.maintenance_agent import parse_response

# Shape of a Cortex Agents Run API (stream: false) response, trimmed to one of each item.
PAYLOAD = {
    "role": "assistant",
    "content": [
        {"type": "tool_use", "tool_use": {"name": "fleet_analyst", "type": "system_agentic_semantic_context",
                                          "input": {"pruning_question": "..."}}},
        {"type": "tool_use", "tool_use": {"name": "system_execute_sql", "type": "system_execute_sql",
                                          "input": {"sql": "SELECT machine_id FROM __prediction"}}},
        {"type": "tool_use", "tool_use": {"name": "nasa_docs", "type": "cortex_search",
                                          "input": {"query": "HPC degradation sensors"}}},
        {"type": "tool_use", "tool_use": {"name": "draft_work_order", "type": "generic",
                                          "input": {"p_machine_id": "FD001-ENG-034", "p_priority": "P1"}}},
        {"type": "table", "table": {"result_set": {
            "data": [["FD001-ENG-034", "4.1"]],
            "resultSetMetaData": {"rowType": [{"name": "MACHINE_ID"}, {"name": "RUL_PREDICTION"}]},
        }}},
        {"type": "text", "text": "<answer>Drafted a P1 inspection.</answer>",
         "annotations": [{"doc_title": "Saxena et al. 2008"}, {"doc_title": "Saxena et al. 2008"}]},
    ],
}


def test_answer_text_is_unwrapped():
    assert parse_response(PAYLOAD)["answer"] == "Drafted a P1 inspection."


def test_steps_skip_schema_lookups_and_keep_sql_search_and_tools():
    steps = parse_response(PAYLOAD)["steps"]
    assert [s["tool"] for s in steps] == ["system_execute_sql", "nasa_docs", "draft_work_order"]
    assert steps[0]["sql"].startswith("SELECT")
    assert steps[1]["query"] == "HPC degradation sensors"
    assert steps[2]["input"]["p_machine_id"] == "FD001-ENG-034"


def test_tables_and_deduplicated_citations():
    result = parse_response(PAYLOAD)
    assert result["tables"] == [{"columns": ["MACHINE_ID", "RUL_PREDICTION"], "rows": [["FD001-ENG-034", "4.1"]]}]
    assert result["citations"] == ["Saxena et al. 2008"]


def test_empty_payload():
    assert parse_response({}) == {"answer": "", "steps": [], "tables": [], "citations": []}
