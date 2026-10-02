-- The maintenance agent: a Cortex Agent that answers fleet questions (Cortex Analyst
-- over PM_SEMANTIC_VIEW), explains engineering background with citations (Cortex
-- Search over the NASA documentation), and drafts work orders (DRAFT_WORK_ORDER).
-- It can only DRAFT: every work order waits for a human to approve or reject it in
-- the Command Center (src/work_orders.py). Deployed by
-- `python scripts/deploy_snowflake.py agent` after sql/005_knowledge_base.sql.

USE SCHEMA PDM.PUBLIC;

CREATE TABLE IF NOT EXISTS WORK_ORDERS (
    WORK_ORDER_ID                STRING PRIMARY KEY,
    MACHINE_ID                   STRING,
    WORK_TYPE                    STRING,   -- INSPECTION | REPAIR | PARTS_ORDER
    PRIORITY                     STRING,   -- P1 (within 24h) | P2 (next window) | P3 (routine)
    JUSTIFICATION                STRING,
    STATUS                       STRING,   -- PENDING_APPROVAL | APPROVED | REJECTED
    CREATED_AT                   TIMESTAMP_NTZ,
    CREATED_BY                   STRING,
    RUL_PREDICTION_AT_DRAFT      FLOAT,
    RUL_LOWER_AT_DRAFT           FLOAT,
    FAILURE_PROBABILITY_AT_DRAFT FLOAT,
    RISK_CLASS_AT_DRAFT          STRING,
    MODEL_VERSION                STRING,
    REVIEWED_AT                  TIMESTAMP_NTZ,
    REVIEWED_BY                  STRING,
    ACTION_ID                    STRING    -- ACTION_OUTCOMES row written on approval
);

-- Each in-service engine's three fastest-drifting sensors, with what they measure
-- (SENSOR_REFERENCE, from Table 2 of the NASA paper; loaded by the deploy script from
-- config.SENSOR_DESCRIPTIONS). Gives the agent per-engine evidence without exposing
-- the 85-column feature table.
CREATE OR REPLACE VIEW ENGINE_DEGRADATION_SIGNALS AS
WITH slopes AS (
    SELECT MACHINE_ID, SENSOR_COLUMN, SLOPE
    FROM ENGINE_FEATURES
    UNPIVOT (SLOPE FOR SENSOR_COLUMN IN (
            SENSOR_1_SLOPE,
            SENSOR_2_SLOPE,
            SENSOR_3_SLOPE,
            SENSOR_4_SLOPE,
            SENSOR_5_SLOPE,
            SENSOR_6_SLOPE,
            SENSOR_7_SLOPE,
            SENSOR_8_SLOPE,
            SENSOR_9_SLOPE,
            SENSOR_10_SLOPE,
            SENSOR_11_SLOPE,
            SENSOR_12_SLOPE,
            SENSOR_13_SLOPE,
            SENSOR_14_SLOPE,
            SENSOR_15_SLOPE,
            SENSOR_16_SLOPE,
            SENSOR_17_SLOPE,
            SENSOR_18_SLOPE,
            SENSOR_19_SLOPE,
            SENSOR_20_SLOPE,
            SENSOR_21_SLOPE
    ))
)
SELECT
    s.MACHINE_ID,
    REPLACE(s.SENSOR_COLUMN, '_SLOPE', '') AS SENSOR,
    r.SYMBOL,
    r.DESCRIPTION,
    ROUND(s.SLOPE, 4) AS SLOPE_SIGMA_PER_CYCLE,
    ROW_NUMBER() OVER (PARTITION BY s.MACHINE_ID ORDER BY ABS(s.SLOPE) DESC) AS DRIFT_RANK
FROM slopes s
JOIN SENSOR_REFERENCE r ON r.SENSOR = REPLACE(s.SENSOR_COLUMN, '_SLOPE', '')
QUALIFY DRIFT_RANK <= 3;

-- Agent tool. Validates its inputs and snapshots the engine's current prediction,
-- so the draft records exactly what the agent saw.
CREATE OR REPLACE PROCEDURE DRAFT_WORK_ORDER(
    P_MACHINE_ID STRING, P_WORK_TYPE STRING, P_PRIORITY STRING, P_JUSTIFICATION STRING
)
RETURNS STRING
LANGUAGE SQL
EXECUTE AS CALLER
AS
$$
DECLARE
    wo_id STRING DEFAULT UUID_STRING();
    engines INTEGER;
    work_type STRING DEFAULT UPPER(TRIM(P_WORK_TYPE));
    priority STRING DEFAULT UPPER(TRIM(P_PRIORITY));
BEGIN
    SELECT COUNT(*) INTO :engines FROM FLEET_PREDICTIONS WHERE MACHINE_ID = :P_MACHINE_ID;
    IF (engines = 0) THEN
        RETURN 'No in-service engine with ID ' || P_MACHINE_ID || ' — no work order drafted.';
    END IF;
    IF (work_type NOT IN ('INSPECTION', 'REPAIR', 'PARTS_ORDER')) THEN
        RETURN 'Work type must be INSPECTION, REPAIR or PARTS_ORDER — no work order drafted.';
    END IF;
    IF (priority NOT IN ('P1', 'P2', 'P3')) THEN
        RETURN 'Priority must be P1, P2 or P3 — no work order drafted.';
    END IF;

    INSERT INTO WORK_ORDERS (
        WORK_ORDER_ID, MACHINE_ID, WORK_TYPE, PRIORITY, JUSTIFICATION, STATUS, CREATED_AT, CREATED_BY,
        RUL_PREDICTION_AT_DRAFT, RUL_LOWER_AT_DRAFT, FAILURE_PROBABILITY_AT_DRAFT, RISK_CLASS_AT_DRAFT, MODEL_VERSION
    )
    SELECT :wo_id, MACHINE_ID, :work_type, :priority, :P_JUSTIFICATION, 'PENDING_APPROVAL', SYSDATE(), 'MAINTENANCE_AGENT',
           RUL_PREDICTION, RUL_LOWER, FAILURE_PROBABILITY, RISK_CLASS, MODEL_VERSION
    FROM FLEET_PREDICTIONS
    WHERE MACHINE_ID = :P_MACHINE_ID;

    RETURN 'Drafted ' || priority || ' ' || work_type || ' work order ' || wo_id || ' for ' || P_MACHINE_ID
        || '. Status PENDING_APPROVAL: a maintenance planner must approve it in the Command Center.';
END;
$$;

CREATE OR REPLACE AGENT MAINTENANCE_AGENT
  COMMENT = 'Turbofan fleet maintenance agent: fleet analytics, NASA engineering references, work-order drafting'
  FROM SPECIFICATION
  $$
  models:
    orchestration: claude-sonnet-4-5

  orchestration:
    budget:
      seconds: 90
      tokens: 32000

  instructions:
    orchestration: >
      Use fleet_analyst for any question about engines, fleets, predictions (remaining useful
      life, failure probability, risk class), OEE, failure history or work orders.
      Use nasa_docs for engineering background: what a sensor measures, how HPC or fan
      degradation shows up, how failure is defined.
      Call draft_work_order only when the user asks to create, draft or raise a work order or
      to schedule maintenance. Pick the work type (INSPECTION, REPAIR, PARTS_ORDER) and priority
      (P1 = within 24 hours, for CRITICAL risk; P2 = next maintenance window, for HIGH;
      P3 = routine) from the engine's current prediction, which you must look up first with
      fleet_analyst, and put that evidence in the justification. Draft one work order per engine.
    response: >
      Be concise and specific: give engine IDs and numbers. RUL is in flight cycles; a predicted
      RUL of 125 means "125 or more". When you use nasa_docs, cite the page. You can draft work
      orders but never approve them: say that a planner must approve them in the Command Center.
    sample_questions:
      - question: "Which in-service engines are most likely to fail in the next 30 cycles?"
      - question: "Why does rising Ps30 indicate HPC degradation?"
      - question: "Draft work orders for the critical engines in fleet FD001."

  tools:
    - tool_spec:
        type: "cortex_analyst_text_to_sql"
        name: "fleet_analyst"
        description: >
          Answers questions about the turbofan fleets with SQL: engines and their status, latest
          predictions (RUL, interval, failure probability, risk class), OEE, real failure history,
          and work orders.
    - tool_spec:
        type: "cortex_search"
        name: "nasa_docs"
        description: >
          Searches NASA's C-MAPSS documentation (Saxena et al. 2008, Damage Propagation Modeling
          for Aircraft Engine Run-to-Failure Simulation, and the dataset readme): sensor
          definitions, degradation physics, the failure criterion.
    - tool_spec:
        type: "generic"
        name: "draft_work_order"
        description: >
          Drafts a maintenance work order for one in-service engine. It is created with status
          PENDING_APPROVAL; a human must approve it.
        input_schema:
          type: "object"
          properties:
            p_machine_id:
              type: "string"
              description: "In-service engine ID, e.g. FD001-ENG-031"
            p_work_type:
              type: "string"
              description: "INSPECTION, REPAIR or PARTS_ORDER"
            p_priority:
              type: "string"
              description: "P1 (within 24 hours), P2 (next maintenance window) or P3 (routine)"
            p_justification:
              type: "string"
              description: "Evidence for the work order: current prediction and the drifting sensors"
          required: ["p_machine_id", "p_work_type", "p_priority", "p_justification"]

  tool_resources:
    fleet_analyst:
      semantic_view: "PDM.PUBLIC.PM_SEMANTIC_VIEW"
      execution_environment:
        type: "warehouse"
        warehouse: "COMPUTE_WH"
    nasa_docs:
      search_service: "PDM.PUBLIC.PDM_DOCS_SEARCH"
      max_results: "4"
      title_column: "DOC_NAME"
    draft_work_order:
      type: "procedure"
      identifier: "PDM.PUBLIC.DRAFT_WORK_ORDER"
      execution_environment:
        type: "warehouse"
        warehouse: "COMPUTE_WH"
  $$;
