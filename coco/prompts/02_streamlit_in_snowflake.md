Build a Streamlit in Snowflake version of the fleet command center and deploy it to the
pdm connection (database PDM, schema PUBLIC, warehouse COMPUTE_WH).

Context: dashboard/app.py is the local Streamlit app. In Snowflake, all results already
exist as tables, kept up to date by SCORE_FLEET_TASK:
- FLEET_PREDICTIONS (MACHINE_ID, LAST_CYCLE, RUL_PREDICTION, RUL_LOWER, RUL_UPPER,
  FAILURE_PROBABILITY, RISK_CLASS, SCORED_AT, MODEL_VERSION)
- MACHINE_DATA (MACHINE_ID, FLEET, OPERATING_CONDITIONS, FAULT_MODES, STATUS, ...)
- ENGINE_RUL_DRIVERS (MACHINE_ID, BASE_RUL, SOURCE, CONTRIBUTION_CYCLES, DESCRIPTION, IMPACT_RANK)
- ENGINE_DEGRADATION_SIGNALS, RAW_SENSOR_DATA, MODEL_METRICS (METRICS is a VARIANT)
- WORK_ORDERS (drafted by the MAINTENANCE_AGENT; STATUS PENDING_APPROVAL / APPROVED / REJECTED)
- ACTION_OUTCOMES, SCORING_RUNS, REPLAY_QUEUE
- Procedures REPLAY_PREPARE(20), REPLAY_STEP(n), REPLAY_RESTORE()
Read src/work_orders.py for how approval works (approve writes an ACTION_OUTCOMES row with
the prediction snapshot and sets STATUS, REVIEWED_AT, REVIEWED_BY, ACTION_ID).

Build streamlit_app/streamlit_app.py (plus streamlit_app/environment.yml if packages are
needed, e.g. plotly) that uses snowflake.snowpark.context.get_active_session() and shows:
1. Fleet KPIs: engines in service, critical/high count, model version and scoring time,
   RUL RMSE from the latest MODEL_METRICS row.
2. A fleet table sorted by risk, filterable by FLEET.
3. Engine detail for a selected engine: prediction with interval, the top SHAP drivers
   from ENGINE_RUL_DRIVERS as a horizontal bar chart, and the raw history of its
   fastest-drifting sensor.
4. Work orders awaiting approval with Approve / Reject buttons, matching src/work_orders.py.
5. Live replay buttons (rewind 20 cycles, stream 1 cycle, restore) calling the procedures.
Keep the code short and readable, with brief comments only.

Deploy it: add a `streamlit` step to scripts/deploy_snowflake.py that creates a stage
STREAMLIT_STAGE, uploads the streamlit_app files with session.file.put (no compression,
overwrite), and runs CREATE OR REPLACE STREAMLIT PDM.PUBLIC.FLEET_COMMAND_CENTER with
MAIN_FILE 'streamlit_app.py' and QUERY_WAREHOUSE COMPUTE_WH. Run it with
`.venv\Scripts\python.exe scripts\deploy_snowflake.py streamlit`, then confirm with
SHOW STREAMLITS. Fix any errors until it deploys.

Rules: do not modify dashboard/app.py, anything under src/, sql/, tests/, or .env. Do not
drop, truncate or alter existing tables, and do not call the replay or approval
procedures yourself. Reply with what you built, the deploy result, and any limitation.
