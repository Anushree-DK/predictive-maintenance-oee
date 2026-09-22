-- Creates the Cortex Analyst Semantic View from sql/003_semantic_model.yaml.
-- NOT YET RUN against any account — needs the correct hackathon Snowflake
-- account (see README "Switching to real Snowflake"). Cortex Analyst also
-- needs SNOWFLAKE.CORTEX_USER or SNOWFLAKE.CORTEX_ANALYST_USER on the role,
-- and is only natively available in a subset of regions (ap-northeast-1,
-- ap-southeast-2, us-east-1, us-west-2, eu-central-1, eu-west-1, and two
-- Azure regions) — elsewhere an ACCOUNTADMIN must enable cross-region
-- inference first: ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';

USE DATABASE PDM;
USE SCHEMA PUBLIC;

-- Option A (recommended, most reliable): Snowsight UI
--   Snowsight -> AI & ML -> Studio -> Semantic Views -> Create -> Import YAML
--   -> upload sql/003_semantic_model.yaml. The UI validates the schema as it
--   imports, which sidesteps any uncertainty about the exact stored-procedure
--   call signature below.

-- Option B: from SQL, via the YAML-import stored procedure. The exact call
-- signature for SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML was not independently
-- verified against live Snowflake for this project (no working account yet)
-- — check `DESCRIBE PROCEDURE SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML` or the
-- current docs before running, and fall back to Option A if this errors.
--
-- CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML(
--     'PDM.PUBLIC.PM_SEMANTIC_VIEW',
--     $$<paste the contents of sql/003_semantic_model.yaml here>$$
-- );

-- Once created, verify with:
-- SHOW SEMANTIC VIEWS IN SCHEMA PDM.PUBLIC;
-- DESCRIBE SEMANTIC VIEW PDM.PUBLIC.PM_SEMANTIC_VIEW;

-- src/cortex_analyst.py references it as "PDM.PUBLIC.PM_SEMANTIC_VIEW" —
-- update that string if you name it differently here.
