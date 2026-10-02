-- Creates the Cortex Analyst Semantic View PDM.PUBLIC.PM_SEMANTIC_VIEW from
-- sql/003_semantic_model.yaml. `python scripts/deploy_snowflake.py agent` does this
-- (and creates the maintenance agent); the SQL below is the manual equivalent.
-- Needs SNOWFLAKE.CORTEX_USER (or CORTEX_ANALYST_USER) on the role, and Cortex
-- Analyst in-region or cross-region inference enabled
-- (ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION'; ACCOUNTADMIN only).

USE SCHEMA PDM.PUBLIC;

-- Validate only (third argument TRUE), then create:
-- CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML('PDM.PUBLIC', $$<contents of 003_semantic_model.yaml>$$, TRUE);
-- CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML('PDM.PUBLIC', $$<contents of 003_semantic_model.yaml>$$);

-- Verify:
-- SHOW SEMANTIC VIEWS IN SCHEMA PDM.PUBLIC;
-- DESCRIBE SEMANTIC VIEW PDM.PUBLIC.PM_SEMANTIC_VIEW;
