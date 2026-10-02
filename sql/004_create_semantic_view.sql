-- Manual creation of PM_SEMANTIC_VIEW (deploy_snowflake.py agent does the same)

USE SCHEMA PDM.PUBLIC;

-- Validate only (third argument TRUE), then create:
-- CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML('PDM.PUBLIC', $$<contents of 003_semantic_model.yaml>$$, TRUE);
-- CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML('PDM.PUBLIC', $$<contents of 003_semantic_model.yaml>$$);

-- Verify:
-- SHOW SEMANTIC VIEWS IN SCHEMA PDM.PUBLIC;
-- DESCRIBE SEMANTIC VIEW PDM.PUBLIC.PM_SEMANTIC_VIEW;
