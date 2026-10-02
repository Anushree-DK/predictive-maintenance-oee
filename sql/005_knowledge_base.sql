-- Agent knowledge base: the NASA C-MAPSS paper and readme, indexed by Cortex Search

USE SCHEMA PDM.PUBLIC;

CREATE OR REPLACE TABLE DOC_CHUNKS (
    DOC_NAME   STRING,
    PAGE       NUMBER,
    CHUNK_ID   NUMBER,
    CHUNK      STRING
);

-- Parse the paper page by page and split it into chunks
INSERT INTO DOC_CHUNKS
WITH parsed AS (
    SELECT AI_PARSE_DOCUMENT(
        TO_FILE('@DOCS_STAGE', 'Damage Propagation Modeling.pdf'),
        {'mode': 'LAYOUT', 'page_split': true}
    ) AS DOC
),
pages AS (
    SELECT p.INDEX + 1 AS PAGE, p.VALUE:content::STRING AS CONTENT
    FROM parsed, LATERAL FLATTEN(INPUT => DOC:pages) p
)
SELECT
    'Saxena et al. 2008 - Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation (NASA)',
    PAGE,
    c.INDEX,
    c.VALUE::STRING
FROM pages,
     LATERAL FLATTEN(INPUT => SNOWFLAKE.CORTEX.SPLIT_TEXT_RECURSIVE_CHARACTER(CONTENT, 'markdown', 1500, 200)) c;

-- The readme as a single chunk
CREATE OR REPLACE FILE FORMAT TEXT_LINES
    TYPE = CSV FIELD_DELIMITER = NONE RECORD_DELIMITER = '\n'
    ESCAPE_UNENCLOSED_FIELD = NONE ENCODING = 'WINDOWS1252';

INSERT INTO DOC_CHUNKS
SELECT 'NASA C-MAPSS dataset readme', 1, 0,
       LISTAGG(COALESCE($1, ''), '\n') WITHIN GROUP (ORDER BY METADATA$FILE_ROW_NUMBER)
FROM @DOCS_STAGE/readme.txt (FILE_FORMAT => 'TEXT_LINES');

CREATE OR REPLACE CORTEX SEARCH SERVICE PDM_DOCS_SEARCH
    ON CHUNK
    ATTRIBUTES DOC_NAME, PAGE
    WAREHOUSE = COMPUTE_WH
    TARGET_LAG = '7 days'
    AS SELECT CHUNK, DOC_NAME, PAGE FROM DOC_CHUNKS;
