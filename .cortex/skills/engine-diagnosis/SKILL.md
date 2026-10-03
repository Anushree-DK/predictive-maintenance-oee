---
name: engine-diagnosis
description: "Explain why one turbofan engine is predicted to fail: model drivers (SHAP), sensor drift, and the physics from NASA's C-MAPSS documentation, with citations. Use when: the user asks why an engine is at risk, to diagnose or explain an engine, or what is degrading. Triggers: diagnose, explain engine, why is engine, root cause, what is wrong with."
---

# Engine diagnosis

Explain one engine's prediction from data in PDM.PUBLIC, grounded in NASA's documentation.

## Input
An in-service engine ID, e.g. FD001-ENG-034.

## Steps
1. Prediction and context:

```sql
SELECT p.*, m.FLEET, m.OPERATING_CONDITIONS, m.FAULT_MODES
FROM PDM.PUBLIC.FLEET_PREDICTIONS p JOIN PDM.PUBLIC.MACHINE_DATA m USING (MACHINE_ID)
WHERE p.MACHINE_ID = '<engine>';
```

2. What drives the prediction (SHAP, in cycles of RUL; negative pulls RUL down):

```sql
SELECT SOURCE, DESCRIPTION, CONTRIBUTION_CYCLES, BASE_RUL
FROM PDM.PUBLIC.ENGINE_RUL_DRIVERS
WHERE MACHINE_ID = '<engine>'
ORDER BY ABS(CONTRIBUTION_CYCLES) DESC
LIMIT 5;
```

3. Physics from NASA's paper, through the Cortex Search service (put the top drivers' sensor
   names and the fleet's fault modes in the query):

```sql
SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
  'PDM.PUBLIC.PDM_DOCS_SEARCH',
  '{"query": "<fault modes> degradation: what rising/falling <sensor names> indicates", "columns": ["CHUNK", "PAGE"], "limit": 3}'
);
```

## Output
- **Prediction:** RUL with 90% interval, risk class, chance of failure within 30 cycles.
- **Drivers:** the top 3 sensors with their contribution in cycles and what each measures.
- **Physics:** two or three sentences on the likely degrading module, citing the paper's page numbers
  from step 3.
- **Recommendation:** inspection or repair, and how urgent, from the risk class.
Only state what the queries and passages support.
