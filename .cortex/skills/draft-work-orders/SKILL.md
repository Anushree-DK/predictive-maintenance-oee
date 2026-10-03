---
name: draft-work-orders
description: "Draft maintenance work orders for at-risk turbofan engines with the DRAFT_WORK_ORDER procedure; drafts wait for a planner to approve them in the command center. Use when: the user asks to create, draft or raise work orders, or schedule maintenance for engines. Triggers: work order, draft work order, schedule maintenance, raise a ticket, send to the shop."
---

# Draft work orders

Create work orders in PDM.PUBLIC.WORK_ORDERS. Drafts only: never approve or reject them; a
planner does that in the command center.

## Input
One or more in-service engine IDs (for example from the fleet-risk-triage skill).

## Steps
1. For each engine, read its current prediction from PDM.PUBLIC.FLEET_PREDICTIONS (RISK_CLASS,
   RUL_PREDICTION, RUL_LOWER, FAILURE_PROBABILITY) and its top drifting sensor from
   PDM.PUBLIC.ENGINE_DEGRADATION_SIGNALS (DRIFT_RANK = 1).
2. Choose:
   - priority: P1 for CRITICAL, P2 for HIGH, P3 otherwise;
   - work type: INSPECTION for CRITICAL or HIGH, PARTS_ORDER for MEDIUM, otherwise skip the engine.
3. Call, once per engine:

```sql
CALL PDM.PUBLIC.DRAFT_WORK_ORDER('<engine>', '<INSPECTION|REPAIR|PARTS_ORDER>', '<P1|P2|P3>',
  '<justification: risk class, RUL and interval, failure probability, top drifting sensor>');
```

4. Show the queue:

```sql
SELECT MACHINE_ID, WORK_TYPE, PRIORITY, STATUS, RISK_CLASS_AT_DRAFT, RUL_PREDICTION_AT_DRAFT, CREATED_AT
FROM PDM.PUBLIC.WORK_ORDERS WHERE STATUS = 'PENDING_APPROVAL' ORDER BY CREATED_AT DESC;
```

## Output
The procedure's message for each engine, then the pending queue. End by saying the drafts are
waiting for approval in the command center's "Work orders awaiting approval" panel.
