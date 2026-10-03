---
name: fleet-risk-triage
description: "Rank the turbofan engines most at risk of failure, from the live predictions in PDM.PUBLIC. Use when: the user asks which engines need attention, are critical, are at risk, or will fail soon, overall or for one fleet (FD001-FD004). Triggers: triage, at-risk engines, critical engines, which engines, fleet risk, failing soon."
---

# Fleet risk triage

Rank in-service engines by failure risk using the predictions that Snowflake keeps current.

## Inputs
- Optional fleet: FD001, FD002, FD003 or FD004 (default: all fleets).
- Optional count (default 5).

## Steps
1. Run this on the active connection (database PDM, schema PUBLIC), filling in the fleet filter and limit:

```sql
SELECT p.MACHINE_ID, m.FLEET, p.RISK_CLASS, p.RUL_PREDICTION, p.RUL_LOWER, p.RUL_UPPER,
       ROUND(p.FAILURE_PROBABILITY * 100) AS FAIL_30_PCT, p.SCORED_AT, p.MODEL_VERSION
FROM PDM.PUBLIC.FLEET_PREDICTIONS p
JOIN PDM.PUBLIC.MACHINE_DATA m USING (MACHINE_ID)
WHERE m.STATUS = 'IN_SERVICE'
  -- AND m.FLEET = '<fleet>'
ORDER BY p.RUL_LOWER, p.FAILURE_PROBABILITY DESC
LIMIT <count>;
```

2. For those engines, get the fastest-drifting sensors:

```sql
SELECT MACHINE_ID, DRIFT_RANK, SYMBOL, DESCRIPTION, SLOPE_SIGMA_PER_CYCLE
FROM PDM.PUBLIC.ENGINE_DEGRADATION_SIGNALS
WHERE MACHINE_ID IN (<engine ids>)
ORDER BY MACHINE_ID, DRIFT_RANK;
```

## Output
A table: engine, fleet, risk class, predicted RUL with its 90% interval, chance of failure within
30 cycles, and the top drifting sensor (symbol and what it measures). Then one line naming the
model version and when the fleet was last scored. RUL is in flight cycles; 125 means "125 or more".
Do not invent numbers: report only what the queries return.
