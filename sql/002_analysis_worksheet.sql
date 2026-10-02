-- Business-question analysis worksheet for the Predictive Maintenance & OEE
-- Command Center. Paste into a Snowsight Worksheet against the PDM database —
-- each query stands alone and answers one question a fleet or reliability
-- manager would actually ask. All data is real NASA C-MAPSS (see DATA_SOURCES.md).

USE DATABASE PDM;
USE SCHEMA PUBLIC;

-- 1. How long do engines in each fleet last before failing?
--    (real run-to-failure events: the reliability baseline the model improves on)
SELECT
    m.FLEET,
    m.FAULT_MODES,
    COUNT(*)                          AS FAILURES_RECORDED,
    MIN(h.EVENT_CYCLE)                AS SHORTEST_LIFE_CYCLES,
    ROUND(AVG(h.EVENT_CYCLE), 1)      AS AVG_LIFE_CYCLES,
    MAX(h.EVENT_CYCLE)                AS LONGEST_LIFE_CYCLES
FROM MAINTENANCE_HISTORY h
JOIN MACHINE_DATA m ON m.MACHINE_ID = h.MACHINE_ID
GROUP BY m.FLEET, m.FAULT_MODES
ORDER BY AVG_LIFE_CYCLES;


-- 2. Lifetime OEE by fleet (same formula as src/oee.py; 2.0 = HOURS_PER_CYCLE and
--    18 = AVG_UNPLANNED_REPAIR_HOURS from config.py).
WITH scored AS (
    SELECT
        m.FLEET,
        (p.CYCLES_FLOWN * 2.0)
            / (p.CYCLES_FLOWN * 2.0 + CASE WHEN p.FAILURE_FLAG THEN 18 ELSE 0 END) AS AVAILABILITY,
        LEAST(GREATEST(p.AVG_HEALTH_INDEX, 0), 1)                               AS PERFORMANCE,
        p.IN_SPEC_CYCLES / p.CYCLES_FLOWN                                        AS QUALITY
    FROM OPERATING_PERIODS p
    JOIN MACHINE_DATA m ON m.MACHINE_ID = p.MACHINE_ID
)
SELECT
    FLEET,
    ROUND(AVG(AVAILABILITY), 3)                          AS AVG_AVAILABILITY,
    ROUND(AVG(PERFORMANCE), 3)                           AS AVG_PERFORMANCE,
    ROUND(AVG(QUALITY), 3)                               AS AVG_QUALITY,
    ROUND(AVG(AVAILABILITY * PERFORMANCE * QUALITY), 3)  AS AVG_OEE
FROM scored
GROUP BY FLEET
ORDER BY AVG_OEE;


-- 3. Which in-service engines are drifting hardest right now on SENSOR_11
--    (HPC static pressure, Ps30 — the classic HPC-degradation signal)?
--    Same closed-form slope as src/feature_engineering.py, over the last 15 cycles.
WITH recent AS (
    SELECT r.MACHINE_ID, r.TIME_CYCLE, r.SENSOR_11
    FROM RAW_SENSOR_DATA r
    JOIN MACHINE_DATA m ON m.MACHINE_ID = r.MACHINE_ID AND m.STATUS = 'IN_SERVICE'
    QUALIFY ROW_NUMBER() OVER (PARTITION BY r.MACHINE_ID ORDER BY r.TIME_CYCLE DESC) <= 15
)
SELECT
    MACHINE_ID,
    (COUNT(*) * SUM(TIME_CYCLE * SENSOR_11) - SUM(TIME_CYCLE) * SUM(SENSOR_11))
        / NULLIF(COUNT(*) * SUM(TIME_CYCLE * TIME_CYCLE) - SUM(TIME_CYCLE) * SUM(TIME_CYCLE), 0)
        AS SENSOR_11_SLOPE
FROM recent
GROUP BY MACHINE_ID
ORDER BY ABS(SENSOR_11_SLOPE) DESC
LIMIT 10;
-- Swap SENSOR_11 for any of SENSOR_1..SENSOR_21. Single-condition fleets (FD001/FD003)
-- compare directly; for FD002/FD004 compare within an operating regime.


-- 4. In-service engines in the worst recent condition: lowest health index and
--    in-spec rate over their last 3 periods.
WITH recent AS (
    SELECT p.*
    FROM OPERATING_PERIODS p
    JOIN MACHINE_DATA m ON m.MACHINE_ID = p.MACHINE_ID AND m.STATUS = 'IN_SERVICE'
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.MACHINE_ID ORDER BY p.PERIOD_INDEX DESC) <= 3
)
SELECT
    MACHINE_ID,
    MAX(END_CYCLE)                                  AS CYCLES_FLOWN,
    ROUND(AVG(AVG_HEALTH_INDEX), 3)                 AS RECENT_HEALTH_INDEX,
    ROUND(SUM(IN_SPEC_CYCLES) / SUM(CYCLES_FLOWN), 3) AS RECENT_IN_SPEC_RATE
FROM recent
GROUP BY MACHINE_ID
ORDER BY RECENT_HEALTH_INDEX, RECENT_IN_SPEC_RATE
LIMIT 10;


-- 5. Agentic action effectiveness: of the actions the AI recommended and a
--    human took, how many prevented (vs. didn't prevent) a failure?
SELECT
    ACTION_TYPE,
    COUNT(*) AS TOTAL_ACTIONS,
    SUM(CASE WHEN FAILURE_OCCURRED_FLAG = FALSE THEN 1 ELSE 0 END) AS FAILURE_AVOIDED,
    SUM(CASE WHEN FAILURE_OCCURRED_FLAG = TRUE THEN 1 ELSE 0 END)  AS FAILURE_OCCURRED_ANYWAY,
    SUM(CASE WHEN FAILURE_OCCURRED_FLAG IS NULL THEN 1 ELSE 0 END) AS OUTCOME_PENDING
FROM ACTION_OUTCOMES
GROUP BY ACTION_TYPE
ORDER BY TOTAL_ACTIONS DESC;
