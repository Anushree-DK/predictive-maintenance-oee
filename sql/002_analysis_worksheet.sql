-- Business-question analysis worksheet for the Predictive Maintenance & OEE
-- Command Center. Paste this into a new Snowsight Worksheet against the PDM
-- database — each query stands alone and answers one question a plant
-- manager or reliability engineer would actually ask.

USE DATABASE PDM;
USE SCHEMA PUBLIC;

-- 1. Which machines are consuming the most maintenance effort?
--    (a proxy for "which machines are already unreliable")
SELECT
    m.MACHINE_ID,
    m.MACHINE_NAME,
    m.LINE,
    COUNT(*) AS MAINTENANCE_EVENTS,
    SUM(h.DOWNTIME_MINUTES) AS TOTAL_DOWNTIME_MINUTES,
    SUM(CASE WHEN h.EVENT_TYPE = 'CORRECTIVE' THEN 1 ELSE 0 END) AS UNPLANNED_REPAIRS
FROM MAINTENANCE_HISTORY h
JOIN MACHINE_DATA m ON m.MACHINE_ID = h.MACHINE_ID
GROUP BY m.MACHINE_ID, m.MACHINE_NAME, m.LINE
ORDER BY TOTAL_DOWNTIME_MINUTES DESC;


-- 2. OEE by production line, most recent 30 days of shifts.
--    (Availability x Performance x Quality, same formula as src/oee.py)
WITH scored AS (
    SELECT
        p.MACHINE_ID,
        m.LINE,
        (p.PLANNED_PRODUCTION_TIME_MIN - p.DOWNTIME_MIN) / p.PLANNED_PRODUCTION_TIME_MIN AS AVAILABILITY,
        LEAST(
            (p.IDEAL_CYCLE_TIME_SEC * p.TOTAL_COUNT / 60.0)
                / NULLIF(p.PLANNED_PRODUCTION_TIME_MIN - p.DOWNTIME_MIN, 0),
            1.0
        ) AS PERFORMANCE,
        p.GOOD_COUNT / NULLIF(p.TOTAL_COUNT, 0) AS QUALITY
    FROM PRODUCTION_DATA p
    JOIN MACHINE_DATA m ON m.MACHINE_ID = p.MACHINE_ID
    WHERE p.SHIFT_DATE >= DATEADD(day, -30, CURRENT_DATE())
)
SELECT
    LINE,
    ROUND(AVG(AVAILABILITY), 3) AS AVG_AVAILABILITY,
    ROUND(AVG(PERFORMANCE), 3) AS AVG_PERFORMANCE,
    ROUND(AVG(QUALITY), 3) AS AVG_QUALITY,
    ROUND(AVG(AVAILABILITY * PERFORMANCE * QUALITY), 3) AS AVG_OEE
FROM scored
GROUP BY LINE
ORDER BY AVG_OEE ASC;


-- 3. Which sensors are drifting the hardest across the whole fleet right now?
--    (same closed-form slope formula as src/feature_engineering.py's
--    build_feature_table_sql, just aggregated fleet-wide instead of per machine)
WITH recent AS (
    SELECT *
    FROM RAW_SENSOR_DATA
    QUALIFY ROW_NUMBER() OVER (PARTITION BY MACHINE_ID ORDER BY TIME_CYCLE DESC) <= 15
),
per_machine_slope AS (
    SELECT
        MACHINE_ID,
        (COUNT(*) * SUM(TIME_CYCLE * SENSOR_9) - SUM(TIME_CYCLE) * SUM(SENSOR_9))
            / NULLIF(COUNT(*) * SUM(TIME_CYCLE * TIME_CYCLE) - SUM(TIME_CYCLE) * SUM(TIME_CYCLE), 0)
            AS SENSOR_9_SLOPE
    FROM recent
    GROUP BY MACHINE_ID
)
SELECT MACHINE_ID, SENSOR_9_SLOPE
FROM per_machine_slope
ORDER BY ABS(SENSOR_9_SLOPE) DESC
LIMIT 10;
-- Swap SENSOR_9 for any of SENSOR_1..SENSOR_21 to check a different channel.


-- 4. Agentic action effectiveness: of the actions the AI recommended and a
--    human took, how many actually prevented (vs. didn't prevent) a failure?
SELECT
    ACTION_TYPE,
    COUNT(*) AS TOTAL_ACTIONS,
    SUM(CASE WHEN FAILURE_OCCURRED_FLAG = FALSE THEN 1 ELSE 0 END) AS FAILURE_AVOIDED,
    SUM(CASE WHEN FAILURE_OCCURRED_FLAG = TRUE THEN 1 ELSE 0 END) AS FAILURE_OCCURRED_ANYWAY,
    SUM(CASE WHEN FAILURE_OCCURRED_FLAG IS NULL THEN 1 ELSE 0 END) AS OUTCOME_PENDING
FROM ACTION_OUTCOMES
GROUP BY ACTION_TYPE
ORDER BY TOTAL_ACTIONS DESC;


-- 5. Machines with high maintenance cost AND poor OEE — the fleet's worst
--    combination, and the best argument for predictive maintenance ROI.
WITH downtime AS (
    SELECT MACHINE_ID, SUM(DOWNTIME_MINUTES) AS MAINT_DOWNTIME_MIN
    FROM MAINTENANCE_HISTORY
    GROUP BY MACHINE_ID
),
oee AS (
    SELECT
        MACHINE_ID,
        AVG(
            ((PLANNED_PRODUCTION_TIME_MIN - DOWNTIME_MIN) / PLANNED_PRODUCTION_TIME_MIN)
            * LEAST((IDEAL_CYCLE_TIME_SEC * TOTAL_COUNT / 60.0)
                     / NULLIF(PLANNED_PRODUCTION_TIME_MIN - DOWNTIME_MIN, 0), 1.0)
            * (GOOD_COUNT / NULLIF(TOTAL_COUNT, 0))
        ) AS AVG_OEE
    FROM PRODUCTION_DATA
    GROUP BY MACHINE_ID
)
SELECT
    m.MACHINE_ID, m.MACHINE_NAME, m.LINE,
    d.MAINT_DOWNTIME_MIN,
    ROUND(o.AVG_OEE, 3) AS AVG_OEE
FROM MACHINE_DATA m
JOIN downtime d ON d.MACHINE_ID = m.MACHINE_ID
JOIN oee o ON o.MACHINE_ID = m.MACHINE_ID
ORDER BY d.MAINT_DOWNTIME_MIN DESC, o.AVG_OEE ASC
LIMIT 10;
