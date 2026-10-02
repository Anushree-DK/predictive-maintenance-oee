Let me read the file to find query 6.Running query 6 now.The query errors on `PREDICTED_RUL`. Let me check the actual columns in `FLEET_PREDICTIONS`.No `PREDICTED_RUL` column (it's `RUL_PREDICTION`) and no `STATUS` column. Let me check `FLEET_GROUND_TRUTH` and `MACHINE_DATA` for the status.`STATUS` is on `MACHINE_DATA`, not `FLEET_PREDICTIONS`. I need to fix the query: rename `fp.PREDICTED_RUL` to `fp.RUL_PREDICTION` and join through `MACHINE_DATA` for the `STATUS` filter.Now let me run the fixed query.Here are the results:

| RISK_CLASS | ENGINES | AVG_PREDICTED_RUL | AVG_TRUE_RUL | FAIL_IN_WINDOW_RATE |
|------------|---------|-------------------|--------------|---------------------|
| CRITICAL   | 122     | 13.0              | 13.1         | 0.795               |
| HIGH       | 151     | 45.5              | 46.5         | 0.483               |
| MEDIUM     | 271     | 94.5              | 100.4        | 0.458               |
| LOW        | 163     | 120.3             | 133.4        | NULL                |

The fixes were: `PREDICTED_RUL` → `RUL_PREDICTION` (actual column name) and the `STATUS` filter moved to a join on `MACHINE_DATA` where that column lives.

The model's CRITICAL class is well-calibrated (80% of engines actually fail within the 15-cycle window and predicted RUL closely matches true RUL), but the HIGH and MEDIUM classes capture under half their true failures within their windows, suggesting the risk thresholds for those tiers need tightening or the model under-predicts degradation for mid-range engines.