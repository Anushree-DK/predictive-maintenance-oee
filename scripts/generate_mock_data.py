"""Generates local CSVs shaped like the Snowflake tables in sql/001_create_tables.sql,
so the rest of the pipeline can run end-to-end before a real Snowflake account exists.

Sensor data follows the NASA C-MAPSS pattern: each machine runs from a healthy
state to failure, with a handful of sensors drifting as degradation progresses.
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

RNG = np.random.default_rng(42)
N_MACHINES = 8
DEGRADING_SENSORS = [2, 3, 4, 7, 11, 12, 15]  # subset of the 21 that trend with wear


def _machine_ids():
    return [f"M-{i:03d}" for i in range(1, N_MACHINES + 1)]


def generate_machine_data(machine_ids=None):
    machine_ids = machine_ids if machine_ids is not None else _machine_ids()
    rows = []
    lines = ["LINE-A", "LINE-B", "LINE-C"]
    models = ["TURBOFAN-X1", "TURBOFAN-X2"]
    for i, mid in enumerate(machine_ids):
        rows.append(
            {
                "MACHINE_ID": mid,
                "MACHINE_NAME": f"Compressor {i + 1}",
                "LINE": lines[i % len(lines)],
                "MODEL": models[i % len(models)],
                "INSTALL_DATE": date(2019, 1, 1) + timedelta(days=180 * i),
            }
        )
    return pd.DataFrame(rows)


def generate_sensor_data():
    all_rows = []
    for mid in _machine_ids():
        life = int(RNG.integers(120, 300))  # total cycles until failure
        current_cycle = int(RNG.integers(int(life * 0.5), life))  # where it is "today"
        baseline = RNG.normal(500, 20, size=21)
        for cycle in range(1, current_cycle + 1):
            frac = cycle / life  # 0 = healthy, 1 = failure
            row = {"MACHINE_ID": mid, "TIME_CYCLE": cycle}
            row["OP_SETTING_1"] = float(RNG.normal(0, 1))
            row["OP_SETTING_2"] = float(RNG.normal(0, 1))
            row["OP_SETTING_3"] = 100.0
            for s in range(1, 22):
                drift = (frac**2) * 15 if s in DEGRADING_SENSORS else 0
                noise = RNG.normal(0, 1.5)
                row[f"SENSOR_{s}"] = float(baseline[s - 1] + drift + noise)
            row["RECORDED_AT"] = datetime(2026, 1, 1) + timedelta(hours=cycle)
            all_rows.append(row)
    return pd.DataFrame(all_rows)


def generate_maintenance_history(machine_ids=None):
    machine_ids = machine_ids if machine_ids is not None else _machine_ids()
    rows = []
    event_types = ["PREVENTIVE", "INSPECTION", "CORRECTIVE"]
    eid = 1
    for mid in machine_ids:
        for _ in range(int(RNG.integers(2, 6))):
            rows.append(
                {
                    "EVENT_ID": f"EVT-{eid:05d}",
                    "MACHINE_ID": mid,
                    "EVENT_DATE": date(2025, 1, 1) + timedelta(days=int(RNG.integers(0, 600))),
                    "EVENT_TYPE": RNG.choice(event_types),
                    "DESCRIPTION": "Routine service" if RNG.random() > 0.3 else "Unplanned repair",
                    "DOWNTIME_MINUTES": int(RNG.integers(30, 480)),
                }
            )
            eid += 1
    return pd.DataFrame(rows)


def generate_production_data(machine_ids=None):
    machine_ids = machine_ids if machine_ids is not None else _machine_ids()
    rows = []
    for mid in machine_ids:
        for d in range(30):
            shift_date = date(2026, 8, 1) + timedelta(days=d)
            for shift in ["DAY", "NIGHT"]:
                planned = 480
                downtime = int(RNG.integers(0, 60))
                ideal_cycle = 12.0
                run_time_sec = (planned - downtime) * 60
                total_count = int(run_time_sec / ideal_cycle * RNG.uniform(0.85, 1.0))
                good_count = int(total_count * RNG.uniform(0.9, 0.995))
                rows.append(
                    {
                        "MACHINE_ID": mid,
                        "SHIFT_DATE": shift_date,
                        "SHIFT": shift,
                        "PLANNED_PRODUCTION_TIME_MIN": planned,
                        "DOWNTIME_MIN": downtime,
                        "IDEAL_CYCLE_TIME_SEC": ideal_cycle,
                        "TOTAL_COUNT": total_count,
                        "GOOD_COUNT": good_count,
                    }
                )
    return pd.DataFrame(rows)


def generate_training_run_to_failure(n_units=40):
    """Full run-to-failure histories (unlike the 'today' snapshot above) so
    src/ml/train.py has labeled RUL examples to learn from."""
    all_rows = []
    for u in range(1, n_units + 1):
        uid = f"TRAIN-{u:03d}"
        life = int(RNG.integers(120, 300))
        baseline = RNG.normal(500, 20, size=21)
        for cycle in range(1, life + 1):
            frac = cycle / life
            row = {"UNIT_ID": uid, "TIME_CYCLE": cycle, "LIFE": life, "RUL": life - cycle}
            for s in range(1, 22):
                drift = (frac**2) * 15 if s in DEGRADING_SENSORS else 0
                noise = RNG.normal(0, 1.5)
                row[f"SENSOR_{s}"] = float(baseline[s - 1] + drift + noise)
            all_rows.append(row)
    return pd.DataFrame(all_rows)


def main():
    config.MOCK_DATA_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "machine_data.csv": generate_machine_data(),
        "raw_sensor_data.csv": generate_sensor_data(),
        "maintenance_history.csv": generate_maintenance_history(),
        "production_data.csv": generate_production_data(),
        "training_run_to_failure.csv": generate_training_run_to_failure(),
    }
    for filename, df in datasets.items():
        path = config.MOCK_DATA_DIR / filename
        df.to_csv(path, index=False)
        print(f"wrote {len(df):>6} rows -> {path}")

    outcomes_path = config.MOCK_DATA_DIR / "action_outcomes.csv"
    if not outcomes_path.exists():
        pd.DataFrame(
            columns=[
                "ACTION_ID", "MACHINE_ID", "ACTION_TYPE", "RECOMMENDED_AT",
                "RUL_PREDICTION_AT_ACTION", "RISK_CLASS_AT_ACTION", "CONFIDENCE_AT_ACTION",
                "TAKEN_FLAG", "TAKEN_AT", "FAILURE_OCCURRED_FLAG", "FAILURE_DATE", "NOTES",
            ]
        ).to_csv(outcomes_path, index=False)
        print(f"wrote      0 rows -> {outcomes_path}")


if __name__ == "__main__":
    main()
