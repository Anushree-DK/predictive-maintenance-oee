"""Single read/write interface over either the local processed C-MAPSS CSVs or the
same tables in Snowflake, selected by SNOWFLAKE_MODE. Every other module in src/
goes through this file instead of touching files or Snowpark directly, so swapping
backends doesn't change pipeline code.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

import config


def _read_local(table_name: str) -> pd.DataFrame:
    path = config.LOCAL_DATA_DIR / f"{table_name.lower()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run: python scripts/load_cmapss.py")
    return pd.read_csv(path)


def _read_snowflake(table_name: str) -> pd.DataFrame:
    from src.connection import get_session

    return get_session().table(table_name).to_pandas()


def _read(table_name: str) -> pd.DataFrame:
    if config.SNOWFLAKE_MODE == "snowflake":
        return _read_snowflake(table_name)
    return _read_local(table_name)


def load_machine_data() -> pd.DataFrame:
    return _read("MACHINE_DATA")


def load_raw_sensor_data() -> pd.DataFrame:
    return _read("RAW_SENSOR_DATA")


def load_in_service_sensor_data() -> pd.DataFrame:
    if config.SNOWFLAKE_MODE == "snowflake":
        from src.connection import get_session

        return get_session().sql(
            "SELECT r.* FROM RAW_SENSOR_DATA r JOIN MACHINE_DATA m USING (MACHINE_ID) WHERE m.STATUS = 'IN_SERVICE'"
        ).to_pandas()
    machines = load_machine_data()
    in_service = machines.loc[machines["STATUS"] == "IN_SERVICE", "MACHINE_ID"]
    sensors = load_raw_sensor_data()
    return sensors[sensors["MACHINE_ID"].isin(in_service)].reset_index(drop=True)


def load_sensor_history(machine_id: str) -> pd.DataFrame:
    if config.SNOWFLAKE_MODE == "snowflake":
        from src.connection import get_session

        return get_session().sql(
            "SELECT * FROM RAW_SENSOR_DATA WHERE MACHINE_ID = ? ORDER BY TIME_CYCLE", params=[machine_id]
        ).to_pandas()
    sensors = load_raw_sensor_data()
    return sensors[sensors["MACHINE_ID"] == machine_id].sort_values("TIME_CYCLE")


def load_maintenance_history() -> pd.DataFrame:
    return _read("MAINTENANCE_HISTORY")


def load_operating_periods() -> pd.DataFrame:
    return _read("OPERATING_PERIODS")


def load_regime_sensor_stats() -> pd.DataFrame:
    return _read("REGIME_SENSOR_STATS")


def load_fleet_ground_truth() -> pd.DataFrame:
    """NASA's true RUL for the in-service fleet. Evaluation only — never an input
    to the live pipeline."""
    return _read("FLEET_GROUND_TRUTH")


def load_action_outcomes() -> pd.DataFrame:
    return _read("ACTION_OUTCOMES")


def append_action_outcome(row: dict) -> None:
    """Writes one new ACTION_OUTCOMES row (agentic action fired)."""
    if config.SNOWFLAKE_MODE == "snowflake":
        from src.connection import get_session

        session = get_session()
        session.create_dataframe([row]).write.mode("append").save_as_table("ACTION_OUTCOMES")
        return

    path = config.LOCAL_DATA_DIR / "action_outcomes.csv"
    df = _read_local("ACTION_OUTCOMES")
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)


def update_action_outcome_result(action_id: str, failure_occurred: bool, failure_date: str | None = None) -> None:
    """Closes the loop: records whether the predicted failure actually happened."""
    if config.SNOWFLAKE_MODE == "snowflake":
        from src.connection import get_session

        get_session().sql(
            "UPDATE ACTION_OUTCOMES SET FAILURE_OCCURRED_FLAG = ?, FAILURE_DATE = ? WHERE ACTION_ID = ?",
            params=[failure_occurred, failure_date, action_id],
        ).collect()
        return

    path = config.LOCAL_DATA_DIR / "action_outcomes.csv"
    df = _read_local("ACTION_OUTCOMES")
    mask = df["ACTION_ID"] == action_id
    df.loc[mask, "FAILURE_OCCURRED_FLAG"] = failure_occurred
    df.loc[mask, "FAILURE_DATE"] = failure_date
    df.to_csv(path, index=False)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
