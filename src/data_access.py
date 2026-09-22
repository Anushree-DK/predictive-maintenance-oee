"""Single read/write interface over either local mock CSVs or real Snowflake tables,
selected by SNOWFLAKE_MODE. Every other module in src/ goes through this file instead
of touching files or Snowpark directly, so swapping modes doesn't change pipeline code.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

import config


def _read_mock(filename: str) -> pd.DataFrame:
    path = config.MOCK_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run: python scripts/generate_mock_data.py")
    return pd.read_csv(path)


def _read_snowflake(table_name: str) -> pd.DataFrame:
    from src.connection import get_session

    return get_session().table(table_name).to_pandas()


def load_machine_data() -> pd.DataFrame:
    if config.SNOWFLAKE_MODE == "snowflake":
        return _read_snowflake("MACHINE_DATA")
    return _read_mock("machine_data.csv")


def load_raw_sensor_data() -> pd.DataFrame:
    if config.SNOWFLAKE_MODE == "snowflake":
        return _read_snowflake("RAW_SENSOR_DATA")
    return _read_mock("raw_sensor_data.csv")


def load_maintenance_history() -> pd.DataFrame:
    if config.SNOWFLAKE_MODE == "snowflake":
        return _read_snowflake("MAINTENANCE_HISTORY")
    return _read_mock("maintenance_history.csv")


def load_production_data() -> pd.DataFrame:
    if config.SNOWFLAKE_MODE == "snowflake":
        return _read_snowflake("PRODUCTION_DATA")
    return _read_mock("production_data.csv")


def load_training_run_to_failure() -> pd.DataFrame:
    # Only ever needed locally to bootstrap the model; a real deployment would
    # instead train against historical RAW_SENSOR_DATA joined to failure events.
    return _read_mock("training_run_to_failure.csv")


def load_action_outcomes() -> pd.DataFrame:
    if config.SNOWFLAKE_MODE == "snowflake":
        return _read_snowflake("ACTION_OUTCOMES")
    return _read_mock("action_outcomes.csv")


def append_action_outcome(row: dict) -> None:
    """Writes one new ACTION_OUTCOMES row (agentic action fired)."""
    if config.SNOWFLAKE_MODE == "snowflake":
        from src.connection import get_session

        session = get_session()
        session.create_dataframe([row]).write.mode("append").save_as_table("ACTION_OUTCOMES")
        return

    path = config.MOCK_DATA_DIR / "action_outcomes.csv"
    df = _read_mock("action_outcomes.csv")
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)


def update_action_outcome_result(action_id: str, failure_occurred: bool, failure_date: str | None = None) -> None:
    """Closes the loop: records whether the predicted failure actually happened."""
    if config.SNOWFLAKE_MODE == "snowflake":
        from src.connection import get_session

        session = get_session()
        session.sql(
            f"""
            UPDATE ACTION_OUTCOMES
            SET FAILURE_OCCURRED_FLAG = {failure_occurred},
                FAILURE_DATE = {"'" + failure_date + "'" if failure_date else "NULL"}
            WHERE ACTION_ID = '{action_id}'
            """
        ).collect()
        return

    path = config.MOCK_DATA_DIR / "action_outcomes.csv"
    df = _read_mock("action_outcomes.csv")
    mask = df["ACTION_ID"] == action_id
    df.loc[mask, "FAILURE_OCCURRED_FLAG"] = failure_occurred
    df.loc[mask, "FAILURE_DATE"] = failure_date
    df.to_csv(path, index=False)


def now_iso() -> str:
    return datetime.utcnow().isoformat()
