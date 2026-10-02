"""Deploys the project to Snowflake (SNOWFLAKE_MODE=snowflake, credentials in .env).

    python scripts/deploy_snowflake.py tables     # create tables, load data/processed/*.csv
    python scripts/deploy_snowflake.py pipeline   # register procs, train in Snowflake, start scoring task
    python scripts/deploy_snowflake.py procedures # same, without retraining (code changes only)
    python scripts/deploy_snowflake.py all

Run scripts/load_cmapss.py first so data/processed/ exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.connection import get_session

# ACTION_OUTCOMES is deliberately absent: it holds real user actions and is never reloaded.
DATA_TABLES = [
    "MACHINE_DATA", "RAW_SENSOR_DATA", "MAINTENANCE_HISTORY",
    "OPERATING_PERIODS", "REGIME_SENSOR_STATS", "FLEET_GROUND_TRUTH",
]


def run_sql_file(session, path: Path) -> None:
    for cursor in session.connection.execute_string(path.read_text(encoding="utf-8")):
        cursor.close()


def deploy_tables(session) -> None:
    run_sql_file(session, config.ROOT_DIR / "sql" / "001_create_tables.sql")
    for table in DATA_TABLES:
        df = pd.read_csv(config.LOCAL_DATA_DIR / f"{table.lower()}.csv")
        session.write_pandas(df, table, auto_create_table=False, overwrite=False)
        count = session.table(table).count()
        if count != len(df):
            raise RuntimeError(f"{table}: loaded {count} rows, expected {len(df)}")
        print(f"{table:<22} {count:>8,} rows")


def deploy_pipeline(session, train: bool = True) -> None:
    from src.snowflake_pipeline import deploy

    deploy(session, train=train)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["tables", "pipeline", "procedures", "all"])
    args = parser.parse_args()

    if config.SNOWFLAKE_MODE != "snowflake":
        sys.exit("Set SNOWFLAKE_MODE=snowflake in .env first.")
    session = get_session()
    if args.step in ("tables", "all"):
        deploy_tables(session)
    if args.step in ("pipeline", "all"):
        deploy_pipeline(session)
    if args.step == "procedures":
        deploy_pipeline(session, train=False)


if __name__ == "__main__":
    main()
