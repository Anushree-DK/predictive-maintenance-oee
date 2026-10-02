"""Deploys the project to Snowflake (SNOWFLAKE_MODE=snowflake, credentials in .env).

    python scripts/deploy_snowflake.py tables     # create tables, load data/processed/*.csv
    python scripts/deploy_snowflake.py pipeline   # register procs, train in Snowflake, start scoring task
    python scripts/deploy_snowflake.py procedures # same, without retraining (code changes only)
    python scripts/deploy_snowflake.py knowledge  # NASA docs -> chunks -> Cortex Search service
    python scripts/deploy_snowflake.py agent      # semantic view, work orders, Cortex Agent
    python scripts/deploy_snowflake.py replay     # live-replay procedures (demo)
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


KNOWLEDGE_FILES = [config.RAW_CMAPSS_DIR / "Damage Propagation Modeling.pdf", config.RAW_CMAPSS_DIR / "readme.txt"]


def deploy_knowledge(session) -> None:
    session.sql(
        "CREATE STAGE IF NOT EXISTS DOCS_STAGE DIRECTORY = (ENABLE = TRUE) ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')"
    ).collect()
    for path in KNOWLEDGE_FILES:
        session.file.put(str(path), "@DOCS_STAGE", auto_compress=False, overwrite=True)
    session.sql("ALTER STAGE DOCS_STAGE REFRESH").collect()
    run_sql_file(session, config.ROOT_DIR / "sql" / "005_knowledge_base.sql")
    for row in session.sql("SELECT DOC_NAME, COUNT(*) AS N FROM DOC_CHUNKS GROUP BY 1").collect():
        print(f"{row['N']:>4} chunks  {row['DOC_NAME']}")


def deploy_agent(session) -> None:
    # WORK_ORDERS (created here) must exist before the semantic view that references it;
    # the agent resolves the view by name only when it runs.
    sensor_reference = pd.DataFrame(
        [(sensor, *text.split(" — ", 1)) for sensor, text in config.SENSOR_DESCRIPTIONS.items()],
        columns=["SENSOR", "SYMBOL", "DESCRIPTION"],
    )
    session.write_pandas(sensor_reference, "SENSOR_REFERENCE", auto_create_table=True, overwrite=True)
    run_sql_file(session, config.ROOT_DIR / "sql" / "006_maintenance_agent.sql")
    print("created SENSOR_REFERENCE, ENGINE_DEGRADATION_SIGNALS, WORK_ORDERS, DRAFT_WORK_ORDER, MAINTENANCE_AGENT")
    semantic_yaml = (config.ROOT_DIR / "sql" / "003_semantic_model.yaml").read_text(encoding="utf-8")
    db_schema = f"{config.SNOWFLAKE_CONNECTION_PARAMS['database']}.{config.SNOWFLAKE_CONNECTION_PARAMS['schema']}"
    print(session.sql("CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML(?, ?, TRUE)", params=[db_schema, semantic_yaml]).collect()[0][0])
    session.sql("DROP SEMANTIC VIEW IF EXISTS PM_SEMANTIC_VIEW").collect()
    print(session.sql("CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML(?, ?)", params=[db_schema, semantic_yaml]).collect()[0][0])


def deploy_pipeline(session, train: bool = True) -> None:
    from src.snowflake_pipeline import deploy

    deploy(session, train=train)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["tables", "pipeline", "procedures", "knowledge", "agent", "replay", "all"])
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
    if args.step in ("knowledge", "all"):
        deploy_knowledge(session)
    if args.step in ("agent", "all"):
        deploy_agent(session)
    if args.step in ("replay", "all"):
        run_sql_file(session, config.ROOT_DIR / "sql" / "007_live_replay.sql")
        print("created REPLAY_QUEUE and procedures REPLAY_PREPARE / REPLAY_STEP / REPLAY_RESTORE")


if __name__ == "__main__":
    main()
