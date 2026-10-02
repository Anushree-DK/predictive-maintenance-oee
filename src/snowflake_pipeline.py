"""The in-Snowflake ML pipeline: training, model registry, and event-driven scoring.

- TRAIN_RUL_MODEL()  Python stored procedure: trains on the failed engines, evaluates
                     on NASA's test set, logs the model to the Snowflake Model Registry
                     as RUL_MODEL (new version, made default) with its metrics, and
                     records them in MODEL_METRICS.
- SCORE_FLEET()      Python stored procedure: computes current features for the
                     in-service fleet in SQL, scores them with the registry's default
                     RUL_MODEL version, and writes ENGINE_FEATURES / FLEET_PREDICTIONS /
                     ENGINE_RUL_DRIVERS (per-sensor SHAP contributions).
- SCORE_FLEET_TASK   Runs SCORE_FLEET() whenever RAW_SENSOR_STREAM (a stream on
                     RAW_SENSOR_DATA) has new readings; idle otherwise.
- RETRAIN_TASK       Weekly TRAIN_RUL_MODEL(); created suspended.

Deployed by: python scripts/deploy_snowflake.py pipeline
"""

import json
import os
import zipfile

from snowflake.snowpark import Session

import config

MODEL_NAME = "RUL_MODEL"
CODE_STAGE = "PIPELINE_CODE"


def _bind(session) -> None:
    """Point the project's data layer at the stored procedure's own session."""
    from src.connection import set_session

    config.SNOWFLAKE_MODE = "snowflake"
    set_session(session)


def _registry(session):
    from snowflake.ml.registry import Registry

    db = config.SNOWFLAKE_CONNECTION_PARAMS["database"] or "PDM"
    schema = config.SNOWFLAKE_CONNECTION_PARAMS["schema"] or "PUBLIC"
    return Registry(session=session, database_name=db, schema_name=schema)


def _code_paths(target_dir: str = "/tmp/pdm_code") -> list[str]:
    """Filesystem paths of the src package and config.py, for the registry to bundle
    with the model. Inside a stored procedure they may be imported from a zip."""
    import src

    paths = []
    for path in (src.__path__[0], config.__file__):
        if os.path.exists(path):
            paths.append(path)
            continue
        zip_path = path[: path.index(".zip") + len(".zip")]
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(target_dir)
        paths.append(os.path.join(target_dir, os.path.basename(path)))
    return paths


def train_procedure(session: Session) -> str:
    import tempfile
    from datetime import datetime, timezone

    import joblib
    import shap
    import sklearn
    from snowflake.ml.model import custom_model

    from src.ml.registry_model import RulModel
    from src.ml.train import build_evaluation_set, build_training_set, evaluate, fit_models

    _bind(session)
    train, test = build_training_set(), build_evaluation_set()
    bundle = fit_models(train)
    metrics = evaluate(bundle, train, test)

    bundle_path = os.path.join(tempfile.mkdtemp(), "rul_bundle.joblib")
    joblib.dump(bundle, bundle_path)

    version = "V" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    registry = _registry(session)
    registry.log_model(
        RulModel(custom_model.ModelContext(artifacts={"bundle": bundle_path})),
        model_name=MODEL_NAME,
        version_name=version,
        sample_input_data=test[bundle["feature_columns"]].head(50),
        conda_dependencies=[
            f"scikit-learn=={sklearn.__version__}", "pandas", "numpy", f"joblib=={joblib.__version__}", f"shap=={shap.__version__}",
        ],
        code_paths=_code_paths(),
        metrics=metrics,
        comment=(
            f"RUL RMSE {metrics['rul']['overall']['rmse']} cycles on NASA C-MAPSS test set; "
            f"{config.PREDICTION_INTERVAL:.0%} interval coverage {metrics['rul_interval']['empirical_coverage']}"
        ),
        target_platforms=["WAREHOUSE"],
    )
    registry.get_model(MODEL_NAME).default = version

    session.sql(
        "INSERT INTO MODEL_METRICS (MODEL_VERSION, TRAINED_AT, TRAINING_ENGINES, METRICS) "
        "SELECT ?, SYSDATE(), ?, PARSE_JSON(?)",
        params=[version, int(train["MACHINE_ID"].nunique()), json.dumps(metrics)],
    ).collect()
    return json.dumps({"version": version, "rmse": metrics["rul"]["overall"]["rmse"]})


def score_procedure(session: Session) -> str:
    from snowflake.snowpark import functions as F

    from src.feature_engineering import build_feature_sql

    _bind(session)
    model = _registry(session).get_model(MODEL_NAME)
    version = model.default
    in_service = "r.MACHINE_ID IN (SELECT MACHINE_ID FROM MACHINE_DATA WHERE STATUS = 'IN_SERVICE')"

    features = session.sql(build_feature_sql(where=in_service))
    features.write.save_as_table("ENGINE_FEATURES", mode="truncate")
    features = session.table("ENGINE_FEATURES")

    scored = version.run(features, function_name="predict")
    predictions = scored.select(
        "MACHINE_ID",
        F.col("TIME_CYCLE").alias("LAST_CYCLE"),
        "RUL_PREDICTION", "RUL_LOWER", "RUL_UPPER", "FAILURE_PROBABILITY", "RISK_CLASS",
        F.sysdate().alias("SCORED_AT"),  # UTC, unlike CURRENT_TIMESTAMP()
        F.lit(version.version_name).alias("MODEL_VERSION"),
    )
    # "truncate" keeps the table object, which the semantic view and agent reference.
    predictions.write.save_as_table("FLEET_PREDICTIONS", mode="truncate")
    engines = session.table("FLEET_PREDICTIONS").count()

    if "EXPLAIN" in {f["name"].upper() for f in version.show_functions()}:
        from src.ml.explain import to_long

        explained = version.run(features, function_name="explain").to_pandas()
        shap_cols = [c for c in explained.columns if c.startswith("SHAP_") or c == "BASE_RUL"]
        drivers = to_long(explained["MACHINE_ID"], explained[shap_cols])
        session.write_pandas(drivers, "ENGINE_RUL_DRIVERS", auto_create_table=True, overwrite=True)

    # Reading the stream inside DML advances its offset, so the task goes idle again.
    session.sql(
        "INSERT INTO SCORING_RUNS (RUN_AT, NEW_SENSOR_ROWS, ENGINES_SCORED, MODEL_VERSION) "
        "SELECT SYSDATE(), (SELECT COUNT(*) FROM RAW_SENSOR_STREAM), ?, ?",
        params=[engines, version.version_name],
    ).collect()
    return json.dumps({"engines_scored": engines, "model_version": version.version_name})


SETUP_SQL = [
    f"CREATE STAGE IF NOT EXISTS {CODE_STAGE}",
    """CREATE TABLE IF NOT EXISTS MODEL_METRICS (
        MODEL_VERSION STRING, TRAINED_AT TIMESTAMP_NTZ, TRAINING_ENGINES NUMBER, METRICS VARIANT)""",
    """CREATE TABLE IF NOT EXISTS SCORING_RUNS (
        RUN_AT TIMESTAMP_NTZ, NEW_SENSOR_ROWS NUMBER, ENGINES_SCORED NUMBER, MODEL_VERSION STRING)""",
    "CREATE STREAM IF NOT EXISTS RAW_SENSOR_STREAM ON TABLE RAW_SENSOR_DATA APPEND_ONLY = TRUE",
]

TASK_SQL = [
    """CREATE OR REPLACE TASK SCORE_FLEET_TASK
        WAREHOUSE = {warehouse}
        SCHEDULE = '1 MINUTE'
        WHEN SYSTEM$STREAM_HAS_DATA('RAW_SENSOR_STREAM')
        AS CALL SCORE_FLEET()""",
    "ALTER TASK SCORE_FLEET_TASK RESUME",
    """CREATE OR REPLACE TASK RETRAIN_TASK
        WAREHOUSE = {warehouse}
        SCHEDULE = 'USING CRON 0 2 * * 0 UTC'
        AS CALL TRAIN_RUL_MODEL()""",
]

# Pinned to what Snowflake's package channel serves: the registry derives the model's
# serving dependencies from the training environment, so an unpinned (newer) build
# here produces a model whose dependencies the warehouse can't install.
PROCEDURE_PACKAGES = [
    "snowflake-snowpark-python", "snowflake-ml-python==2.2.0", "scikit-learn==1.9.1", "pandas", "numpy", "joblib==1.5.3",
    "shap==0.51.0",
]


def register_procedures(session) -> None:
    from snowflake.snowpark.types import StringType

    root = config.ROOT_DIR
    # Snowpark reuses an already-staged import zip rather than re-uploading changed
    # code, so clear the stage to make the procedures run the current source.
    session.sql(f"REMOVE @{CODE_STAGE}").collect()
    for func, name in ((train_procedure, "TRAIN_RUL_MODEL"), (score_procedure, "SCORE_FLEET")):
        session.sproc.register(
            func,
            name=name,
            return_type=StringType(),
            input_types=[],
            packages=PROCEDURE_PACKAGES,
            imports=[(str(root / "src"), "src"), (str(root / "config.py"), "config")],
            is_permanent=True,
            stage_location=f"@{CODE_STAGE}",
            replace=True,
        )
        print(f"registered procedure {name}")


def deploy(session, train: bool = True) -> None:
    for statement in SETUP_SQL:
        session.sql(statement).collect()
    register_procedures(session)
    if train:
        print("training in Snowflake:", session.sql("CALL TRAIN_RUL_MODEL()").collect()[0][0])
    print("scoring in Snowflake:", session.sql("CALL SCORE_FLEET()").collect()[0][0])
    warehouse = config.SNOWFLAKE_CONNECTION_PARAMS["warehouse"]
    for statement in TASK_SQL:
        session.sql(statement.format(warehouse=warehouse)).collect()
    print("SCORE_FLEET_TASK resumed (runs when RAW_SENSOR_STREAM has data); RETRAIN_TASK created suspended")
