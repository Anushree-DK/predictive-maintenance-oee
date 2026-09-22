import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent
MOCK_DATA_DIR = ROOT_DIR / "data" / "mock"
MODELS_DIR = ROOT_DIR / "models"

SNOWFLAKE_MODE = os.getenv("SNOWFLAKE_MODE", "mock").lower()

SNOWFLAKE_CONNECTION_PARAMS = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "role": os.getenv("SNOWFLAKE_ROLE"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE", "PDM"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
}

SENSOR_COLUMNS = [f"SENSOR_{i}" for i in range(1, 22)]
OP_SETTING_COLUMNS = ["OP_SETTING_1", "OP_SETTING_2", "OP_SETTING_3"]

# Risk thresholds on predicted remaining-useful-life (in cycles).
RISK_THRESHOLDS = {
    "CRITICAL": 15,
    "HIGH": 40,
    "MEDIUM": 90,
}  # anything above MEDIUM threshold is LOW

# Business-impact assumptions (src/business_impact.py). COST_PER_DOWNTIME_HOUR_USD
# is a cited industry benchmark (MaintainX 2024 State of Industrial Maintenance
# report, average across surveyed manufacturers); the two duration figures are
# explicit, editable assumptions — not measured for any real site — used only to
# illustrate the planned-vs-unplanned-repair cost gap this system is meant to close.
COST_PER_DOWNTIME_HOUR_USD = 25_000
AVG_UNPLANNED_REPAIR_HOURS = 18  # emergency repair: parts sourcing, unplanned labor
AVG_PLANNED_REPAIR_HOURS = 4     # scheduled maintenance window, parts on hand
