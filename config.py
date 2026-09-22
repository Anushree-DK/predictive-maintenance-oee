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
