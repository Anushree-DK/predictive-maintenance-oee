import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # inside Snowflake (stored procedures, registry models) there is no .env
    pass

ROOT_DIR = Path(__file__).parent
RAW_CMAPSS_DIR = ROOT_DIR / "data" / "raw" / "CMAPSS"
LOCAL_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
DEMO_DIR = ROOT_DIR / "data" / "demo"  # snapshot used when the full data isn't present (hosted demo)
if not LOCAL_DATA_DIR.exists() and DEMO_DIR.exists():
    LOCAL_DATA_DIR = MODELS_DIR = DEMO_DIR
REPORTS_DIR = ROOT_DIR / "reports"

# local = CSVs in data/processed, snowflake = the same tables in Snowflake
SNOWFLAKE_MODE = os.getenv("SNOWFLAKE_MODE", "local").lower()

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

# Sensor names from Table 2 of Saxena et al. 2008
SENSOR_DESCRIPTIONS = {
    "SENSOR_1": "T2 — total temperature at fan inlet",
    "SENSOR_2": "T24 — total temperature at LPC outlet",
    "SENSOR_3": "T30 — total temperature at HPC outlet",
    "SENSOR_4": "T50 — total temperature at LPT outlet",
    "SENSOR_5": "P2 — pressure at fan inlet",
    "SENSOR_6": "P15 — total pressure in bypass duct",
    "SENSOR_7": "P30 — total pressure at HPC outlet",
    "SENSOR_8": "Nf — physical fan speed",
    "SENSOR_9": "Nc — physical core speed",
    "SENSOR_10": "epr — engine pressure ratio (P50/P2)",
    "SENSOR_11": "Ps30 — static pressure at HPC outlet",
    "SENSOR_12": "phi — ratio of fuel flow to Ps30",
    "SENSOR_13": "NRf — corrected fan speed",
    "SENSOR_14": "NRc — corrected core speed",
    "SENSOR_15": "BPR — bypass ratio",
    "SENSOR_16": "farB — burner fuel-air ratio",
    "SENSOR_17": "htBleed — bleed enthalpy",
    "SENSOR_18": "Nf_dmd — demanded fan speed",
    "SENSOR_19": "PCNfR_dmd — demanded corrected fan speed",
    "SENSOR_20": "W31 — HPT coolant bleed",
    "SENSOR_21": "W32 — LPT coolant bleed",
}

# Non-constant sensors, used for the in-spec check and health index
INFORMATIVE_SENSORS = [f"SENSOR_{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]

# The four C-MAPSS subsets, treated as four fleets. Descriptions are from NASA's readme.
CMAPSS_FLEETS = {
    "FD001": {"OPERATING_CONDITIONS": "1 (sea level)", "FAULT_MODES": "HPC degradation"},
    "FD002": {"OPERATING_CONDITIONS": "6", "FAULT_MODES": "HPC degradation"},
    "FD003": {"OPERATING_CONDITIONS": "1 (sea level)", "FAULT_MODES": "HPC degradation, fan degradation"},
    "FD004": {"OPERATING_CONDITIONS": "6", "FAULT_MODES": "HPC degradation, fan degradation"},
}

# RUL label cap (piecewise-linear target)
RUL_CAP = 125
FAILURE_HORIZON_CYCLES = 30   # FAILURE_PROBABILITY = P(failure within this many cycles)
PREDICTION_INTERVAL = 0.90    # coverage target for the conformal RUL interval

# Risk thresholds on the RUL lower bound, in cycles
RISK_THRESHOLDS = {
    "CRITICAL": 15,
    "HIGH": 40,
    "MEDIUM": 90,
}  # anything above MEDIUM threshold is LOW

# OEE derivation from real sensor data (src/cmapss.py::operating_periods).
HEALTHY_BASELINE_CYCLES = 30  # each failed engine's first N cycles define "healthy"
IN_SPEC_Z = 3.0               # a cycle is in-spec if every informative sensor is within 3σ of healthy
PERIOD_CYCLES = 10            # cycles per OEE reporting period

# Cost assumptions ($/hour from the MaintainX 2024 report; repair hours are estimates)
COST_PER_DOWNTIME_HOUR_USD = 25_000
AVG_UNPLANNED_REPAIR_HOURS = 18  # emergency repair: parts sourcing, unplanned labor
AVG_PLANNED_REPAIR_HOURS = 4     # scheduled maintenance window, parts on hand
HOURS_PER_CYCLE = 2.0            # one C-MAPSS cycle is one flight; assumed average duration

# Maintenance scheduling (src/scheduler.py) — planning assumptions, adjustable in the dashboard.
SHOP_SLOT_CYCLES = 5          # one shop slot every 5 cycles
PLANNING_HORIZON_SLOTS = 6    # plan 6 slots (30 cycles) ahead
SHOP_CAPACITY_PER_SLOT = 10   # engines the shop can take per slot
