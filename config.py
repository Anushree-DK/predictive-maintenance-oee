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
REPORTS_DIR = ROOT_DIR / "reports"

# "local" reads the processed C-MAPSS CSVs in data/processed/; "snowflake" reads the
# same tables from Snowflake. Both hold the same real data — only the backend differs.
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

# What each sensor column measures: the 21 sensor columns follow the order of the
# participant parameters in Table 2 of Saxena et al. 2008 (the paper shipped with
# C-MAPSS, indexed in the agent's knowledge base). Units: °R, psia, rpm.
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

# The 14 sensors that are not constant under a fixed operating condition — the set
# used throughout the C-MAPSS literature for health assessment. Used for the
# in-spec (quality) check and health index; the RUL model itself uses all 21,
# because under the six-condition fleets the others still carry signal.
INFORMATIVE_SENSORS = [f"SENSOR_{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]

# The four C-MAPSS subsets, treated as four fleets. Descriptions are from NASA's readme.
CMAPSS_FLEETS = {
    "FD001": {"OPERATING_CONDITIONS": "1 (sea level)", "FAULT_MODES": "HPC degradation"},
    "FD002": {"OPERATING_CONDITIONS": "6", "FAULT_MODES": "HPC degradation"},
    "FD003": {"OPERATING_CONDITIONS": "1 (sea level)", "FAULT_MODES": "HPC degradation, fan degradation"},
    "FD004": {"OPERATING_CONDITIONS": "6", "FAULT_MODES": "HPC degradation, fan degradation"},
}

# Standard piecewise-linear RUL label from the C-MAPSS literature: early in life the
# sensors look like a healthy engine's, so we only claim to estimate RUL once
# degradation is visible. Predictions at the cap read as "125+ cycles".
RUL_CAP = 125
FAILURE_HORIZON_CYCLES = 30   # FAILURE_PROBABILITY = P(failure within this many cycles)
PREDICTION_INTERVAL = 0.90    # coverage target for the conformal RUL interval

# Risk class, on the lower bound of the RUL interval (in cycles) — i.e. how soon the
# engine could plausibly fail, not just the point estimate.
RISK_THRESHOLDS = {
    "CRITICAL": 15,
    "HIGH": 40,
    "MEDIUM": 90,
}  # anything above MEDIUM threshold is LOW

# OEE derivation from real sensor data (src/cmapss.py::operating_periods).
HEALTHY_BASELINE_CYCLES = 30  # each failed engine's first N cycles define "healthy"
IN_SPEC_Z = 3.0               # a cycle is in-spec if every informative sensor is within 3σ of healthy
PERIOD_CYCLES = 10            # cycles per OEE reporting period

# Business-impact assumptions (src/business_impact.py, src/oee.py). C-MAPSS has no
# cost or duration data, so these are the only non-measured numbers in the project.
# COST_PER_DOWNTIME_HOUR_USD is a cited industry benchmark (MaintainX 2024 State of
# Industrial Maintenance report, average across surveyed manufacturers); the duration
# figures are explicit, editable assumptions, not measured for any real operator.
COST_PER_DOWNTIME_HOUR_USD = 25_000
AVG_UNPLANNED_REPAIR_HOURS = 18  # emergency repair: parts sourcing, unplanned labor
AVG_PLANNED_REPAIR_HOURS = 4     # scheduled maintenance window, parts on hand
HOURS_PER_CYCLE = 2.0            # one C-MAPSS cycle is one flight; assumed average duration

# Maintenance scheduling (src/scheduler.py) — planning assumptions, adjustable in the dashboard.
SHOP_SLOT_CYCLES = 5          # one shop slot every 5 cycles
PLANNING_HORIZON_SLOTS = 6    # plan 6 slots (30 cycles) ahead
SHOP_CAPACITY_PER_SLOT = 10   # engines the shop can take per slot
