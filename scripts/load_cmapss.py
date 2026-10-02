"""Builds every project table from the real NASA C-MAPSS dataset and writes them to
data/processed/*.csv (the "local" backend; the same files are what gets loaded
into Snowflake). See src/cmapss.py for how each table is derived.

Download NASA's "Turbofan Engine Degradation Simulation Data Set" and unzip it so
data/raw/CMAPSS/ holds train_FD00X.txt / test_FD00X.txt / RUL_FD00X.txt, then:

    python scripts/load_cmapss.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.cmapss import build_all_tables

ACTION_OUTCOME_COLUMNS = [
    "ACTION_ID", "MACHINE_ID", "ACTION_TYPE", "RECOMMENDED_AT",
    "RUL_PREDICTION_AT_ACTION", "RUL_LOWER_AT_ACTION", "FAILURE_PROBABILITY_AT_ACTION",
    "RISK_CLASS_AT_ACTION", "TAKEN_FLAG", "TAKEN_AT", "FAILURE_OCCURRED_FLAG", "FAILURE_DATE", "NOTES",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=str(config.RAW_CMAPSS_DIR), help="Folder with the C-MAPSS .txt files")
    parser.add_argument("--fleets", nargs="+", default=list(config.CMAPSS_FLEETS), choices=list(config.CMAPSS_FLEETS))
    args = parser.parse_args()

    config.LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in build_all_tables(Path(args.raw_dir), fleets=tuple(args.fleets)).items():
        path = config.LOCAL_DATA_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {len(df):>7} rows -> {path}")

    # Starts empty: rows come only from real interaction with the dashboard.
    outcomes_path = config.LOCAL_DATA_DIR / "action_outcomes.csv"
    if not outcomes_path.exists():
        pd.DataFrame(columns=ACTION_OUTCOME_COLUMNS).to_csv(outcomes_path, index=False)
        print(f"wrote       0 rows -> {outcomes_path}")


if __name__ == "__main__":
    main()
