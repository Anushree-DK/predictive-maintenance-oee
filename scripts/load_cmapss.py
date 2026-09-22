"""Loads the real NASA C-MAPSS turbofan degradation dataset and writes it into the
same data/mock/*.csv files scripts/generate_mock_data.py produces, so nothing else
in the pipeline (feature engineering, training, dashboard) needs to change.

Download the dataset first (NASA's Prognostics Center of Excellence data repository,
"Turbofan Engine Degradation Simulation Data Set"), unzip it, and point --raw-dir at
the folder containing train_FD00X.txt / test_FD00X.txt / RUL_FD00X.txt.

Usage:
    python scripts/load_cmapss.py --raw-dir data/raw/CMAPSS --dataset FD001
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from scripts.generate_mock_data import (
    generate_machine_data,
    generate_maintenance_history,
    generate_production_data,
)

COLUMNS = ["UNIT_ID", "TIME_CYCLE", "OP_SETTING_1", "OP_SETTING_2", "OP_SETTING_3"] + [
    f"SENSOR_{i}" for i in range(1, 22)
]


def _read_cmapss_txt(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the C-MAPSS dataset and unzip it so this file exists, "
            f"then pass --raw-dir pointing at that folder."
        )
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLUMNS)]
    df.columns = COLUMNS
    return df


RUL_CAP = 125  # standard piecewise-linear RUL label used in C-MAPSS literature
# (see Saxena & Goebel, "Damage Propagation Modeling..."): early in a unit's life,
# sensors look the same as a healthy unit, so an uncapped linear RUL target teaches
# the model to guess wildly high for anything that still looks healthy. Capping the
# label says "we only claim to estimate RUL once degradation is actually visible."


def build_training_set(raw_dir: Path, dataset: str) -> pd.DataFrame:
    """Full run-to-failure trajectories -> labeled (features, RUL) training rows."""
    train = _read_cmapss_txt(raw_dir / f"train_{dataset}.txt")
    train["UNIT_ID"] = train["UNIT_ID"].apply(lambda u: f"{dataset}-TRAIN-{int(u):03d}")

    life = train.groupby("UNIT_ID")["TIME_CYCLE"].transform("max")
    train["LIFE"] = life
    train["RUL"] = (life - train["TIME_CYCLE"]).clip(upper=RUL_CAP)
    return train


def build_fleet_snapshot(raw_dir: Path, dataset: str, max_machines: int | None):
    """Partial (not-yet-failed) test trajectories -> today's fleet state, i.e. what
    RAW_SENSOR_DATA would hold in production. RUL_FD00X.txt has the true RUL for each
    test unit at its cutoff — useful for checking the trained model, not for training it.
    """
    test = _read_cmapss_txt(raw_dir / f"test_{dataset}.txt")
    true_rul = pd.read_csv(raw_dir / f"RUL_{dataset}.txt", header=None, names=["TRUE_RUL"])

    unit_numbers = sorted(test["UNIT_ID"].unique())
    if max_machines is not None:
        unit_numbers = unit_numbers[:max_machines]
    true_rul = true_rul.iloc[: len(unit_numbers)]

    unit_to_machine = {u: f"M-{i + 1:03d}" for i, u in enumerate(unit_numbers)}
    test = test[test["UNIT_ID"].isin(unit_numbers)].copy()
    test["MACHINE_ID"] = test["UNIT_ID"].map(unit_to_machine)
    test = test.drop(columns=["UNIT_ID"])
    test["RECORDED_AT"] = test["TIME_CYCLE"].apply(lambda c: datetime(2026, 1, 1) + timedelta(hours=int(c)))

    machine_ids = [unit_to_machine[u] for u in unit_numbers]
    true_rul_df = pd.DataFrame({"MACHINE_ID": machine_ids, "TRUE_RUL": true_rul["TRUE_RUL"].values})

    id_cols = ["MACHINE_ID", "TIME_CYCLE"]
    test = test[id_cols + [c for c in test.columns if c not in id_cols]]
    return test, machine_ids, true_rul_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/CMAPSS", help="Folder with train/test/RUL .txt files")
    parser.add_argument("--dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--max-machines", type=int, default=20, help="Cap fleet size for the dashboard demo")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    config.MOCK_DATA_DIR.mkdir(parents=True, exist_ok=True)

    training_set = build_training_set(raw_dir, args.dataset)
    fleet_snapshot, machine_ids, true_rul_df = build_fleet_snapshot(raw_dir, args.dataset, args.max_machines)

    datasets = {
        "training_run_to_failure.csv": training_set,
        "raw_sensor_data.csv": fleet_snapshot,
        "machine_data.csv": generate_machine_data(machine_ids),
        "maintenance_history.csv": generate_maintenance_history(machine_ids),
        "production_data.csv": generate_production_data(machine_ids),
        "test_true_rul.csv": true_rul_df,
    }
    for filename, df in datasets.items():
        path = config.MOCK_DATA_DIR / filename
        df.to_csv(path, index=False)
        print(f"wrote {len(df):>6} rows -> {path}")

    outcomes_path = config.MOCK_DATA_DIR / "action_outcomes.csv"
    if not outcomes_path.exists():
        pd.DataFrame(
            columns=[
                "ACTION_ID", "MACHINE_ID", "ACTION_TYPE", "RECOMMENDED_AT",
                "RUL_PREDICTION_AT_ACTION", "RISK_CLASS_AT_ACTION", "CONFIDENCE_AT_ACTION",
                "TAKEN_FLAG", "TAKEN_AT", "FAILURE_OCCURRED_FLAG", "FAILURE_DATE", "NOTES",
            ]
        ).to_csv(outcomes_path, index=False)
        print(f"wrote      0 rows -> {outcomes_path}")

    print(
        f"\n{args.dataset}: {training_set['UNIT_ID'].nunique()} training units, "
        f"{len(machine_ids)} fleet machines. test_true_rul.csv holds ground truth for "
        f"validating predictions (not used by the live pipeline)."
    )


if __name__ == "__main__":
    main()
