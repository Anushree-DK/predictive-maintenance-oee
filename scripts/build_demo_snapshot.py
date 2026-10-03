"""Copies what the dashboard needs into data/demo/ so it can run as a hosted demo."""

import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

SOURCE = config.ROOT_DIR / "data" / "processed"
DEMO = config.ROOT_DIR / "data" / "demo"


def main():
    DEMO.mkdir(parents=True, exist_ok=True)
    machines = pd.read_csv(SOURCE / "machine_data.csv")
    in_service = machines.loc[machines["STATUS"] == "IN_SERVICE", "MACHINE_ID"]

    sensors = pd.read_csv(SOURCE / "raw_sensor_data.csv")
    sensors[sensors["MACHINE_ID"].isin(in_service)].to_csv(DEMO / "raw_sensor_data.csv.gz", index=False)
    machines.to_csv(DEMO / "machine_data.csv", index=False)
    for name in ("operating_periods", "regime_sensor_stats"):
        shutil.copy(SOURCE / f"{name}.csv", DEMO / f"{name}.csv")
    pd.read_csv(SOURCE / "action_outcomes.csv").head(0).to_csv(DEMO / "action_outcomes.csv", index=False)
    shutil.copy(config.ROOT_DIR / "models" / "rul_model.joblib", DEMO / "rul_model.joblib")

    for path in sorted(DEMO.iterdir()):
        print(f"{path.stat().st_size / 1e6:6.1f} MB  {path.name}")


if __name__ == "__main__":
    main()
