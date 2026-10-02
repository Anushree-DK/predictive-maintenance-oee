"""Turns the raw NASA C-MAPSS files into the project's tables.

Nothing here is randomly generated. Every row is either a C-MAPSS reading or a
deterministic derivation of one; DATA_SOURCES.md lists which is which.

The four subsets (FD001-FD004) are four fleets. In each one:
- the *train* trajectories are engines that ran until they failed, which become
  STATUS = FAILED machines with a real failure event in MAINTENANCE_HISTORY;
- the *test* trajectories stop before failure, which become the IN_SERVICE fleet
  the dashboard monitors. NASA's RUL_FD00X.txt holds their true remaining life,
  kept apart in FLEET_GROUND_TRUTH and used only to evaluate the model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import config

RAW_COLUMNS = ["UNIT_NUMBER", "TIME_CYCLE"] + config.OP_SETTING_COLUMNS + config.SENSOR_COLUMNS


def machine_id(fleet: str, unit_number: int, status: str) -> str:
    return f"{fleet}-{'ENG' if status == 'IN_SERVICE' else 'HIST'}-{unit_number:03d}"


def read_raw_file(raw_dir: Path, kind: str, fleet: str) -> pd.DataFrame:
    path = Path(raw_dir) / f"{kind}_{fleet}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download NASA's 'Turbofan Engine Degradation Simulation "
            f"Data Set' (C-MAPSS) and unzip it into {raw_dir} — see README."
        )
    df = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, : len(RAW_COLUMNS)]
    df.columns = RAW_COLUMNS
    return df


def load_fleet(raw_dir: Path, fleet: str) -> dict[str, pd.DataFrame]:
    """One subset -> sensor rows, machine rows, failure events, and ground truth."""
    sensor_frames, machine_rows = [], []

    for kind, status in (("train", "FAILED"), ("test", "IN_SERVICE")):
        raw = read_raw_file(raw_dir, kind, fleet)
        raw["MACHINE_ID"] = [machine_id(fleet, u, status) for u in raw["UNIT_NUMBER"]]
        sensor_frames.append(raw.drop(columns=["UNIT_NUMBER"]))

        cycles = raw.groupby(["MACHINE_ID", "UNIT_NUMBER"])["TIME_CYCLE"].max().reset_index()
        for _, row in cycles.iterrows():
            machine_rows.append({
                "MACHINE_ID": row["MACHINE_ID"],
                "FLEET": fleet,
                "UNIT_NUMBER": int(row["UNIT_NUMBER"]),
                "SOURCE_FILE": f"{kind}_{fleet}.txt",
                **config.CMAPSS_FLEETS[fleet],
                "STATUS": status,
                "CYCLES_OBSERVED": int(row["TIME_CYCLE"]),
            })

    machines = pd.DataFrame(machine_rows)
    sensors = pd.concat(sensor_frames, ignore_index=True)
    sensors = sensors[["MACHINE_ID", "TIME_CYCLE"] + config.OP_SETTING_COLUMNS + config.SENSOR_COLUMNS]

    # A train trajectory's last cycle is the cycle the engine failed on.
    failed = machines[machines["STATUS"] == "FAILED"]
    failures = pd.DataFrame({
        "EVENT_ID": "FAIL-" + failed["MACHINE_ID"],
        "MACHINE_ID": failed["MACHINE_ID"],
        "EVENT_CYCLE": failed["CYCLES_OBSERVED"],
        "EVENT_TYPE": "UNPLANNED_FAILURE",
        "DESCRIPTION": "Run to failure; fleet fault modes: " + failed["FAULT_MODES"],
    })

    in_service = machines[machines["STATUS"] == "IN_SERVICE"].sort_values("UNIT_NUMBER")
    true_rul = pd.read_csv(Path(raw_dir) / f"RUL_{fleet}.txt", header=None, names=["TRUE_RUL"])
    if len(true_rul) != len(in_service):
        raise ValueError(f"RUL_{fleet}.txt has {len(true_rul)} rows for {len(in_service)} test engines")
    ground_truth = pd.DataFrame({
        "MACHINE_ID": in_service["MACHINE_ID"].values,
        "LAST_OBSERVED_CYCLE": in_service["CYCLES_OBSERVED"].values,
        "TRUE_RUL": true_rul["TRUE_RUL"].values,
    })

    return {"machines": machines, "sensors": sensors, "failures": failures, "ground_truth": ground_truth}


def add_regime(sensors: pd.DataFrame) -> pd.DataFrame:
    """C-MAPSS's six operating conditions are separable on OP_SETTING_1 alone
    (altitude, in kft: 0/10/20/25/35/42). The single-condition fleets are all 0."""
    return sensors.assign(REGIME=sensors["OP_SETTING_1"].round(0).astype(int))


def regime_sensor_stats(sensors: pd.DataFrame, rows: pd.Series | None = None) -> pd.DataFrame:
    """Per-regime mean/std of every sensor, one row per REGIME (wide, so it joins
    cleanly in SQL). `rows` optionally restricts which readings define the stats."""
    df = add_regime(sensors if rows is None else sensors[rows])
    grouped = df.groupby("REGIME")[config.SENSOR_COLUMNS]
    means = grouped.mean().add_suffix("_MEAN")
    # A sensor that is constant within a regime gets std 1, so its z-score is just
    # its (zero) deviation rather than a division by zero.
    stds = grouped.std(ddof=0).where(lambda s: s > 1e-8, 1.0).add_suffix("_STD")
    return pd.concat([means, stds], axis=1).reset_index()


def zscore_by_regime(sensors: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    df = add_regime(sensors).merge(stats, on="REGIME", how="left")
    if df[f"{config.SENSOR_COLUMNS[0]}_MEAN"].isna().any():
        raise ValueError("sensor readings in an operating regime the stats table doesn't cover")
    for col in config.SENSOR_COLUMNS:
        df[col] = (df[col] - df[f"{col}_MEAN"]) / df[f"{col}_STD"]
    return df[sensors.columns.tolist() + ["REGIME"]]


def failure_cycles(failures: pd.DataFrame) -> pd.Series:
    return failures.set_index("MACHINE_ID")["EVENT_CYCLE"]


def fit_health_index(z_sensors: pd.DataFrame, failures: pd.DataFrame) -> LinearRegression:
    """Linear health index over the informative sensors, fitted on failed engines to
    the same piecewise-linear shape as the RUL label: 1 while healthy, falling to 0
    over the last RUL_CAP cycles before failure — the standard data-driven health
    indicator from the PHM literature. Learned from the real trajectories, not tuned."""
    life = failure_cycles(failures)
    hist = z_sensors[z_sensors["MACHINE_ID"].isin(life.index)]
    rul = hist["MACHINE_ID"].map(life) - hist["TIME_CYCLE"]
    target = rul.clip(upper=config.RUL_CAP) / config.RUL_CAP
    return LinearRegression().fit(hist[config.INFORMATIVE_SENSORS], target)


def operating_periods(
    sensors: pd.DataFrame, machines: pd.DataFrame, failures: pd.DataFrame
) -> pd.DataFrame:
    """Per machine, per PERIOD_CYCLES-cycle period: the inputs src/oee.py needs.

    - CYCLES_FLOWN: cycles actually recorded in the period.
    - IN_SPEC_CYCLES: cycles where every informative sensor is within IN_SPEC_Z
      standard deviations of the healthy baseline (failed engines' first
      HEALTHY_BASELINE_CYCLES cycles, per operating regime).
    - AVG_HEALTH_INDEX: mean of the fitted health index, clipped to [0, 1].
    - FAILURE_FLAG: the engine failed during this period.
    """
    failed_ids = machines.loc[machines["STATUS"] == "FAILED", "MACHINE_ID"]
    healthy_rows = sensors["MACHINE_ID"].isin(failed_ids) & (sensors["TIME_CYCLE"] <= config.HEALTHY_BASELINE_CYCLES)
    healthy_z = zscore_by_regime(sensors, regime_sensor_stats(sensors, rows=healthy_rows))

    hi_model = fit_health_index(healthy_z, failures)
    df = healthy_z[["MACHINE_ID", "TIME_CYCLE"]].copy()
    df["IN_SPEC"] = (healthy_z[config.INFORMATIVE_SENSORS].abs() <= config.IN_SPEC_Z).all(axis=1)
    df["HEALTH_INDEX"] = np.clip(hi_model.predict(healthy_z[config.INFORMATIVE_SENSORS]), 0, 1)
    df["PERIOD_INDEX"] = (df["TIME_CYCLE"] - 1) // config.PERIOD_CYCLES

    life = failure_cycles(failures)
    df["IS_FAILURE_CYCLE"] = df["TIME_CYCLE"] == df["MACHINE_ID"].map(life)

    periods = df.groupby(["MACHINE_ID", "PERIOD_INDEX"]).agg(
        START_CYCLE=("TIME_CYCLE", "min"),
        END_CYCLE=("TIME_CYCLE", "max"),
        CYCLES_FLOWN=("TIME_CYCLE", "count"),
        IN_SPEC_CYCLES=("IN_SPEC", "sum"),
        AVG_HEALTH_INDEX=("HEALTH_INDEX", "mean"),
        FAILURE_FLAG=("IS_FAILURE_CYCLE", "any"),
    ).reset_index()
    periods["AVG_HEALTH_INDEX"] = periods["AVG_HEALTH_INDEX"].round(4)
    return periods


def build_all_tables(raw_dir: Path, fleets=tuple(config.CMAPSS_FLEETS)) -> dict[str, pd.DataFrame]:
    parts = [load_fleet(raw_dir, f) for f in fleets]
    machines = pd.concat([p["machines"] for p in parts], ignore_index=True)
    sensors = pd.concat([p["sensors"] for p in parts], ignore_index=True)
    failures = pd.concat([p["failures"] for p in parts], ignore_index=True)
    ground_truth = pd.concat([p["ground_truth"] for p in parts], ignore_index=True)

    # Feature normalization uses only failed (historical) engines' readings, so
    # nothing about the in-service fleet leaks into what the model is trained on.
    failed_ids = machines.loc[machines["STATUS"] == "FAILED", "MACHINE_ID"]
    stats = regime_sensor_stats(sensors, rows=sensors["MACHINE_ID"].isin(failed_ids))

    return {
        "machine_data": machines,
        "raw_sensor_data": sensors,
        "maintenance_history": failures,
        "operating_periods": operating_periods(sensors, machines, failures),
        "regime_sensor_stats": stats,
        "fleet_ground_truth": ground_truth,
    }
