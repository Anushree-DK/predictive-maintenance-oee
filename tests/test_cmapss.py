"""Checks the C-MAPSS -> project-table derivation on a tiny hand-written subset in
NASA's file format (fixture data for the test, not used anywhere else)."""

import numpy as np
import pandas as pd
import pytest

import config
from src.cmapss import add_regime, build_all_tables, load_fleet, machine_id


def _write_subset(raw_dir, fleet="FD001", train_lengths=(6, 8), test_lengths=(3, 5), true_rul=(40, 7)):
    def rows(lengths):
        out = []
        for unit, n in enumerate(lengths, start=1):
            for cycle in range(1, n + 1):
                wear = cycle / n
                sensors = [500 + s + wear * (s % 3) for s in range(1, 22)]
                out.append([unit, cycle, 0.001, -0.0002, 100.0] + sensors)
        return out

    for kind, lengths in (("train", train_lengths), ("test", test_lengths)):
        pd.DataFrame(rows(lengths)).to_csv(raw_dir / f"{kind}_{fleet}.txt", sep=" ", header=False, index=False)
    pd.Series(true_rul).to_csv(raw_dir / f"RUL_{fleet}.txt", header=False, index=False)


def test_train_engines_become_failed_machines_with_failure_at_last_cycle(tmp_path):
    _write_subset(tmp_path)
    fleet = load_fleet(tmp_path, "FD001")
    failed = fleet["machines"][fleet["machines"]["STATUS"] == "FAILED"].set_index("MACHINE_ID")
    failures = fleet["failures"].set_index("MACHINE_ID")
    assert list(failed.index) == ["FD001-HIST-001", "FD001-HIST-002"]
    assert failures.loc["FD001-HIST-002", "EVENT_CYCLE"] == 8
    assert (failures["EVENT_CYCLE"] == failed.loc[failures.index, "CYCLES_OBSERVED"]).all()


def test_ground_truth_lines_up_with_test_engines_in_unit_order(tmp_path):
    _write_subset(tmp_path)
    truth = load_fleet(tmp_path, "FD001")["ground_truth"].set_index("MACHINE_ID")
    assert truth.loc["FD001-ENG-001", "TRUE_RUL"] == 40
    assert truth.loc["FD001-ENG-002", "TRUE_RUL"] == 7
    assert truth.loc["FD001-ENG-002", "LAST_OBSERVED_CYCLE"] == 5


def test_mismatched_rul_file_is_rejected(tmp_path):
    _write_subset(tmp_path, true_rul=(40,))
    with pytest.raises(ValueError, match="RUL_FD001"):
        load_fleet(tmp_path, "FD001")


def test_every_sensor_reading_is_kept_unchanged(tmp_path):
    _write_subset(tmp_path)
    sensors = load_fleet(tmp_path, "FD001")["sensors"]
    assert len(sensors) == 6 + 8 + 3 + 5
    raw = pd.read_csv(tmp_path / "train_FD001.txt", sep=r"\s+", header=None)
    first = sensors[sensors["MACHINE_ID"] == "FD001-HIST-001"].iloc[0]
    assert first["SENSOR_1"] == raw.iloc[0, 5]


def test_regime_is_rounded_altitude_setting():
    df = pd.DataFrame({"OP_SETTING_1": [0.0019, 41.998, 24.9996, 10.0047]})
    assert list(add_regime(df)["REGIME"]) == [0, 42, 25, 10]


def test_operating_periods_cover_every_cycle_and_flag_the_failure(tmp_path):
    _write_subset(tmp_path)
    tables = build_all_tables(tmp_path, fleets=("FD001",))
    periods = tables["operating_periods"]
    assert periods["CYCLES_FLOWN"].sum() == len(tables["raw_sensor_data"])
    assert periods["FAILURE_FLAG"].sum() == 2  # one per failed engine, none for in-service
    assert set(periods.loc[periods["FAILURE_FLAG"], "MACHINE_ID"]) == {"FD001-HIST-001", "FD001-HIST-002"}
    assert periods["AVG_HEALTH_INDEX"].between(0, 1).all()
    assert (periods["IN_SPEC_CYCLES"] <= periods["CYCLES_FLOWN"]).all()


def test_feature_stats_come_only_from_failed_engines(tmp_path):
    _write_subset(tmp_path)
    tables = build_all_tables(tmp_path, fleets=("FD001",))
    sensors, machines = tables["raw_sensor_data"], tables["machine_data"]
    failed = machines.loc[machines["STATUS"] == "FAILED", "MACHINE_ID"]
    expected = sensors[sensors["MACHINE_ID"].isin(failed)]["SENSOR_3"].mean()
    assert np.isclose(tables["regime_sensor_stats"].iloc[0]["SENSOR_3_MEAN"], expected)


def test_machine_id_format():
    assert machine_id("FD004", 7, "IN_SERVICE") == "FD004-ENG-007"
    assert machine_id("FD004", 7, "FAILED") == "FD004-HIST-007"
    assert set(config.CMAPSS_FLEETS) == {"FD001", "FD002", "FD003", "FD004"}
