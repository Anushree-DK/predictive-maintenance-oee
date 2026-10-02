import pandas as pd
import pytest

import config
from src.oee import compute_oee, latest_oee_by_machine, oee_by_fleet


def _period(**overrides):
    row = {
        "MACHINE_ID": "FD001-ENG-001",
        "PERIOD_INDEX": 0,
        "START_CYCLE": 1,
        "END_CYCLE": 10,
        "CYCLES_FLOWN": 10,
        "IN_SPEC_CYCLES": 10,
        "AVG_HEALTH_INDEX": 1.0,
        "FAILURE_FLAG": False,
    }
    row.update(overrides)
    return row


def test_healthy_period_without_failure_is_100_percent():
    scored = compute_oee(pd.DataFrame([_period()])).iloc[0]
    assert (scored["AVAILABILITY"], scored["PERFORMANCE"], scored["QUALITY"], scored["OEE"]) == (1.0, 1.0, 1.0, 1.0)


def test_failure_adds_unplanned_repair_downtime():
    scored = compute_oee(pd.DataFrame([_period(FAILURE_FLAG=True)])).iloc[0]
    flight_hours = 10 * config.HOURS_PER_CYCLE
    assert scored["AVAILABILITY"] == pytest.approx(flight_hours / (flight_hours + config.AVG_UNPLANNED_REPAIR_HOURS))


def test_out_of_spec_cycles_reduce_quality():
    assert compute_oee(pd.DataFrame([_period(IN_SPEC_CYCLES=7)])).iloc[0]["QUALITY"] == pytest.approx(0.7)


def test_performance_is_health_index_clipped_to_unit_interval():
    scored = compute_oee(pd.DataFrame([_period(AVG_HEALTH_INDEX=1.2), _period(AVG_HEALTH_INDEX=-0.1)]))
    assert list(scored["PERFORMANCE"]) == [1.0, 0.0]


def test_latest_oee_uses_only_the_most_recent_periods():
    periods = pd.DataFrame([
        _period(PERIOD_INDEX=0, IN_SPEC_CYCLES=0),   # old and bad: outside the lookback
        _period(PERIOD_INDEX=1),
        _period(PERIOD_INDEX=2, AVG_HEALTH_INDEX=0.5),
    ])
    result = latest_oee_by_machine(periods, lookback_periods=2).iloc[0]
    assert result["QUALITY"] == 1.0
    assert result["PERFORMANCE"] == pytest.approx(0.75)


def test_oee_by_fleet_groups_machines():
    periods = pd.DataFrame([_period(MACHINE_ID="A"), _period(MACHINE_ID="B", IN_SPEC_CYCLES=5)])
    machines = pd.DataFrame({"MACHINE_ID": ["A", "B"], "FLEET": ["FD001", "FD002"]})
    result = oee_by_fleet(periods, machines).set_index("FLEET")
    assert result.loc["FD001", "OEE"] == 1.0
    assert result.loc["FD002", "OEE"] == pytest.approx(0.5)
