import pandas as pd

from src.oee import compute_oee, latest_oee_by_machine


def _row(**overrides):
    row = {
        "MACHINE_ID": "M-001",
        "SHIFT_DATE": "2026-01-01",
        "SHIFT": "DAY",
        "PLANNED_PRODUCTION_TIME_MIN": 480,
        "DOWNTIME_MIN": 0,
        "IDEAL_CYCLE_TIME_SEC": 10.0,
        "TOTAL_COUNT": 2880,  # exactly fills 480 min at 10s/unit -> performance = 1.0
        "GOOD_COUNT": 2880,   # no defects -> quality = 1.0
    }
    row.update(overrides)
    return row


def test_compute_oee_perfect_shift_is_100_percent():
    df = pd.DataFrame([_row()])
    scored = compute_oee(df)
    assert scored.loc[0, "AVAILABILITY"] == 1.0
    assert scored.loc[0, "PERFORMANCE"] == 1.0
    assert scored.loc[0, "QUALITY"] == 1.0
    assert scored.loc[0, "OEE"] == 1.0


def test_compute_oee_downtime_reduces_availability():
    df = pd.DataFrame([_row(DOWNTIME_MIN=60, TOTAL_COUNT=2520, GOOD_COUNT=2520)])
    scored = compute_oee(df)
    assert scored.loc[0, "AVAILABILITY"] == (480 - 60) / 480
    assert round(scored.loc[0, "PERFORMANCE"], 4) == 1.0


def test_compute_oee_defects_reduce_quality():
    df = pd.DataFrame([_row(GOOD_COUNT=2736)])  # 5% defective
    scored = compute_oee(df)
    assert round(scored.loc[0, "QUALITY"], 4) == 0.95


def test_compute_oee_performance_is_capped_at_one():
    # TOTAL_COUNT implying faster-than-ideal throughput shouldn't push performance above 1.0
    df = pd.DataFrame([_row(TOTAL_COUNT=5000, GOOD_COUNT=5000)])
    scored = compute_oee(df)
    assert scored.loc[0, "PERFORMANCE"] == 1.0


def test_latest_oee_by_machine_averages_recent_shifts():
    rows = [
        _row(SHIFT_DATE=f"2026-01-{d:02d}", DOWNTIME_MIN=60 if d == 1 else 0,
             TOTAL_COUNT=2520 if d == 1 else 2880, GOOD_COUNT=2520 if d == 1 else 2880)
        for d in range(1, 4)
    ]
    df = pd.DataFrame(rows)
    result = latest_oee_by_machine(df, lookback_shifts=3)
    assert list(result["MACHINE_ID"]) == ["M-001"]
    assert 0.0 < result.loc[0, "AVAILABILITY"] < 1.0
