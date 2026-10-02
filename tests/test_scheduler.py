import numpy as np
import pandas as pd
import pytest

import config
from src.scheduler import failure_cdf, optimize_schedule, planned_cost, unplanned_cost


def _fleet(rows):
    """rows: (machine_id, rul, interval_half_width)"""
    return pd.DataFrame(
        [{"MACHINE_ID": m, "RUL_PREDICTION": r, "RUL_LOWER": max(r - w, 0), "RUL_UPPER": r + w, "FAILURE_PROBABILITY": 0.0}
         for m, r, w in rows]
    )


def test_failure_cdf_is_half_at_the_point_prediction_and_increasing():
    cdf = failure_cdf(_fleet([("A", 20, 10)]), [10, 20, 30])[0]
    assert cdf[1] == pytest.approx(0.5)
    assert cdf[0] < cdf[1] < cdf[2]


def test_healthy_engines_are_not_scheduled():
    result = optimize_schedule(_fleet([("HEALTHY", 125, 5)]), capacity=5, n_slots=3, slot_cycles=5)
    assert result["schedule"].empty
    assert result["expected_cost_optimized"] == pytest.approx(result["expected_cost_do_nothing"])


def test_engines_beyond_saving_are_grounded_not_scheduled():
    result = optimize_schedule(_fleet([("FAILING", 1, 2), ("AT_RISK", 12, 4)]), capacity=5, n_slots=3, slot_cycles=5)
    assert list(result["ground_now"]["MACHINE_ID"]) == ["FAILING"]
    assert list(result["schedule"]["MACHINE_ID"]) == ["AT_RISK"]


def test_capacity_is_respected_and_most_valuable_engines_win():
    # slot 2 comes too late for these engines, so only slot 1 is used
    fleet = _fleet([(f"E{i}", rul, 3) for i, rul in enumerate([20, 22, 24, 26, 28, 40])])
    result = optimize_schedule(fleet, capacity=2, n_slots=2, slot_cycles=15)
    schedule = result["schedule"]
    assert len(schedule) == 2
    assert (schedule["SLOT"] == 1).all()
    assert set(schedule["MACHINE_ID"]) <= {"E0", "E1", "E2", "E3"}


def test_servicing_only_pays_when_failure_risk_exceeds_the_cost_ratio():
    # a 5% failure risk doesn't justify a planned repair
    result = optimize_schedule(_fleet([("A", 8, 3)]), capacity=1, n_slots=1, slot_cycles=5)
    assert result["schedule"].empty


def test_optimized_is_never_worse_than_doing_nothing_or_worst_first():
    rng = np.random.default_rng(0)
    fleet = _fleet([(f"E{i}", float(r), float(w)) for i, (r, w) in
                    enumerate(zip(rng.uniform(5, 125, 60), rng.uniform(3, 30, 60)))])
    result = optimize_schedule(fleet, capacity=3, n_slots=4, slot_cycles=5)
    assert result["expected_cost_optimized"] <= result["expected_cost_do_nothing"] + 1e-6
    assert result["expected_cost_optimized"] <= result["expected_cost_worst_first"] + 1e-6


def test_expected_saving_matches_cost_model():
    fleet = _fleet([("A", 20, 4)])
    result = optimize_schedule(fleet, capacity=1, n_slots=2, slot_cycles=15)
    row = result["schedule"].iloc[0]
    p_service = failure_cdf(fleet, [row["SERVICE_BY_CYCLE"]])[0, 0]
    p_horizon = failure_cdf(fleet, [30])[0, 0]
    expected = unplanned_cost() * p_horizon - (planned_cost() * (1 - p_service) + unplanned_cost() * p_service)
    assert row["EXPECTED_SAVING_USD"] == pytest.approx(expected, abs=1)
    assert config.SHOP_CAPACITY_PER_SLOT > 0
