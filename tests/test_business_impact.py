import pandas as pd

import config
from src.business_impact import add_expected_cost_column, expected_cost_avoided, fleet_risk_exposure


def test_expected_cost_avoided_zero_probability_is_zero():
    assert expected_cost_avoided(0.0) == 0.0


def test_expected_cost_avoided_scales_linearly_with_probability():
    low = expected_cost_avoided(0.1)
    high = expected_cost_avoided(0.5)
    assert round(high / low, 6) == 5.0


def test_expected_cost_avoided_matches_manual_formula():
    p = 0.4
    hours_saved = config.AVG_UNPLANNED_REPAIR_HOURS - config.AVG_PLANNED_REPAIR_HOURS
    expected = p * hours_saved * config.COST_PER_DOWNTIME_HOUR_USD
    assert expected_cost_avoided(p) == expected


def test_add_expected_cost_column_adds_column_without_mutating_input():
    df = pd.DataFrame({"FAILURE_PROBABILITY": [0.0, 0.5, 1.0]})
    result = add_expected_cost_column(df)
    assert "EXPECTED_COST_AVOIDED_USD" not in df.columns  # original untouched
    assert list(result["EXPECTED_COST_AVOIDED_USD"]) == [
        expected_cost_avoided(0.0), expected_cost_avoided(0.5), expected_cost_avoided(1.0)
    ]


def test_fleet_risk_exposure_sums_across_machines():
    df = pd.DataFrame({"FAILURE_PROBABILITY": [0.1, 0.2, 0.3]})
    expected = sum(expected_cost_avoided(p) for p in [0.1, 0.2, 0.3])
    assert fleet_risk_exposure(df) == expected
