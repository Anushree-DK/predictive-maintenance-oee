"""Translates technical predictions into a dollar figure, so the dashboard
speaks to a plant manager's actual question ("what does this cost us?"),
not just a data scientist's ("what's the RUL?").

See config.py for the assumptions this is built on and what's a cited
industry benchmark vs. an explicit, editable placeholder.
"""

import pandas as pd

import config


def expected_cost_avoided(failure_probability: float) -> float:
    """Expected value, in dollars, of acting on this prediction now instead
    of waiting for an unplanned failure: P(failure) x (hours an emergency
    repair costs beyond a scheduled one) x $/hour of downtime.
    """
    hours_saved = config.AVG_UNPLANNED_REPAIR_HOURS - config.AVG_PLANNED_REPAIR_HOURS
    return failure_probability * hours_saved * config.COST_PER_DOWNTIME_HOUR_USD


def add_expected_cost_column(predictions_df: pd.DataFrame) -> pd.DataFrame:
    df = predictions_df.copy()
    df["EXPECTED_COST_AVOIDED_USD"] = df["FAILURE_PROBABILITY"].apply(expected_cost_avoided)
    return df


def fleet_risk_exposure(predictions_df: pd.DataFrame) -> float:
    """Sum of expected downtime-cost exposure across the whole fleet right now."""
    return float(predictions_df["FAILURE_PROBABILITY"].apply(expected_cost_avoided).sum())
