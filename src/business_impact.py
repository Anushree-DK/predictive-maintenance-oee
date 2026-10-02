"""Converts failure probabilities into expected downtime cost."""

import pandas as pd

import config


def expected_cost_avoided(failure_probability: float) -> float:
    """P(failure) x extra hours of an unplanned repair x cost per downtime hour."""
    hours_saved = config.AVG_UNPLANNED_REPAIR_HOURS - config.AVG_PLANNED_REPAIR_HOURS
    return failure_probability * hours_saved * config.COST_PER_DOWNTIME_HOUR_USD


def add_expected_cost_column(predictions_df: pd.DataFrame) -> pd.DataFrame:
    df = predictions_df.copy()
    df["EXPECTED_COST_AVOIDED_USD"] = df["FAILURE_PROBABILITY"].apply(expected_cost_avoided)
    return df


def fleet_risk_exposure(predictions_df: pd.DataFrame) -> float:
    """Sum of expected downtime-cost exposure across the whole fleet right now."""
    return float(predictions_df["FAILURE_PROBABILITY"].apply(expected_cost_avoided).sum())
