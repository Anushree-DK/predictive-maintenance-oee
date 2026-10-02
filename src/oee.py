"""OEE per operating period: availability x performance (health index) x quality (in-spec share)."""

import pandas as pd

import config


def compute_oee(periods_df: pd.DataFrame) -> pd.DataFrame:
    df = periods_df.copy()

    flight_hours = df["CYCLES_FLOWN"] * config.HOURS_PER_CYCLE
    downtime_hours = df["FAILURE_FLAG"].astype(bool) * config.AVG_UNPLANNED_REPAIR_HOURS
    df["AVAILABILITY"] = flight_hours / (flight_hours + downtime_hours)
    df["PERFORMANCE"] = df["AVG_HEALTH_INDEX"].clip(0.0, 1.0)
    df["QUALITY"] = df["IN_SPEC_CYCLES"] / df["CYCLES_FLOWN"]

    df["OEE"] = df["AVAILABILITY"] * df["PERFORMANCE"] * df["QUALITY"]
    return df


OEE_COLUMNS = ["AVAILABILITY", "PERFORMANCE", "QUALITY", "OEE"]


def latest_oee_by_machine(periods_df: pd.DataFrame, lookback_periods: int = 3) -> pd.DataFrame:
    """Each machine's OEE averaged over its most recent periods."""
    scored = compute_oee(periods_df).sort_values(["MACHINE_ID", "PERIOD_INDEX"])
    recent = scored.groupby("MACHINE_ID").tail(lookback_periods)
    return recent.groupby("MACHINE_ID")[OEE_COLUMNS].mean().reset_index()


def oee_by_fleet(periods_df: pd.DataFrame, machines_df: pd.DataFrame) -> pd.DataFrame:
    """Lifetime OEE per fleet across every period of every engine, failed or not."""
    scored = compute_oee(periods_df).merge(machines_df[["MACHINE_ID", "FLEET"]], on="MACHINE_ID")
    return scored.groupby("FLEET")[OEE_COLUMNS].mean().reset_index()
