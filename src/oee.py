"""Availability x Performance x Quality, aggregated per machine over its most recent shifts."""

import pandas as pd


def compute_oee(production_df: pd.DataFrame) -> pd.DataFrame:
    df = production_df.copy()

    run_time_min = df["PLANNED_PRODUCTION_TIME_MIN"] - df["DOWNTIME_MIN"]
    df["AVAILABILITY"] = run_time_min / df["PLANNED_PRODUCTION_TIME_MIN"]

    ideal_run_time_min = (df["IDEAL_CYCLE_TIME_SEC"] * df["TOTAL_COUNT"]) / 60
    df["PERFORMANCE"] = (ideal_run_time_min / run_time_min).clip(upper=1.0)

    df["QUALITY"] = df["GOOD_COUNT"] / df["TOTAL_COUNT"]

    df["OEE"] = df["AVAILABILITY"] * df["PERFORMANCE"] * df["QUALITY"]
    return df


def latest_oee_by_machine(production_df: pd.DataFrame, lookback_shifts: int = 6) -> pd.DataFrame:
    scored = compute_oee(production_df)
    scored = scored.sort_values(["MACHINE_ID", "SHIFT_DATE"])

    def _tail_mean(group: pd.DataFrame) -> pd.Series:
        recent = group.tail(lookback_shifts)
        return pd.Series(
            {
                "AVAILABILITY": recent["AVAILABILITY"].mean(),
                "PERFORMANCE": recent["PERFORMANCE"].mean(),
                "QUALITY": recent["QUALITY"].mean(),
                "OEE": recent["OEE"].mean(),
            }
        )

    return scored.groupby("MACHINE_ID").apply(_tail_mean, include_groups=False).reset_index()
