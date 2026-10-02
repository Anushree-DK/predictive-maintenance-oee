"""Regime-normalized rolling sensor features, in pandas and as equivalent Snowflake SQL."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.cmapss import zscore_by_regime

ROLLING_WINDOW = 15
SUFFIXES = ("_MEAN", "_STD", "_SLOPE", "_LAST")


def feature_columns() -> list[str]:
    return ["TIME_CYCLE"] + [f"{s}{suffix}" for s in config.SENSOR_COLUMNS for suffix in SUFFIXES]


def build_feature_history(
    raw_sensor_df: pd.DataFrame, stats: pd.DataFrame, group_col: str = "MACHINE_ID"
) -> pd.DataFrame:
    """One feature row per (group, cycle), each using only that cycle and the
    ROLLING_WINDOW - 1 before it — what training needs, and causal by construction."""
    df = zscore_by_regime(raw_sensor_df, stats).sort_values([group_col, "TIME_CYCLE"]).reset_index(drop=True)
    x = df["TIME_CYCLE"].astype(float)
    keys = df[group_col]

    def rolling_sum(series: pd.Series) -> pd.Series:
        return series.groupby(keys).rolling(ROLLING_WINDOW, min_periods=1).sum().reset_index(level=0, drop=True)

    n = rolling_sum(pd.Series(1.0, index=df.index))
    sum_x, sum_xx = rolling_sum(x), rolling_sum(x * x)
    slope_denominator = (n * sum_xx - sum_x * sum_x).replace(0, np.nan)

    features = {group_col: df[group_col], "TIME_CYCLE": df["TIME_CYCLE"]}
    for col in config.SENSOR_COLUMNS:
        y = df[col]
        sum_y, sum_yy, sum_xy = rolling_sum(y), rolling_sum(y * y), rolling_sum(x * y)
        mean = sum_y / n
        features[f"{col}_MEAN"] = mean
        features[f"{col}_STD"] = np.sqrt((sum_yy / n - mean * mean).clip(lower=0))
        features[f"{col}_SLOPE"] = ((n * sum_xy - sum_x * sum_y) / slope_denominator).fillna(0.0)
        features[f"{col}_LAST"] = y
    return pd.DataFrame(features)


def build_feature_table(
    raw_sensor_df: pd.DataFrame, stats: pd.DataFrame, group_col: str = "MACHINE_ID"
) -> pd.DataFrame:
    """Latest feature row per group — the machine's current condition."""
    history = build_feature_history(raw_sensor_df, stats, group_col=group_col)
    return history.groupby(group_col).tail(1).reset_index(drop=True)


def build_feature_sql(
    group_col: str = "MACHINE_ID",
    sensor_table: str = "RAW_SENSOR_DATA",
    stats_table: str = "REGIME_SENSOR_STATS",
    where: str | None = None,
    latest_only: bool = True,
) -> str:
    """SQL version of build_feature_history.

    Slope and std are expanded from SUM/COUNT since Snowflake's regression window functions need unbounded frames."""
    window = f"PARTITION BY {group_col} ORDER BY TIME_CYCLE ROWS BETWEEN {ROLLING_WINDOW - 1} PRECEDING AND CURRENT ROW"
    n = f"COUNT(*) OVER ({window})"
    sum_x = f"SUM(TIME_CYCLE) OVER ({window})"
    sum_xx = f"SUM(TIME_CYCLE * TIME_CYCLE) OVER ({window})"

    z_exprs, feature_exprs = [], []
    for col in config.SENSOR_COLUMNS:
        z_exprs.append(f"(r.{col} - s.{col}_MEAN) / s.{col}_STD AS {col}")
        sum_y = f"SUM({col}) OVER ({window})"
        mean = f"({sum_y} / {n})"
        slope = f"({n} * SUM(TIME_CYCLE * {col}) OVER ({window}) - {sum_x} * {sum_y}) / NULLIF({n} * {sum_xx} - {sum_x} * {sum_x}, 0)"
        feature_exprs += [
            f"{mean} AS {col}_MEAN",
            f"SQRT(GREATEST(SUM({col} * {col}) OVER ({window}) / {n} - {mean} * {mean}, 0)) AS {col}_STD",
            f"COALESCE({slope}, 0) AS {col}_SLOPE",
            f"{col} AS {col}_LAST",
        ]

    qualify = f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {group_col} ORDER BY TIME_CYCLE DESC) = 1" if latest_only else ""
    return f"""
        WITH z AS (
            SELECT r.{group_col}, r.TIME_CYCLE, {", ".join(z_exprs)}
            FROM {sensor_table} r
            JOIN {stats_table} s ON s.REGIME = ROUND(r.OP_SETTING_1)
            {f"WHERE {where}" if where else ""}
        )
        SELECT {group_col}, TIME_CYCLE, {", ".join(feature_exprs)}
        FROM z
        {qualify}
        ORDER BY {group_col}
    """


def get_feature_table(group_col: str = "MACHINE_ID") -> pd.DataFrame:
    """Current features for the in-service fleet. Pushes down to Snowflake SQL when
    SNOWFLAKE_MODE=snowflake, otherwise computes the same features in pandas."""
    from src import data_access

    if config.SNOWFLAKE_MODE == "snowflake":
        from src.connection import get_session

        where = f"r.{group_col} IN (SELECT MACHINE_ID FROM MACHINE_DATA WHERE STATUS = 'IN_SERVICE')"
        return get_session().sql(build_feature_sql(group_col=group_col, where=where)).to_pandas()

    sensors = data_access.load_in_service_sensor_data()
    return build_feature_table(sensors, data_access.load_regime_sensor_stats(), group_col=group_col)
