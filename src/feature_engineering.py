"""Turns raw per-cycle sensor readings into one feature row per machine
(rolling stats + degradation slope), which is what the RUL model consumes.
"""

import numpy as np
import pandas as pd

import config

ROLLING_WINDOW = 15


def _degradation_slope(series: pd.Series) -> float:
    """Linear-fit slope of a sensor over its most recent cycles — a simple,
    interpretable stand-in for 'is this sensor trending worse.'"""
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series))
    slope, _ = np.polyfit(x, series.to_numpy(), 1)
    return float(slope)


def build_features_for_group(group: pd.DataFrame) -> dict:
    group = group.sort_values("TIME_CYCLE")
    recent = group.tail(ROLLING_WINDOW)

    features = {"TIME_CYCLE": int(group["TIME_CYCLE"].max())}
    for col in config.SENSOR_COLUMNS:
        if col not in group.columns:
            continue
        features[f"{col}_MEAN"] = float(recent[col].mean())
        features[f"{col}_STD"] = float(recent[col].std(ddof=0))
        features[f"{col}_SLOPE"] = _degradation_slope(recent[col])
        features[f"{col}_LAST"] = float(group[col].iloc[-1])
    return features


def build_feature_table(raw_sensor_df: pd.DataFrame, group_col: str = "MACHINE_ID") -> pd.DataFrame:
    """One row per group_col (machine or training unit) summarizing its latest condition."""
    rows = []
    for key, group in raw_sensor_df.groupby(group_col):
        feats = build_features_for_group(group)
        feats[group_col] = key
        rows.append(feats)
    df = pd.DataFrame(rows)
    id_cols = [group_col, "TIME_CYCLE"]
    return df[id_cols + [c for c in df.columns if c not in id_cols]]


def build_feature_table_sql(group_col: str = "MACHINE_ID", table_name: str = "RAW_SENSOR_DATA") -> pd.DataFrame:
    """Same rolling mean/std/slope/last features as build_feature_table, computed as SQL
    window functions in Snowflake instead of pulling raw rows into pandas first. Only used
    when SNOWFLAKE_MODE=snowflake.

    The slope is the closed-form least-squares formula, not REGR_SLOPE: Snowflake's
    regression window functions only support unbounded frames, not a bounded rolling
    window (ROWS BETWEEN n PRECEDING), so it's expanded from SUM()/COUNT() aggregates,
    which do support bounded frames.
    """
    from src.connection import get_session

    window = f"PARTITION BY {group_col} ORDER BY TIME_CYCLE ROWS BETWEEN {ROLLING_WINDOW - 1} PRECEDING AND CURRENT ROW"
    n = f"COUNT(*) OVER ({window})"
    sum_x = f"SUM(TIME_CYCLE) OVER ({window})"
    sum_xx = f"SUM(TIME_CYCLE * TIME_CYCLE) OVER ({window})"

    sensor_exprs = []
    for col in config.SENSOR_COLUMNS:
        sum_y = f"SUM({col}) OVER ({window})"
        sum_xy = f"SUM(TIME_CYCLE * {col}) OVER ({window})"
        slope = f"({n} * {sum_xy} - {sum_x} * {sum_y}) / NULLIF({n} * {sum_xx} - {sum_x} * {sum_x}, 0)"
        sensor_exprs.append(f"AVG({col}) OVER ({window}) AS {col}_MEAN")
        sensor_exprs.append(f"STDDEV_POP({col}) OVER ({window}) AS {col}_STD")
        sensor_exprs.append(f"{slope} AS {col}_SLOPE")
        sensor_exprs.append(f"{col} AS {col}_LAST")

    query = f"""
        SELECT {group_col}, TIME_CYCLE, {", ".join(sensor_exprs)}
        FROM {table_name}
        QUALIFY ROW_NUMBER() OVER (PARTITION BY {group_col} ORDER BY TIME_CYCLE DESC) = 1
    """
    return get_session().sql(query).to_pandas()


def get_feature_table(group_col: str = "MACHINE_ID") -> pd.DataFrame:
    """Entry point pipeline code should call: pushes down to Snowflake SQL when
    SNOWFLAKE_MODE=snowflake, otherwise computes the same features in pandas."""
    if config.SNOWFLAKE_MODE == "snowflake":
        return build_feature_table_sql(group_col=group_col)

    from src import data_access

    return build_feature_table(data_access.load_raw_sensor_data(), group_col=group_col)
