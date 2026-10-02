import numpy as np
import pandas as pd
import pytest

import config
from src.feature_engineering import (
    ROLLING_WINDOW,
    build_feature_history,
    build_feature_sql,
    build_feature_table,
    feature_columns,
)


def _identity_stats(regimes=(0,)) -> pd.DataFrame:
    """Mean 0 / std 1 for every sensor, so z-scores equal the raw readings."""
    row = {f"{s}_MEAN": 0.0 for s in config.SENSOR_COLUMNS} | {f"{s}_STD": 1.0 for s in config.SENSOR_COLUMNS}
    return pd.DataFrame([{"REGIME": r, **row} for r in regimes])


def _sensor_frame(n_cycles, slope=1.0, intercept=500.0, machine_id="M-001", op_setting_1=0.0):
    cycles = np.arange(1, n_cycles + 1)
    df = pd.DataFrame({"MACHINE_ID": machine_id, "TIME_CYCLE": cycles, "OP_SETTING_1": op_setting_1,
                       "OP_SETTING_2": 0.0, "OP_SETTING_3": 100.0})
    for col in config.SENSOR_COLUMNS:
        df[col] = 0.0
    df["SENSOR_1"] = intercept + slope * cycles
    return df


def _latest(df, stats=None):
    return build_feature_table(df, _identity_stats() if stats is None else stats).iloc[0]


def test_slope_matches_known_linear_trend():
    assert round(_latest(_sensor_frame(20, slope=2.0))["SENSOR_1_SLOPE"], 6) == 2.0


def test_flat_signal_has_zero_slope_and_std():
    feats = _latest(_sensor_frame(20, slope=0.0))
    assert round(feats["SENSOR_1_SLOPE"], 6) == 0.0
    assert round(feats["SENSOR_1_STD"], 6) == 0.0


def test_last_value_is_most_recent_cycle():
    assert _latest(_sensor_frame(20, slope=1.0, intercept=500.0))["SENSOR_1_LAST"] == 520.0


def test_only_uses_rolling_window_not_full_history():
    group = _sensor_frame(ROLLING_WINDOW * 2, slope=0.0)
    group.loc[ROLLING_WINDOW:, "SENSOR_1"] = 500.0 + np.arange(ROLLING_WINDOW)
    feats = _latest(group)
    assert round(feats["SENSOR_1_SLOPE"], 6) == 1.0
    assert round(feats["SENSOR_1_MEAN"], 6) == 500.0 + (ROLLING_WINDOW - 1) / 2


def test_sensors_are_zscored_against_their_operating_regime():
    stats = _identity_stats(regimes=(0, 42))
    stats.loc[stats["REGIME"] == 42, "SENSOR_1_MEAN"] = 100.0
    stats.loc[stats["REGIME"] == 42, "SENSOR_1_STD"] = 2.0
    feats = _latest(_sensor_frame(5, slope=0.0, intercept=110.0, op_setting_1=41.99), stats)
    assert feats["SENSOR_1_LAST"] == 5.0  # (110 - 100) / 2


def test_unknown_regime_is_rejected_not_silently_nan():
    with pytest.raises(ValueError, match="regime"):
        _latest(_sensor_frame(5, op_setting_1=20.0))


def test_history_is_causal_one_row_per_cycle():
    df = _sensor_frame(20, slope=1.0)
    history = build_feature_history(df, _identity_stats())
    assert len(history) == 20
    truncated = build_feature_history(df[df["TIME_CYCLE"] <= 10], _identity_stats())
    pd.testing.assert_frame_equal(history.iloc[:10].reset_index(drop=True), truncated, check_exact=False)


def test_one_row_per_machine_with_model_columns():
    df = pd.concat([_sensor_frame(20, machine_id="M-001"), _sensor_frame(15, machine_id="M-002")], ignore_index=True)
    table = build_feature_table(df, _identity_stats())
    assert sorted(table["MACHINE_ID"]) == ["M-001", "M-002"]
    assert set(feature_columns()) <= set(table.columns)


def test_sql_features_match_pandas_features():
    """The Snowflake SQL path must produce the same numbers the model was trained on.
    Runs the generated SQL on DuckDB, which shares the window-function syntax used."""
    duckdb = pytest.importorskip("duckdb")
    rng = np.random.default_rng(0)
    frames = []
    for machine_id, n, regime in (("M-001", 40, 0.0), ("M-002", 9, 42.0)):
        df = _sensor_frame(n, machine_id=machine_id, op_setting_1=regime)
        for col in config.SENSOR_COLUMNS:
            df[col] = 500 + rng.normal(size=n).cumsum()
        frames.append(df)
    sensors = pd.concat(frames, ignore_index=True)
    stats = _identity_stats(regimes=(0, 42))
    for col in config.SENSOR_COLUMNS:
        stats[f"{col}_MEAN"] = [480.0, 510.0]
        stats[f"{col}_STD"] = [3.0, 7.0]

    con = duckdb.connect()
    con.register("RAW_SENSOR_DATA", sensors)
    con.register("REGIME_SENSOR_STATS", stats)

    for latest_only in (True, False):
        sql_result = con.execute(build_feature_sql(latest_only=latest_only)).df()
        expected = (build_feature_table if latest_only else build_feature_history)(sensors, stats)
        sql_result = sql_result.sort_values(["MACHINE_ID", "TIME_CYCLE"]).reset_index(drop=True)
        expected = expected.sort_values(["MACHINE_ID", "TIME_CYCLE"]).reset_index(drop=True)
        cols = ["MACHINE_ID"] + feature_columns()
        pd.testing.assert_frame_equal(sql_result[cols], expected[cols], check_dtype=False, rtol=1e-6, atol=1e-6)
