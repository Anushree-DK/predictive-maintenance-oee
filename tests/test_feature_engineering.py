import numpy as np
import pandas as pd

from src.feature_engineering import ROLLING_WINDOW, build_feature_table, build_features_for_group


def _sensor_frame(n_cycles, slope=1.0, intercept=500.0, machine_id="M-001"):
    cycles = np.arange(1, n_cycles + 1)
    return pd.DataFrame({
        "MACHINE_ID": machine_id,
        "TIME_CYCLE": cycles,
        "SENSOR_1": intercept + slope * cycles,
    })


def test_build_features_for_group_slope_matches_known_linear_trend():
    group = _sensor_frame(n_cycles=20, slope=2.0, intercept=500.0)
    feats = build_features_for_group(group)
    assert round(feats["SENSOR_1_SLOPE"], 6) == 2.0


def test_build_features_for_group_flat_signal_has_zero_slope():
    group = _sensor_frame(n_cycles=20, slope=0.0, intercept=500.0)
    feats = build_features_for_group(group)
    assert round(feats["SENSOR_1_SLOPE"], 6) == 0.0
    assert round(feats["SENSOR_1_STD"], 6) == 0.0


def test_build_features_for_group_last_value_is_most_recent_cycle():
    group = _sensor_frame(n_cycles=20, slope=1.0, intercept=500.0)
    feats = build_features_for_group(group)
    assert feats["SENSOR_1_LAST"] == 500.0 + 20


def test_build_features_for_group_only_uses_rolling_window_not_full_history():
    # first half of history is flat, second half (within the rolling window) ramps up
    group = _sensor_frame(n_cycles=ROLLING_WINDOW * 2, slope=0.0, intercept=500.0)
    ramp_start = len(group) - ROLLING_WINDOW
    group.loc[ramp_start:, "SENSOR_1"] = 500.0 + np.arange(ROLLING_WINDOW)
    feats = build_features_for_group(group)
    assert feats["SENSOR_1_SLOPE"] > 0


def test_build_feature_table_one_row_per_machine():
    df = pd.concat([
        _sensor_frame(n_cycles=20, machine_id="M-001"),
        _sensor_frame(n_cycles=15, machine_id="M-002"),
    ], ignore_index=True)
    table = build_feature_table(df, group_col="MACHINE_ID")
    assert sorted(table["MACHINE_ID"]) == ["M-001", "M-002"]
    assert len(table) == 2
