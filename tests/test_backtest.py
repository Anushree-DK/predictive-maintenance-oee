import numpy as np
import pandas as pd
import pytest

from src.ml.backtest import fixed_interval_policy, predictive_policy, summarize


def _oof(engine, fleet, life, lower_by_cycle):
    """Out-of-fold rows for one engine; lower_by_cycle(cycle) gives RUL_LOWER."""
    cycles = np.arange(1, life + 1)
    return pd.DataFrame({
        "MACHINE_ID": engine, "FLEET": fleet, "TIME_CYCLE": cycles, "TRUE_RUL": life - cycles,
        "RUL_LOWER": [lower_by_cycle(c) for c in cycles],
    })


def test_predictive_policy_acts_at_first_trigger_and_judges_by_true_rul():
    oof = pd.concat([
        _oof("EARLY", "F", 100, lambda c: 100 - c - 5),   # lower bound hits 15 at cycle 80 -> 20 cycles warning
        _oof("LATE", "F", 100, lambda c: 100 - c + 10),   # lower bound lags: hits 15 at cycle 95 -> 5 cycles warning
        _oof("NEVER", "F", 100, lambda c: 125),           # never triggers
    ])
    out = predictive_policy(oof, threshold=15, lead=10)
    assert out.loc["EARLY", "RUL_AT_ACTION"] == 20 and out.loc["EARLY", "CAUGHT"]
    assert out.loc["LATE", "RUL_AT_ACTION"] == 5 and not out.loc["LATE", "CAUGHT"]
    assert np.isnan(out.loc["NEVER", "RUL_AT_ACTION"]) and not out.loc["NEVER", "CAUGHT"]
    assert out.loc["LATE", "WASTED_CYCLES"] == 0  # a miss wastes no life — it fails


def test_fixed_interval_sets_age_per_fleet_from_failure_quantile():
    lives = pd.DataFrame({"MACHINE_ID": ["A", "B", "C", "D"], "FLEET": ["X", "X", "Y", "Y"], "LIFE": [100, 200, 300, 300]})
    out = fixed_interval_policy(lives, miss_rate=0.0, lead=10)
    assert out.loc["A", "OVERHAUL_AGE"] == 90 and out.loc["C", "OVERHAUL_AGE"] == 290
    assert out["CAUGHT"].all()
    assert out.loc["B", "WASTED_CYCLES"] == 110


def test_summary_counts_shop_visits_per_cycles_flown():
    outcomes = pd.DataFrame({"RUL_AT_ACTION": [20.0, np.nan], "CAUGHT": [True, False], "WASTED_CYCLES": [20.0, 0.0]},
                            index=pd.Index(["A", "B"], name="MACHINE_ID"))
    lives = pd.Series({"A": 100, "B": 100})
    summary = summarize(outcomes, lives)
    assert summary["catch_rate"] == 0.5 and summary["unplanned_failures"] == 1
    assert summary["mean_life_left_unused_cycles"] == 20.0
    assert summary["shop_visits_per_100k_cycles"] == pytest.approx(100_000 * 2 / 180, abs=0.1)
