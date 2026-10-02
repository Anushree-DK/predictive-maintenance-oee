"""Backtest: would acting on this model's predictions have beaten fixed-interval
maintenance on the real C-MAPSS failures?

Every failed engine is replayed cycle by cycle with *out-of-fold* predictions: the
709 failed engines are split into 5 folds by engine, and each fold is scored by
models (src/ml/train.py::fit_models) trained only on the other four. So no engine
is ever scored by a model that saw its own failure.

Policies, each judged against the engine's real failure cycle:
- predictive(L): send the engine to the shop at the first cycle where the lower
  bound of its 90% RUL interval is <= L cycles (L = 15 is the CRITICAL risk class);
- fixed-interval(T): overhaul every engine of a fleet at age T, with T set per fleet
  from that fleet's own failure-age distribution — a strong, in-sample baseline.

An intervention "catches" the failure if it happens at least LEAD_CYCLES before
the engine would have failed (time to get the engine into the shop). Life left on
the table by intervening early is the engine's true RUL at the intervention.

    python -m src.ml.backtest      # ~10 min; writes reports/backtest.json + backtest_curve.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.ml.predict import predict_with_bundle
from src.ml.train import build_training_set, fit_models

LEAD_CYCLES = 10
N_FOLDS = 5
THRESHOLDS = list(range(0, 61, 5))
OOF_PATH = config.LOCAL_DATA_DIR / "backtest_oof_predictions.csv"
SUMMARY_PATH = config.REPORTS_DIR / "backtest.json"
CURVE_PATH = config.REPORTS_DIR / "backtest_curve.csv"


def out_of_fold_predictions(train: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for fold, (fit_idx, held_idx) in enumerate(GroupKFold(n_splits=N_FOLDS).split(train, groups=train["MACHINE_ID"])):
        bundle = fit_models(train.iloc[fit_idx])
        held = train.iloc[held_idx]
        preds = predict_with_bundle(bundle, held, id_col=None)
        parts.append(pd.concat([held[["MACHINE_ID", "FLEET", "TIME_CYCLE", "TRUE_RUL"]].reset_index(drop=True), preds], axis=1))
        print(f"fold {fold + 1}/{N_FOLDS}: scored {held['MACHINE_ID'].nunique()} held-out engines")
    return pd.concat(parts, ignore_index=True)


def predictive_policy(oof: pd.DataFrame, threshold: float, lead: int = LEAD_CYCLES) -> pd.DataFrame:
    """Per engine: true RUL at the first cycle whose RUL lower bound <= threshold
    (NaN if the policy never triggered before failure)."""
    triggered = oof[oof["RUL_LOWER"] <= threshold]
    first = triggered.sort_values("TIME_CYCLE").groupby("MACHINE_ID").head(1).set_index("MACHINE_ID")["TRUE_RUL"]
    engines = oof.groupby("MACHINE_ID")["FLEET"].first().to_frame()
    engines["RUL_AT_ACTION"] = first.reindex(engines.index)
    return _outcomes(engines, lead)


def fixed_interval_policy(lives: pd.DataFrame, miss_rate: float, lead: int = LEAD_CYCLES) -> pd.DataFrame:
    """Per fleet, the overhaul age T at which `miss_rate` of that fleet's engines would
    fail (or come within `lead` cycles of failing) before T."""
    rows = []
    for fleet, group in lives.groupby("FLEET"):
        age = np.quantile(group["LIFE"], miss_rate) - lead
        engines = group.set_index("MACHINE_ID")[["FLEET"]].copy()
        engines["RUL_AT_ACTION"] = group.set_index("MACHINE_ID")["LIFE"] - age
        engines.loc[engines["RUL_AT_ACTION"] < 0, "RUL_AT_ACTION"] = np.nan  # failed before the overhaul
        engines["OVERHAUL_AGE"] = age
        rows.append(engines)
    return _outcomes(pd.concat(rows), lead)


def _outcomes(engines: pd.DataFrame, lead: int) -> pd.DataFrame:
    engines["CAUGHT"] = engines["RUL_AT_ACTION"] >= lead
    engines["WASTED_CYCLES"] = engines["RUL_AT_ACTION"].where(engines["CAUGHT"], 0.0)
    return engines


def summarize(outcomes: pd.DataFrame, lives: pd.Series) -> dict:
    caught = outcomes["CAUGHT"]
    life = lives.reindex(outcomes.index)
    life_used = 1 - outcomes["WASTED_CYCLES"] / life
    # Every engine makes exactly one shop visit (planned if caught, unplanned if not),
    # after flying its life minus whatever was left unused.
    cycles_flown = float((life - outcomes["WASTED_CYCLES"]).sum())
    return {
        "engines": int(len(outcomes)),
        "failures_caught": int(caught.sum()),
        "catch_rate": round(float(caught.mean()), 4),
        "unplanned_failures": int((~caught).sum()),
        "median_warning_cycles": float(outcomes.loc[caught, "RUL_AT_ACTION"].median()) if caught.any() else None,
        "mean_life_left_unused_cycles": round(float(outcomes.loc[caught, "WASTED_CYCLES"].mean()), 1) if caught.any() else None,
        "mean_share_of_life_used": round(float(life_used.mean()), 4),
        "shop_visits_per_100k_cycles": round(100_000 * len(outcomes) / cycles_flown, 1),
    }


def downtime_cost(summary: dict) -> float:
    """Downtime cost of a policy across the fleet, using config's cost assumptions:
    every caught failure is a planned repair, every missed one an unplanned repair."""
    rate = config.COST_PER_DOWNTIME_HOUR_USD
    return (
        summary["failures_caught"] * config.AVG_PLANNED_REPAIR_HOURS * rate
        + summary["unplanned_failures"] * config.AVG_UNPLANNED_REPAIR_HOURS * rate
    )


def run_backtest(oof: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    lives = oof.groupby("MACHINE_ID").agg(FLEET=("FLEET", "first"), LIFE=("TIME_CYCLE", "max")).reset_index()
    life_by_engine = lives.set_index("MACHINE_ID")["LIFE"]

    curve = []
    for threshold in THRESHOLDS:
        predictive = summarize(predictive_policy(oof, threshold), life_by_engine)
        miss_rate = 1 - predictive["catch_rate"]
        fixed = summarize(fixed_interval_policy(lives, miss_rate), life_by_engine)
        curve.append({
            "RUL_LOWER_THRESHOLD": threshold,
            "CATCH_RATE": predictive["catch_rate"],
            "PREDICTIVE_LIFE_LEFT_UNUSED": predictive["mean_life_left_unused_cycles"],
            "PREDICTIVE_SHARE_OF_LIFE_USED": predictive["mean_share_of_life_used"],
            "PREDICTIVE_SHOP_VISITS_PER_100K": predictive["shop_visits_per_100k_cycles"],
            "FIXED_INTERVAL_LIFE_LEFT_UNUSED": fixed["mean_life_left_unused_cycles"],
            "FIXED_INTERVAL_SHARE_OF_LIFE_USED": fixed["mean_share_of_life_used"],
            "FIXED_INTERVAL_SHOP_VISITS_PER_100K": fixed["shop_visits_per_100k_cycles"],
        })

    headline_threshold = config.RISK_THRESHOLDS["CRITICAL"]
    predictive = summarize(predictive_policy(oof, headline_threshold), life_by_engine)
    fixed = summarize(fixed_interval_policy(lives, 1 - predictive["catch_rate"]), life_by_engine)
    run_to_failure = {"engines": len(lives), "failures_caught": 0, "unplanned_failures": len(lives)}

    summary = {
        "method": (
            f"{N_FOLDS}-fold out-of-fold replay of all {len(lives)} real run-to-failure engines; "
            f"a failure counts as caught if the engine is pulled at least {LEAD_CYCLES} cycles before it fails."
        ),
        "headline_policy": f"pull an engine when its RUL lower bound <= {headline_threshold} cycles (CRITICAL)",
        "predictive": predictive,
        "fixed_interval_same_catch_rate": fixed,
        "life_left_unused_reduction": round(
            1 - predictive["mean_life_left_unused_cycles"] / fixed["mean_life_left_unused_cycles"], 3
        ),
        "shop_visit_reduction": round(
            1 - predictive["shop_visits_per_100k_cycles"] / fixed["shop_visits_per_100k_cycles"], 3
        ),
        "downtime_cost_usd": {
            "run_to_failure": downtime_cost(run_to_failure),
            "predictive": downtime_cost(predictive),
            "fixed_interval_same_catch_rate": downtime_cost(fixed),
        },
        "cost_assumptions": "config.py: $/downtime hour and planned/unplanned repair hours",
    }
    return summary, pd.DataFrame(curve)


def main():
    if OOF_PATH.exists() and "--refit" not in sys.argv:
        oof = pd.read_csv(OOF_PATH)
        print(f"using cached out-of-fold predictions {OOF_PATH} (pass --refit to recompute)")
    else:
        oof = out_of_fold_predictions(build_training_set())
        oof.to_csv(OOF_PATH, index=False)

    summary, curve = run_backtest(oof)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    curve.to_csv(CURVE_PATH, index=False)
    print(json.dumps(summary, indent=2))
    print(curve.to_string(index=False))


if __name__ == "__main__":
    main()
