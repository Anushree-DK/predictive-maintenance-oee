"""Assigns engines to limited shop slots to minimize expected downtime cost (MILP)."""

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import norm

import config

Z_90 = norm.ppf(0.95)  # half-width of a two-sided 90% interval, in sigmas


def planned_cost() -> float:
    return config.AVG_PLANNED_REPAIR_HOURS * config.COST_PER_DOWNTIME_HOUR_USD


def unplanned_cost() -> float:
    return config.AVG_UNPLANNED_REPAIR_HOURS * config.COST_PER_DOWNTIME_HOUR_USD


def failure_cdf(predictions: pd.DataFrame, cycles) -> np.ndarray:
    """P(engine fails within `cycles`) per engine, shape (engines, len(cycles))."""
    mu = predictions["RUL_PREDICTION"].to_numpy()[:, None]
    sigma = np.maximum((predictions["RUL_UPPER"] - predictions["RUL_LOWER"]).to_numpy() / (2 * Z_90), 1.0)[:, None]
    return norm.cdf((np.asarray(cycles, dtype=float)[None, :] - mu) / sigma)


def optimize_schedule(
    predictions: pd.DataFrame,
    capacity: int = config.SHOP_CAPACITY_PER_SLOT,
    n_slots: int = config.PLANNING_HORIZON_SLOTS,
    slot_cycles: int = config.SHOP_SLOT_CYCLES,
) -> dict:
    """Returns engines to ground now, the optimal shop schedule for the rest, and the
    expected downtime cost (over the rest) of doing nothing, worst-first, and optimal."""
    service_cycles = slot_cycles * np.arange(1, n_slots + 1)
    p_first_slot = failure_cdf(predictions, service_cycles[:1])[:, 0]
    ground_now = predictions.loc[p_first_slot >= 0.5, ["MACHINE_ID", "RUL_PREDICTION", "RUL_LOWER", "FAILURE_PROBABILITY"]]
    predictions = predictions[p_first_slot < 0.5].reset_index(drop=True)

    horizon = service_cycles[-1]
    p_fail_by_service = failure_cdf(predictions, service_cycles)          # (engines, slots)
    p_fail_in_horizon = failure_cdf(predictions, [horizon])[:, 0]          # (engines,)

    cost_if_scheduled = planned_cost() * (1 - p_fail_by_service) + unplanned_cost() * p_fail_by_service
    cost_if_not = unplanned_cost() * p_fail_in_horizon

    # Only engines for which servicing could ever pay off are worth a variable.
    candidates = np.flatnonzero(cost_if_scheduled.min(axis=1) < cost_if_not)
    n, s = len(candidates), n_slots
    schedule = pd.DataFrame(columns=["SLOT", "SERVICE_BY_CYCLE", "MACHINE_ID", "P_FAIL_BEFORE_SERVICE", "EXPECTED_SAVING_USD"])
    chosen_cost = 0.0
    if n:
        # objective relative to scheduling nothing
        c = (cost_if_scheduled[candidates] - cost_if_not[candidates, None]).ravel()
        one_slot_per_engine = np.kron(np.eye(n), np.ones(s))
        slot_capacity = np.kron(np.ones(n), np.eye(s))
        result = milp(
            c,
            constraints=[LinearConstraint(one_slot_per_engine, 0, 1), LinearConstraint(slot_capacity, 0, capacity)],
            integrality=np.ones(n * s),
            bounds=Bounds(0, 1),
        )
        if not result.success:
            raise RuntimeError(f"schedule optimization failed: {result.message}")
        x = np.round(result.x).reshape(n, s).astype(bool)
        chosen_cost = float(c.reshape(n, s)[x].sum())
        rows = []
        for i, slot in zip(*np.nonzero(x)):
            engine = candidates[i]
            rows.append({
                "SLOT": int(slot) + 1,
                "SERVICE_BY_CYCLE": int(service_cycles[slot]),
                "MACHINE_ID": predictions["MACHINE_ID"].iloc[engine],
                "P_FAIL_BEFORE_SERVICE": round(float(p_fail_by_service[engine, slot]), 3),
                "EXPECTED_SAVING_USD": round(float(cost_if_not[engine] - cost_if_scheduled[engine, slot]), 0),
            })
        schedule = pd.DataFrame(rows).sort_values(["SLOT", "P_FAIL_BEFORE_SERVICE"], ascending=[True, False])

    do_nothing = float(cost_if_not.sum())
    return {
        "ground_now": ground_now.sort_values("RUL_PREDICTION").reset_index(drop=True),
        "schedule": schedule.reset_index(drop=True),
        "expected_cost_do_nothing": do_nothing,
        "expected_cost_optimized": do_nothing + chosen_cost,
        "expected_cost_worst_first": _worst_first_cost(predictions, cost_if_scheduled, cost_if_not, capacity, n_slots),
        "engines_at_risk_in_horizon": int((p_fail_in_horizon >= 0.5).sum()),
        "capacity_total": capacity * n_slots,
        "horizon_cycles": int(horizon),
    }


def _worst_first_cost(predictions, cost_if_scheduled, cost_if_not, capacity, n_slots) -> float:
    """Baseline: fill slots in order of lowest predicted RUL, regardless of value."""
    order = np.argsort(predictions["RUL_PREDICTION"].to_numpy(), kind="stable")[: capacity * n_slots]
    slots = np.arange(len(order)) // capacity
    return float(cost_if_not.sum() + (cost_if_scheduled[order, slots] - cost_if_not[order]).sum())


def check_distribution() -> dict:
    """How well the Normal approximation's P(fail within the 30-cycle horizon) matches
    what actually happened to NASA's test engines, next to the trained classifier's."""
    from sklearn.metrics import brier_score_loss, roc_auc_score

    from src.ml.predict import predict_for_features
    from src.ml.train import build_evaluation_set

    test = build_evaluation_set()
    preds = predict_for_features(test)
    failed = (test["TRUE_RUL"] <= config.FAILURE_HORIZON_CYCLES).astype(int).to_numpy()
    approx = failure_cdf(preds, [config.FAILURE_HORIZON_CYCLES])[:, 0]
    return {
        "normal_approximation": {"brier": round(float(brier_score_loss(failed, approx)), 4),
                                 "roc_auc": round(float(roc_auc_score(failed, approx)), 4)},
        "calibrated_classifier": {"brier": round(float(brier_score_loss(failed, preds["FAILURE_PROBABILITY"])), 4),
                                  "roc_auc": round(float(roc_auc_score(failed, preds["FAILURE_PROBABILITY"])), 4)},
    }


if __name__ == "__main__":
    import json

    print(json.dumps(check_distribution(), indent=2))
