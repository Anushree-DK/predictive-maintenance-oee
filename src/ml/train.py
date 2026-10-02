"""Trains the RUL models on the failed engines' run-to-failure histories and
evaluates them on NASA's official C-MAPSS test set (the in-service fleet, scored
against RUL_FD00X.txt). Writes models/rul_model.joblib and reports/model_metrics.json.

Three models, all HistGradientBoosting (no native-library dependency):
- point RUL: squared-error regressor on the RUL label capped at config.RUL_CAP;
- RUL interval: 5%/95% quantile regressors, widened by split-conformal calibration
  (conformalized quantile regression) so the interval's coverage is checked on
  held-out engines rather than assumed;
- failure probability: classifier for "fails within FAILURE_HORIZON_CYCLES",
  isotonic-calibrated on held-out engines, so 0.3 means about 30% of such engines fail.

Every split is by engine (a whole trajectory goes to one side), never by row: rows
of one engine are near-duplicates, and a row split lets the model memorize engines
it is then "tested" on. evaluate() reports how much that inflates the score.

    python -m src.ml.train
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src import data_access
from src.feature_engineering import build_feature_history, build_feature_table, feature_columns
from src.ml.predict import predict_with_bundle

MODEL_PATH = config.MODELS_DIR / "rul_model.joblib"
METRICS_PATH = config.REPORTS_DIR / "model_metrics.json"
CALIBRATION_FRACTION = 0.2
CONFORMAL_BAND_EDGES = [30, 60, 100]  # bands of predicted RUL, in cycles
SEED = 42

REGRESSOR_PARAMS = dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0)
# Quantile loss has sign-only gradients and converges slowly; these settings give
# intervals ~40% narrower near failure than REGRESSOR_PARAMS at the same coverage.
QUANTILE_PARAMS = dict(max_iter=1500, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=20)


def build_training_set() -> pd.DataFrame:
    """Feature row for every cycle of every failed engine, labeled with its RUL."""
    machines = data_access.load_machine_data()
    failed = machines.loc[machines["STATUS"] == "FAILED", ["MACHINE_ID", "FLEET"]]
    sensors = data_access.load_raw_sensor_data()
    sensors = sensors[sensors["MACHINE_ID"].isin(failed["MACHINE_ID"])]

    history = build_feature_history(sensors, data_access.load_regime_sensor_stats())
    failure_cycle = data_access.load_maintenance_history().set_index("MACHINE_ID")["EVENT_CYCLE"]
    history["TRUE_RUL"] = history["MACHINE_ID"].map(failure_cycle) - history["TIME_CYCLE"]
    history["RUL"] = history["TRUE_RUL"].clip(upper=config.RUL_CAP)
    return history.merge(failed, on="MACHINE_ID")


def build_evaluation_set() -> pd.DataFrame:
    """Latest features of each in-service engine, joined to NASA's true RUL."""
    machines = data_access.load_machine_data()
    in_service = machines.loc[machines["STATUS"] == "IN_SERVICE", ["MACHINE_ID", "FLEET"]]
    sensors = data_access.load_raw_sensor_data()
    sensors = sensors[sensors["MACHINE_ID"].isin(in_service["MACHINE_ID"])]
    latest = build_feature_table(sensors, data_access.load_regime_sensor_stats())
    return latest.merge(in_service, on="MACHINE_ID").merge(
        data_access.load_fleet_ground_truth()[["MACHINE_ID", "TRUE_RUL"]], on="MACHINE_ID"
    )


def _regressor(**overrides) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**{**REGRESSOR_PARAMS, "random_state": SEED, **overrides})


def fit_models(train: pd.DataFrame) -> dict:
    cols = feature_columns()
    alpha = 1 - config.PREDICTION_INTERVAL

    fit_idx, cal_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=CALIBRATION_FRACTION, random_state=SEED).split(train, groups=train["MACHINE_ID"])
    )
    fit, cal = train.iloc[fit_idx], train.iloc[cal_idx]

    point = _regressor().fit(train[cols], train["RUL"])
    point_fit = _regressor().fit(fit[cols], fit["RUL"])  # only to assign calibration rows to bands

    lower = _regressor(loss="quantile", quantile=alpha / 2, **QUANTILE_PARAMS).fit(fit[cols], fit["RUL"])
    upper = _regressor(loss="quantile", quantile=1 - alpha / 2, **QUANTILE_PARAMS).fit(fit[cols], fit["RUL"])
    # Conformalized quantile regression, Mondrian-style: one margin per band of the
    # point prediction, so engines near failure are calibrated against other engines
    # near failure instead of being averaged in with the (far more numerous) healthy ones.
    cal_point = point_fit.predict(cal[cols])
    scores = np.maximum(lower.predict(cal[cols]) - cal["RUL"], cal["RUL"] - upper.predict(cal[cols]))
    bands = np.digitize(cal_point, CONFORMAL_BAND_EDGES)
    conformal_margins = []
    for band in range(len(CONFORMAL_BAND_EDGES) + 1):
        band_scores = scores[bands == band]
        level = min(1.0, np.ceil((len(band_scores) + 1) * (1 - alpha)) / len(band_scores))
        conformal_margins.append(float(np.quantile(band_scores, level, method="higher")))

    fails_soon = lambda df: (df["TRUE_RUL"] <= config.FAILURE_HORIZON_CYCLES).astype(int)
    classifier = HistGradientBoostingClassifier(**REGRESSOR_PARAMS, random_state=SEED).fit(fit[cols], fails_soon(fit))
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
        classifier.predict_proba(cal[cols])[:, 1], fails_soon(cal)
    )

    return {
        "feature_columns": cols,
        "point": point,
        "lower": lower,
        "upper": upper,
        "conformal_band_edges": CONFORMAL_BAND_EDGES,
        "conformal_margins": conformal_margins,
        "classifier": classifier,
        "calibrator": calibrator,
    }


def _band_labels(edges: list[int]) -> list[str]:
    bounds = [0] + list(edges) + [config.RUL_CAP]
    return [f"{lo}-{hi}" for lo, hi in zip(bounds, bounds[1:])]


def nasa_score(true_rul, pred_rul) -> float:
    """PHM08 asymmetric score: late predictions (pred > true) are penalized harder,
    since an engine failing before its predicted RUL is the costly mistake."""
    d = np.asarray(pred_rul) - np.asarray(true_rul)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def _regression_metrics(true_rul, pred_rul) -> dict:
    err = np.asarray(pred_rul) - np.asarray(true_rul)
    return {
        "rmse": round(float(np.sqrt(np.mean(err**2))), 2),
        "mae": round(float(np.mean(np.abs(err))), 2),
        "nasa_score": round(nasa_score(true_rul, pred_rul), 1),
        "n_engines": int(len(err)),
    }


def evaluate(bundle: dict, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    cols = bundle["feature_columns"]
    preds = predict_with_bundle(bundle, test, id_col="MACHINE_ID")
    test = test.merge(preds, on="MACHINE_ID")
    truth_capped = test["TRUE_RUL"].clip(upper=config.RUL_CAP)

    per_fleet = {
        fleet: _regression_metrics(truth_capped[g.index], g["RUL_PREDICTION"])
        for fleet, g in test.groupby("FLEET")
    }

    covered = (truth_capped >= test["RUL_LOWER"]) & (truth_capped <= test["RUL_UPPER"])
    width = test["RUL_UPPER"] - test["RUL_LOWER"]
    labels = _band_labels(bundle["conformal_band_edges"])

    def coverage_by(band: np.ndarray) -> dict:
        return {
            label: {
                "n_engines": int((band == i).sum()),
                "coverage": round(float(covered[band == i].mean()), 3),
                "mean_width_cycles": round(float(width[band == i].mean()), 1),
            }
            for i, label in enumerate(labels)
        }

    # Conformal calibration guarantees coverage per band of the *prediction* (what an
    # operator sees); per band of the unknown truth is reported too, as the harder test.
    by_predicted_band = coverage_by(np.digitize(test["RUL_PREDICTION"], bundle["conformal_band_edges"]))
    by_true_band = coverage_by(np.digitize(truth_capped, bundle["conformal_band_edges"]))

    fails_soon = (test["TRUE_RUL"] <= config.FAILURE_HORIZON_CYCLES).astype(int)
    flagged = (test["FAILURE_PROBABILITY"] >= 0.5).astype(int)

    # Cross-validation by engine on the training fleet, for a second, independent estimate.
    cv_rmse = []
    for tr, va in GroupKFold(n_splits=5).split(train, groups=train["MACHINE_ID"]):
        model = _regressor().fit(train.iloc[tr][cols], train.iloc[tr]["RUL"])
        cv_rmse.append(_regression_metrics(train.iloc[va]["RUL"], model.predict(train.iloc[va][cols]))["rmse"])

    # What the previous version of this project did: split FD001 rows at random.
    fd001 = train[train["FLEET"] == "FD001"]
    leaky_tr, leaky_va = train_test_split(fd001, test_size=0.2, random_state=SEED)
    leaky_model = _regressor().fit(leaky_tr[cols], leaky_tr["RUL"])
    leaky_rmse = _regression_metrics(leaky_va["RUL"], leaky_model.predict(leaky_va[cols]))["rmse"]

    constant = np.full(len(test), train["RUL"].mean())

    return {
        "evaluated_on": "NASA C-MAPSS official test set (RUL_FD00X.txt), last observed cycle of each engine",
        "truth": f"true RUL capped at {config.RUL_CAP}, matching the training label",
        "rul": {
            "overall": _regression_metrics(truth_capped, test["RUL_PREDICTION"]),
            "per_fleet": per_fleet,
            "overall_vs_uncapped_truth": _regression_metrics(test["TRUE_RUL"], test["RUL_PREDICTION"]),
        },
        "rul_interval": {
            "target_coverage": config.PREDICTION_INTERVAL,
            "empirical_coverage": round(float(covered.mean()), 3),
            "mean_width_cycles": round(float((test["RUL_UPPER"] - test["RUL_LOWER"]).mean()), 1),
            "conformal_margins_by_band": dict(zip(
                _band_labels(bundle["conformal_band_edges"]), [round(m, 2) for m in bundle["conformal_margins"]]
            )),
            "by_predicted_rul_band": by_predicted_band,
            "by_true_rul_band": by_true_band,
        },
        "failure_probability": {
            "horizon_cycles": config.FAILURE_HORIZON_CYCLES,
            "positives": int(fails_soon.sum()),
            "brier_score": round(float(brier_score_loss(fails_soon, test["FAILURE_PROBABILITY"])), 4),
            "roc_auc": round(float(roc_auc_score(fails_soon, test["FAILURE_PROBABILITY"])), 4),
            "precision_at_0.5": round(float(precision_score(fails_soon, flagged, zero_division=0)), 3),
            "recall_at_0.5": round(float(recall_score(fails_soon, flagged, zero_division=0)), 3),
        },
        "baselines": {
            "constant_mean_rul_rmse": _regression_metrics(truth_capped, constant)["rmse"],
            "groupkfold_cv_rmse_by_engine": {"folds": cv_rmse, "mean": round(float(np.mean(cv_rmse)), 2)},
            "leaky_random_row_split_fd001_rmse": leaky_rmse,
        },
    }


def train_and_save() -> dict:
    train = build_training_set()
    test = build_evaluation_set()
    print(f"training rows: {len(train):,} from {train['MACHINE_ID'].nunique()} failed engines; "
          f"evaluating on {len(test)} in-service engines")

    bundle = fit_models(train)
    metrics = evaluate(bundle, train, test)
    bundle["trained_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    bundle["metrics"] = metrics

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"saved -> {MODEL_PATH}\nsaved -> {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    train_and_save()
