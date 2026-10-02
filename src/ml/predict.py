"""Turns features into RUL, prediction interval, failure probability and risk class."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

import config


def risk_class(rul_lower: float) -> str:
    """Risk from the interval's lower bound: how soon the engine could plausibly fail."""
    if rul_lower <= config.RISK_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    if rul_lower <= config.RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    if rul_lower <= config.RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def conformal_interval(lower_raw, upper_raw, point, band_edges, margins) -> tuple[np.ndarray, np.ndarray]:
    """Adjust the quantile bounds by the conformal margin of the point prediction's
    band, keep them inside [0, RUL_CAP], and make sure they contain the point."""
    point = np.asarray(point)
    margin = np.asarray(margins)[np.digitize(point, band_edges)]
    lower = np.clip(np.minimum(np.asarray(lower_raw) - margin, point), 0, config.RUL_CAP)
    upper = np.clip(np.maximum(np.asarray(upper_raw) + margin, point), 0, config.RUL_CAP)
    return lower, upper


def load_model() -> dict:
    path = config.MODELS_DIR / "rul_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run: python -m src.ml.train")
    return joblib.load(path)


def predict_with_bundle(bundle: dict, feature_table: pd.DataFrame, id_col: str | None = "MACHINE_ID") -> pd.DataFrame:
    """One output row per feature row, in the same order. id_col=None omits the ID
    column (the Model Registry joins outputs back to inputs itself)."""
    cols = bundle["feature_columns"]
    missing = [c for c in cols if c not in feature_table.columns]
    if missing:
        raise ValueError(f"feature_table is missing columns the model was trained on: {missing}")

    X = feature_table[cols]
    point = np.clip(bundle["point"].predict(X), 0, config.RUL_CAP)
    lower, upper = conformal_interval(
        bundle["lower"].predict(X), bundle["upper"].predict(X), point,
        bundle["conformal_band_edges"], bundle["conformal_margins"],
    )
    prob = bundle["calibrator"].predict(bundle["classifier"].predict_proba(X)[:, 1])

    result = pd.DataFrame({
        "RUL_PREDICTION": np.round(point, 1),
        "RUL_LOWER": np.round(lower, 1),
        "RUL_UPPER": np.round(upper, 1),
        "FAILURE_PROBABILITY": np.round(prob, 3),
        "RISK_CLASS": [risk_class(lo) for lo in lower],
    })
    if id_col is not None:
        result.insert(0, id_col, feature_table[id_col].values)
    return result


def predict_for_features(feature_table: pd.DataFrame, id_col: str = "MACHINE_ID") -> pd.DataFrame:
    return predict_with_bundle(load_model(), feature_table, id_col=id_col)
