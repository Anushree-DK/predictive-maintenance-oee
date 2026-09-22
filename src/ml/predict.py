"""Loads the trained RUL model and turns current machine features into the four
outputs the diagram calls for: RUL, failure probability, risk class, confidence.
"""

import numpy as np
import pandas as pd
import joblib

import config


def _risk_class(rul: float) -> str:
    if rul <= config.RISK_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    if rul <= config.RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    if rul <= config.RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def _failure_probability(rul: float, horizon: int = 30) -> float:
    """P(failure within `horizon` cycles), as a smooth function of predicted RUL."""
    return float(1 / (1 + np.exp((rul - horizon) / (horizon / 4))))


def _bag_predictions(models, X_row: pd.DataFrame) -> np.ndarray:
    return np.array([m.predict(X_row)[0] for m in models])


def _confidence_from_bag(bag_preds: np.ndarray) -> float:
    """Agreement across the bagged models: tight spread -> high confidence."""
    spread = bag_preds.std()
    mean = max(bag_preds.mean(), 1e-6)
    coeff_of_variation = spread / mean
    return float(np.clip(1 - coeff_of_variation, 0.0, 1.0))


def load_model():
    path = config.MODELS_DIR / "rul_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run: python -m src.ml.train")
    return joblib.load(path)


def predict_for_features(feature_table: pd.DataFrame, id_col: str = "MACHINE_ID") -> pd.DataFrame:
    bundle = load_model()
    models, feature_columns = bundle["models"], bundle["feature_columns"]

    missing = [c for c in feature_columns if c not in feature_table.columns]
    if missing:
        raise ValueError(f"feature_table is missing columns the model was trained on: {missing}")

    results = []
    for _, row in feature_table.iterrows():
        X_row = row[feature_columns].to_frame().T
        bag_preds = _bag_predictions(models, X_row)
        rul = float(bag_preds.mean())
        results.append(
            {
                id_col: row[id_col],
                "RUL_PREDICTION": round(rul, 1),
                "FAILURE_PROBABILITY": round(_failure_probability(rul), 3),
                "RISK_CLASS": _risk_class(rul),
                "CONFIDENCE_SCORE": round(_confidence_from_bag(bag_preds), 3),
            }
        )
    return pd.DataFrame(results)
