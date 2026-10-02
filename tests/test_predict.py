import numpy as np

import config
from src.ml.predict import conformal_interval, risk_class
from src.ml.train import nasa_score


def test_risk_class_boundaries_match_config_thresholds():
    assert risk_class(0) == "CRITICAL"
    assert risk_class(config.RISK_THRESHOLDS["CRITICAL"]) == "CRITICAL"
    assert risk_class(config.RISK_THRESHOLDS["CRITICAL"] + 1) == "HIGH"
    assert risk_class(config.RISK_THRESHOLDS["HIGH"]) == "HIGH"
    assert risk_class(config.RISK_THRESHOLDS["HIGH"] + 1) == "MEDIUM"
    assert risk_class(config.RISK_THRESHOLDS["MEDIUM"]) == "MEDIUM"
    assert risk_class(config.RISK_THRESHOLDS["MEDIUM"] + 1) == "LOW"
    assert risk_class(10_000) == "LOW"


def test_conformal_interval_applies_the_margin_of_the_points_band():
    lower, upper = conformal_interval(
        lower_raw=[10, 70], upper_raw=[20, 80], point=[15, 75], band_edges=[30, 60], margins=[1.0, 2.0, 5.0]
    )
    assert list(lower) == [9.0, 65.0]
    assert list(upper) == [21.0, 85.0]


def test_conformal_interval_negative_margin_narrows_but_keeps_the_point():
    lower, upper = conformal_interval(lower_raw=[10], upper_raw=[20], point=[19], band_edges=[30], margins=[-3.0, 0.0])
    assert (lower[0], upper[0]) == (13.0, 19.0)


def test_conformal_interval_is_clipped_to_valid_rul_range():
    lower, upper = conformal_interval(lower_raw=[-5], upper_raw=[200], point=[50], band_edges=[30], margins=[0.0, 10.0])
    assert (lower[0], upper[0]) == (0.0, config.RUL_CAP)


def test_nasa_score_penalizes_late_predictions_more_than_early():
    early = nasa_score([50], [40])  # predicted failure 10 cycles too soon
    late = nasa_score([50], [60])   # predicted failure 10 cycles too late
    assert late > early > 0
    assert nasa_score([50], [50]) == 0.0
    assert np.isclose(late, np.exp(1) - 1)
