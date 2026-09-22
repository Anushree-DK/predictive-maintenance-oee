import numpy as np

import config
from src.ml.predict import _confidence_from_bag, _failure_probability, _risk_class


def test_risk_class_boundaries_match_config_thresholds():
    assert _risk_class(0) == "CRITICAL"
    assert _risk_class(config.RISK_THRESHOLDS["CRITICAL"]) == "CRITICAL"
    assert _risk_class(config.RISK_THRESHOLDS["CRITICAL"] + 1) == "HIGH"
    assert _risk_class(config.RISK_THRESHOLDS["HIGH"]) == "HIGH"
    assert _risk_class(config.RISK_THRESHOLDS["HIGH"] + 1) == "MEDIUM"
    assert _risk_class(config.RISK_THRESHOLDS["MEDIUM"]) == "MEDIUM"
    assert _risk_class(config.RISK_THRESHOLDS["MEDIUM"] + 1) == "LOW"
    assert _risk_class(10_000) == "LOW"


def test_failure_probability_is_high_near_zero_rul():
    assert _failure_probability(0, horizon=30) > 0.95


def test_failure_probability_is_low_far_from_horizon():
    assert _failure_probability(300, horizon=30) < 0.01


def test_failure_probability_is_monotonically_decreasing_in_rul():
    ruls = [0, 10, 30, 60, 120]
    probs = [_failure_probability(r, horizon=30) for r in ruls]
    assert probs == sorted(probs, reverse=True)


def test_confidence_from_bag_perfect_agreement_is_high_confidence():
    identical_preds = np.array([50.0, 50.0, 50.0, 50.0])
    assert _confidence_from_bag(identical_preds) > 0.99


def test_confidence_from_bag_high_spread_is_low_confidence():
    spread_preds = np.array([10.0, 90.0, 20.0, 80.0])
    assert _confidence_from_bag(spread_preds) < 0.5


def test_confidence_from_bag_is_bounded_between_zero_and_one():
    for preds in [np.array([1.0, 1000.0]), np.array([0.0, 0.0]), np.array([-5.0, 5.0])]:
        c = _confidence_from_bag(preds)
        assert 0.0 <= c <= 1.0
