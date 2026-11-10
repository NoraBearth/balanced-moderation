import numpy as np
import pytest

from balanced_moderation import _calibration
from balanced_moderation._calibration import CALIBRATION_METHODS, calibrate_crossfit


def overconfident_scores(n=600, seed=0):
    """True propensity in [0.1, 0.9]; predictions pushed to (exactly) 0 and 1."""
    rng = np.random.RandomState(seed)
    true_p = rng.uniform(0.1, 0.9, n)
    w = rng.binomial(1, true_p).astype(float)
    p_hat = np.clip((true_p - 0.5) * 3 + 0.5, 0, 1)  # exact 0s and 1s
    return p_hat, w, true_p


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_calibrated_scores_are_valid_probabilities(method):
    p_hat, w, _ = overconfident_scores()
    assert (p_hat == 0).any() and (p_hat == 1).any()
    p_cal = calibrate_crossfit(p_hat, w, method, 2, np.random.RandomState(0))
    assert p_cal.shape == p_hat.shape
    assert np.all(np.isfinite(p_cal))
    assert np.all((p_cal > 0) & (p_cal < 1))


@pytest.mark.parametrize("method", ["venn_abers", "platt"])
def test_calibration_removes_exact_zeros_and_ones(method):
    p_hat, w, _ = overconfident_scores()
    p_cal = calibrate_crossfit(p_hat, w, method, 2, np.random.RandomState(0))
    assert p_cal.min() > 0.01 and p_cal.max() < 0.99


@pytest.mark.parametrize("method", ["venn_abers", "platt", "beta"])
def test_calibration_improves_brier_score(method):
    p_hat, w, true_p = overconfident_scores(n=4000)
    p_cal = calibrate_crossfit(p_hat, w, method, 2, np.random.RandomState(0))
    assert np.mean((p_cal - true_p) ** 2) < np.mean((p_hat - true_p) ** 2)


@pytest.mark.parametrize("n_splits", [2, 3, 5])
def test_calibrator_never_sees_the_observations_it_calibrates(monkeypatch, n_splits):
    calls = []

    def spy(p_fit, w_fit, p_new):
        calls.append((set(p_fit.tolist()), set(p_new.tolist())))
        return np.full(p_new.shape, 0.5)

    monkeypatch.setitem(_calibration._CALIBRATORS, "platt", spy)
    n = 101
    p = np.linspace(0.01, 0.99, n)  # unique values identify observations
    w = np.tile([0.0, 1.0], n)[:n]
    out = calibrate_crossfit(p, w, "platt", n_splits, np.random.RandomState(1))
    assert len(calls) == n_splits
    calibrated = set()
    for fit, new in calls:
        assert fit.isdisjoint(new)
        assert len(fit) + len(new) == n
        calibrated |= new
    assert calibrated == set(p.tolist())  # every observation calibrated once
    assert np.all(out == 0.5)


def test_calibration_split_with_one_group_raises():
    p = np.linspace(0.1, 0.9, 20)
    w = np.zeros(20)
    with pytest.raises(ValueError, match="only one group"):
        calibrate_crossfit(p, w, "platt", 2, np.random.RandomState(0))
