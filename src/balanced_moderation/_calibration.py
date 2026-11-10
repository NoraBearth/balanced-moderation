"""Propensity score calibration for DML.

Implements the cross-fitted calibration procedure of Ballinari & Bearth
(2024), "Improving the Finite Sample Estimation of Average Treatment Effects
using Double/Debiased Machine Learning with Propensity Score Calibration",
arXiv:2409.04874 (Algorithm 1): the propensity model is trained on the
training folds; its predictions on the held-out fold are calibrated with an
additional split of that fold, so a calibrator is never evaluated on the
observations it was fitted on.

The calibration maps follow the replication code of that paper
(github.com/dballinari/Improving-the-Finite-Sample-Performance-of-DML-with-Propensity-Score-Calibration).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

CALIBRATION_METHODS = ("platt", "beta", "venn_abers", "isotonic")

# Scores are kept away from exactly 0 and 1 before taking logs (beta scaling).
_EPS = 1e-12


def _fit_logistic(features: np.ndarray, w: np.ndarray, nonneg_slopes: bool) -> np.ndarray:
    """Maximum likelihood logistic regression with intercept.

    Slopes are constrained to be non-negative if ``nonneg_slopes`` (so that
    the calibration map is monotone). Returns [intercept, slopes...].
    """
    x = np.column_stack((np.ones(features.shape[0]), features))

    def negloglik(coef):
        g = x @ coef
        return -np.sum(w * g - np.logaddexp(0, g))

    def grad(coef):
        return -x.T @ (w - expit(x @ coef))

    bounds = [(None, None)] + [(0, None) if nonneg_slopes else (None, None)] * features.shape[1]
    return minimize(negloglik, np.zeros(x.shape[1]), jac=grad,
                    method="L-BFGS-B", bounds=bounds).x


def _platt(p_fit, w_fit, p_new):
    """Platt (1999) scaling: logistic regression of D on the score."""
    coef = _fit_logistic(p_fit.reshape(-1, 1), w_fit, nonneg_slopes=True)
    return expit(coef[0] + coef[1] * p_new)


def _beta_features(p):
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.column_stack((np.log(p), -np.log(1 - p)))


def _beta(p_fit, w_fit, p_new):
    """Beta calibration of Kull, Silva Filho & Flach (2017)."""
    coef = _fit_logistic(_beta_features(p_fit), w_fit, nonneg_slopes=True)
    return expit(coef[0] + _beta_features(p_new) @ coef[1:])


def _venn_abers(p_fit, w_fit, p_new):
    """Inductive Venn-Abers calibration (Vovk, Petej & Fedorova, 2015)."""
    from venn_abers import VennAbers

    va = VennAbers()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        va.fit(np.column_stack((1 - p_fit, p_fit)), w_fit.astype(int))
        p_prime, _ = va.predict_proba(np.column_stack((1 - p_new, p_new)))
    return p_prime[:, 1]


def _isotonic(p_fit, w_fit, p_new):
    """Isotonic regression, bounded to [0.001, 0.999]."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    iso.fit(p_fit, w_fit)
    return iso.predict(p_new)


_CALIBRATORS = {
    "platt": _platt,
    "beta": _beta,
    "venn_abers": _venn_abers,
    "isotonic": _isotonic,
}


def calibrate_crossfit(
    p: np.ndarray, w: np.ndarray, method: str, n_splits: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Calibrate held-out propensity scores ``p`` of the binary variable ``w``.

    The observations are split into ``n_splits`` parts; scores in each part are
    calibrated with a map fitted on the other parts.
    """
    calibrate = _CALIBRATORS[method]
    n = p.shape[0]
    parts = np.array_split(rng.choice(np.arange(n), size=n, replace=False), n_splits)
    out = np.full(n, np.nan)
    for j in range(n_splits):
        new = parts[j]
        fit = np.concatenate(parts[:j] + parts[j + 1:])
        if np.unique(w[fit]).size < 2:
            raise ValueError(
                "Cannot calibrate the propensity score: a calibration split "
                "contains only one group. Use fewer folds, more data, or "
                "calibration=None.")
        out[new] = calibrate(p[fit], w[fit], p[new])
    return out
