"""Double/debiased machine learning estimators for GATE and BGATE differences."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ._calibration import calibrate_crossfit
from ._crossfit import (
    aipw_difference,
    aipw_scores,
    fit_predict_proba,
    fit_predict_regression,
    make_folds,
    train_test,
)


@dataclass
class Nuisances:
    outcome_model: object
    propensity_model: object
    effect_model: object
    moderator_model: object
    calibration: str | None = None
    calibration_splits: int = 2

    def calibrate(self, p, w, rng):
        """Cross-fitted calibration of held-out propensity scores (if enabled)."""
        if self.calibration is None:
            return p
        return calibrate_crossfit(p, w, self.calibration, self.calibration_splits, rng)


def first_step_scores(
    y, d, x, nuis: Nuisances, n_folds, rng, normalize, clip
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Cross-fitted DML score of the treatment effect for every observation.

    The nuisance functions E[Y | D=d, X] and P(D=1 | X) are estimated with
    ``n_folds``-fold cross-fitting. Returns the scores and the folds used.
    """
    n = y.shape[0]
    mu1 = np.full(n, np.nan)
    mu0 = np.full(n, np.nan)
    p = np.full(n, np.nan)
    folds = make_folds(n, n_folds, rng)

    for k in range(n_folds):
        tr, te = train_test(folds, k)
        d_tr = d[tr]
        mu1[te] = fit_predict_regression(
            nuis.outcome_model, x[tr][d_tr == 1], y[tr][d_tr == 1], x[te],
            "outcome model for the treated")
        mu0[te] = fit_predict_regression(
            nuis.outcome_model, x[tr][d_tr == 0], y[tr][d_tr == 0], x[te],
            "outcome model for the non-treated")
        p[te] = nuis.calibrate(
            fit_predict_proba(nuis.propensity_model, x[tr], d_tr, x[te], "propensity model"),
            d[te], rng)

    check_overlap(p, "treatment propensity P(D=1 | X)")
    return aipw_difference(y, d, mu1, mu0, p, normalize, clip), folds


def gate_scores(y, d, z, x, nuis, n_folds, rng, clip):
    """Scores for the GATEs in groups Z=1 and Z=0 (no balancing).

    The ATE is estimated separately within each group, first Z=1, then Z=0.
    """
    tau1, _ = first_step_scores(
        y[z == 1], d[z == 1], x[z == 1], nuis, n_folds, rng, normalize=True, clip=clip)
    tau0, _ = first_step_scores(
        y[z == 0], d[z == 0], x[z == 0], nuis, n_folds, rng, normalize=True, clip=clip)
    return tau1, tau0


def bgate_scores(y, d, z, x, w, nuis, n_folds, n_folds_second, rng, normalize, clip):
    """Scores for BGATE(Z=1) and BGATE(Z=0), balanced over the variables ``w``.

    Step 1: cross-fitted DML scores ``tau_i`` of the treatment effect, using
    covariates (X, Z).
    Step 2: within each step-1 test fold, a second cross-fitting splits the
    data again; E[tau | W, Z=z] and P(Z=1 | W) are estimated and combined into
    doubly robust scores for E_W[ E[tau | W, Z=z] ], z = 0, 1.
    """
    n = y.shape[0]
    xz = np.column_stack((x, z))
    tau, folds = first_step_scores(y, d, xz, nuis, n_folds, rng, normalize=True, clip=clip)

    m1 = np.full(n, np.nan)
    m0 = np.full(n, np.nan)
    pz = np.full(n, np.nan)

    for k in range(n_folds):
        idx_first = folds[k]
        z_f, w_f, tau_f = z[idx_first], w[idx_first], tau[idx_first]
        folds2 = make_folds(idx_first.shape[0], n_folds_second, rng)
        for j in range(n_folds_second):
            tr, te = train_test(folds2, j)
            z_tr = z_f[tr]
            out = idx_first[te]
            m1[out] = fit_predict_regression(
                nuis.effect_model, w_f[tr][z_tr == 1], tau_f[tr][z_tr == 1], w_f[te],
                "effect model for group Z=1")
            m0[out] = fit_predict_regression(
                nuis.effect_model, w_f[tr][z_tr == 0], tau_f[tr][z_tr == 0], w_f[te],
                "effect model for group Z=0")
            pz[out] = nuis.calibrate(
                fit_predict_proba(nuis.moderator_model, w_f[tr], z_tr, w_f[te],
                                  "moderator model"),
                z_f[te], rng)

    check_overlap(pz, "moderator propensity P(Z=1 | W)")
    return aipw_scores(tau, z, m1, m0, pz, normalize, clip)


OVERLAP_BOUND = 0.01


def check_overlap(p: np.ndarray, what: str) -> None:
    """Warn if estimated probabilities are close to 0 or 1 (weak overlap)."""
    extreme = (p < OVERLAP_BOUND) | (p > 1 - OVERLAP_BOUND)
    if extreme.any():
        warnings.warn(
            f"{extreme.sum()} of {p.size} estimated values of the {what} are "
            f"below {OVERLAP_BOUND} or above {1 - OVERLAP_BOUND}. The inverse "
            "probability weights become very large and the estimate may be "
            "unreliable. Consider a smoother classifier (e.g. a random forest "
            "with larger min_samples_leaf, or logistic regression), fewer "
            "balancing variables, or a larger `propensity_clip`.",
            UserWarning,
            stacklevel=4,
        )
