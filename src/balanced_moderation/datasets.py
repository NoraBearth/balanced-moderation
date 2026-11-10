"""Simulated data from the simulation study of Bearth & Lechner (2024)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.utils import check_random_state

_BETA_1 = [0.7, 0.1, 0.7, 0.4, 0.2]
_BETA_0 = [0.2, 0.3, 0.6, 0.3, 0.5]
_DESIGNS = ("linear", "nonlinear", "z_affects_x")


def _simulate(n, p, design, rng):
    if design not in _DESIGNS:
        raise ValueError(f"`design` must be one of {_DESIGNS}, got {design!r}.")
    if p < 7:
        raise ValueError("`p` (number of covariates) must be at least 7.")

    X = np.full((n, p), np.nan)
    X[:, 0] = rng.uniform(0, 1, n)
    X[:, 1] = rng.uniform(0, 1, n)
    X[:, 2:p] = rng.normal(0.5, np.sqrt(1 / 12), (n, p - 2))

    pz = 0.1 + 0.8 * beta.cdf(X[:, 0] * X[:, 1], 2, 4)
    Z = rng.binomial(1, p=pz, size=n)

    if design == "z_affects_x":
        X[:, 5] = rng.binomial(1, p=1 / (1 + np.exp(-2 * Z)), size=n)

    p_d = 0.2 + 0.6 * beta.cdf((X[:, 0] + X[:, 1] + X[:, 2] + X[:, 5] + Z) / 5, 2, 4)
    D = rng.binomial(1, p=p_d, size=n)

    mu0 = (np.sin(np.pi * X[:, 0] * X[:, 1]) + (X[:, 2] - 0.5) ** 2
           + 0.1 * X[:, 3] + 0.3 * X[:, 5])
    tau1 = _tau(X, 1, design)
    tau0 = _tau(X, 0, design)
    eps = [rng.normal(0, 1, n) for _ in range(4)]  # order: 01, 00, 11, 10

    Y = ((mu0 + tau1 + eps[2]) * D * Z + (mu0 + eps[0]) * (1 - D) * Z
         + (mu0 + tau0 + eps[3]) * D * (1 - Z) + (mu0 + eps[1]) * (1 - D) * (1 - Z))
    return Y, D, Z, X, tau1, tau0


def _tau(X, z, design):
    """Conditional treatment effect at covariates X in group Z=z."""
    b = _BETA_1 if z == 1 else _BETA_0
    if design == "linear":
        return b[0] * X[:, 0] + b[1] * X[:, 1] + b[2] * X[:, 2] + b[3] * X[:, 5] + b[4] * z
    power = 4 if z == 1 else 2
    return (np.sin(7 * (b[0] * X[:, 0])) + np.sin(20 * b[1] * X[:, 1])
            + b[2] * X[:, 2] ** power + b[3] * X[:, 5] + b[4] * z)


def make_moderation_data(
    n: int = 1000, p: int = 10, design: str = "z_affects_x", random_state=None
) -> pd.DataFrame:
    """Simulate data with a binary treatment ``d`` and binary moderator ``z``.

    These are the data generating processes of the simulation study in
    Bearth & Lechner (2024).

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of covariates (at least 7).
    design : {"linear", "nonlinear", "z_affects_x"}
        ``"linear"``: effect heterogeneity is linear in the covariates.
        ``"nonlinear"``: effect heterogeneity is nonlinear.
        ``"z_affects_x"``: as ``"nonlinear"``, but the moderator also shifts
        the distribution of covariate ``x5``, so GATE and BGATE differ.
    random_state : int, numpy RandomState or None
        Seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Columns ``y`` (outcome), ``d`` (treatment), ``z`` (moderator) and
        ``x0, ..., x{p-1}`` (covariates).
    """
    rng = check_random_state(random_state)
    Y, D, Z, X, _, _ = _simulate(n, p, design, rng)
    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(p)])
    df.insert(0, "z", Z)
    df.insert(0, "d", D)
    df.insert(0, "y", Y)
    return df


def true_moderation_effects(
    p: int = 10, design: str = "z_affects_x", n_draws: int = 1_000_000,
    random_state=0,
) -> dict[str, float]:
    """Approximate the true GATE and BGATE differences by Monte Carlo.

    Returns a dict with ``"GATE"`` (no balancing) and ``"BGATE_all"`` (balanced
    over all covariates ``x0, ..., x{p-1}``). For ``design="z_affects_x"``,
    where ``x5`` is binary, it also returns ``"BGATE_x5"`` (balanced over
    ``x5`` only).
    """
    rng = check_random_state(random_state)
    _, _, Z, X, tau1, tau0 = _simulate(n_draws, p, design, rng)
    truth = {
        "GATE": float(tau1[Z == 1].mean() - tau0[Z == 0].mean()),
        "BGATE_all": float(tau1.mean() - tau0.mean()),
    }
    if design == "z_affects_x":
        # BGATE(z) = sum_v P(x5 = v) * E[tau_z | Z = z, x5 = v]
        x5 = X[:, 5]
        bgate = {z: 0.0 for z in (0, 1)}
        for v in (0.0, 1.0):
            share = np.mean(x5 == v)
            bgate[1] += share * tau1[(Z == 1) & (x5 == v)].mean()
            bgate[0] += share * tau0[(Z == 0) & (x5 == v)].mean()
        truth["BGATE_x5"] = float(bgate[1] - bgate[0])
    return truth
