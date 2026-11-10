"""Low-level cross-fitting and score functions for the DML estimators.

These functions follow the estimators used in Bearth & Lechner (2024),
"Causal Machine Learning for Moderation Effects" (arXiv:2401.08290).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone


def make_folds(n: int, n_folds: int, rng: np.random.RandomState) -> list[np.ndarray]:
    """Randomly split the indices ``0, ..., n-1`` into ``n_folds`` folds."""
    idx = rng.choice(np.arange(n), size=n, replace=False)
    return np.array_split(idx, n_folds)


def train_test(folds: list[np.ndarray], k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (train, test) indices where fold ``k`` is the test fold."""
    test = folds[k]
    train = np.concatenate(folds[:k] + folds[k + 1:])
    return train, test


def fit_predict_regression(model, x_train, y_train, x_test, what: str) -> np.ndarray:
    """Fit a fresh copy of a regression model and predict on the test data."""
    if x_train.shape[0] == 0:
        raise ValueError(
            f"Cannot fit the {what}: a cross-fitting training fold contains no "
            "observations for this group. Use fewer folds or more data."
        )
    model = clone(model)
    model.fit(x_train, np.ravel(y_train))
    return model.predict(x_test)


def fit_predict_proba(model, x_train, w_train, x_test, what: str) -> np.ndarray:
    """Fit a fresh copy of a classifier and return P(W=1 | x) on the test data."""
    w_train = np.ravel(w_train)
    if np.unique(w_train).size < 2:
        raise ValueError(
            f"Cannot fit the {what}: a cross-fitting training fold contains only "
            "one group. Use fewer folds or more data."
        )
    model = clone(model)
    model.fit(x_train, w_train)
    classes = list(model.classes_)
    return model.predict_proba(x_test)[:, classes.index(1)]


def aipw_weights(
    w: np.ndarray, p: np.ndarray, normalize: bool, clip: float
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-probability weights for the groups ``w == 1`` and ``w == 0``.

    Propensity scores are clipped to ``[clip, 1 - clip]``. If ``normalize`` is
    true, each weight vector is divided by its mean (Hajek normalisation).
    """
    p = np.clip(p, clip, 1 - clip)
    w1 = w / p
    w0 = (1 - w) / (1 - p)
    if normalize:
        w1 = w1 / np.mean(w1)
        w0 = w0 / np.mean(w0)
    return w1, w0


def aipw_scores(
    y: np.ndarray,
    w: np.ndarray,
    mu1: np.ndarray,
    mu0: np.ndarray,
    p: np.ndarray,
    normalize: bool,
    clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Doubly robust scores for E[mu1] and E[mu0].

    Returns ``(psi1, psi0)``; ``psi1 - psi0`` is the usual AIPW / DML score
    for the average effect of ``w``.
    """
    w1, w0 = aipw_weights(w, p, normalize, clip)
    psi1 = mu1 + w1 * (y - mu1)
    psi0 = mu0 + w0 * (y - mu0)
    return psi1, psi0


def aipw_difference(y, w, mu1, mu0, p, normalize, clip) -> np.ndarray:
    """Doubly robust score for E[mu1 - mu0] (the DML treatment effect score).

    Equal to ``psi1 - psi0`` from :func:`aipw_scores`, computed in the same
    floating point order as the original replication code.
    """
    w1, w0 = aipw_weights(w, p, normalize, clip)
    return mu1 - mu0 + w1 * (y - mu1) - w0 * (y - mu0)


def mean_and_variance(score: np.ndarray) -> tuple[float, float]:
    """Sample mean of a score and the variance of that mean."""
    return float(np.mean(score)), float(np.var(score) / score.shape[0])
