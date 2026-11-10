"""User-facing estimator for GATE and BGATE moderation effects."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils import check_random_state

from ._calibration import CALIBRATION_METHODS
from ._crossfit import mean_and_variance
from ._dml import Nuisances, bgate_scores, gate_scores
from .results import ModerationResults


class BGATE:
    """Moderation effects via (balanced) group average treatment effects.

    Estimates how the effect of a binary treatment D on an outcome Y differs
    between two groups defined by a binary moderator Z, using double/debiased
    machine learning (Bearth & Lechner, 2024).

    * Without balancing variables, it estimates the group average treatment
      effects GATE(Z=1) and GATE(Z=0) and their difference.
    * With balancing variables W, it estimates the balanced GATEs
      BGATE(Z=z) = E_W[ E[Y(1) - Y(0) | Z=z, W] ], i.e. the effect in group z
      if W had the same (population) distribution in both groups. The
      difference BGATE(Z=1) - BGATE(Z=0) is the moderation effect of Z
      holding the distribution of W fixed.

    Parameters
    ----------
    outcome_model : scikit-learn regressor, optional
        Model for E[Y | D=d, X]. Fitted separately for treated and
        non-treated. Default: ``RandomForestRegressor``.
    propensity_model : scikit-learn classifier, optional
        Model for P(D=1 | X). Must implement ``predict_proba``.
        Default: ``RandomForestClassifier``.
    effect_model : scikit-learn regressor, optional
        BGATE only: model for E[tau | W, Z=z], fitted separately for Z=1
        and Z=0. Default: same as ``outcome_model``.
    moderator_model : scikit-learn classifier, optional
        BGATE only: model for P(Z=1 | W). Default: same as
        ``propensity_model``.
    n_folds : int, default 2
        Number of cross-fitting folds (2, as in the simulation study of
        Bearth & Lechner, 2024).
    n_folds_second : int, optional
        BGATE only: number of folds of the second cross-fitting step, which
        is carried out within each first-step fold. Default: ``n_folds``.
        Because the second step only uses the observations of one first-step
        fold, more folds leave less data for the second-step models; with
        many balancing variables this requires a large sample.
    calibration : {"venn_abers", "platt", "beta", "isotonic"} or None, default "venn_abers"
        Calibrate the estimated propensity scores P(D=1 | X) and P(Z=1 | W)
        with the cross-fitted procedure of Ballinari & Bearth (2024). Within
        each cross-fitting fold the held-out predictions are split again; the
        calibration map is fitted on one part and applied to the other.
        Calibration corrects over-confident classifiers (e.g. random forests
        predicting probabilities of exactly 0 or 1). ``None`` disables it.
        Venn-Abers (the default) keeps calibrated scores away from 0 and 1
        even when the classifier predicts exactly 0 or 1. ``"beta"`` works on
        log-probabilities and can produce extreme scores in that case.
        Ballinari & Bearth recommend choosing the method by Brier score and
        advise against ``"isotonic"``.
    calibration_splits : int, default 2
        Number of splits of each held-out fold used for calibration.
    normalize_weights : bool, default True
        BGATE only: normalise the inverse-probability weights of the second
        (balancing) step to have mean one. First-step weights are always
        normalised.
    propensity_clip : float, default 1e-10
        Propensity scores are clipped to [clip, 1 - clip].
    random_state : int, numpy RandomState or None
        Seed for the random fold assignment. For fully reproducible results
        also set ``random_state`` in the machine learning models.

    Examples
    --------
    >>> from balanced_moderation import BGATE, make_moderation_data
    >>> df = make_moderation_data(n=2000, random_state=1)
    >>> res = BGATE(random_state=1).fit(
    ...     df, y="y", d="d", z="z", x=["x0", "x1", "x2", "x3", "x4", "x5", "x6"],
    ...     balance=["x5"])
    >>> print(res.summary())  # doctest: +SKIP
    """

    def __init__(
        self,
        outcome_model=None,
        propensity_model=None,
        effect_model=None,
        moderator_model=None,
        n_folds: int = 2,
        n_folds_second: int | None = None,
        calibration: str | None = "venn_abers",
        calibration_splits: int = 2,
        normalize_weights: bool = True,
        propensity_clip: float = 1e-10,
        random_state=None,
    ):
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model
        self.effect_model = effect_model
        self.moderator_model = moderator_model
        self.n_folds = n_folds
        self.n_folds_second = n_folds_second
        self.calibration = calibration
        self.calibration_splits = calibration_splits
        self.normalize_weights = normalize_weights
        self.propensity_clip = propensity_clip
        self.random_state = random_state

    def _nuisances(self) -> Nuisances:
        outcome = self.outcome_model
        if outcome is None:
            outcome = RandomForestRegressor(
                n_estimators=500, min_samples_leaf=5, random_state=0)
        propensity = self.propensity_model
        if propensity is None:
            propensity = RandomForestClassifier(
                n_estimators=500, min_samples_leaf=5, random_state=0)
        return Nuisances(
            outcome_model=outcome,
            propensity_model=propensity,
            effect_model=outcome if self.effect_model is None else self.effect_model,
            moderator_model=(
                propensity if self.moderator_model is None else self.moderator_model),
            calibration=self.calibration,
            calibration_splits=self.calibration_splits,
        )

    def fit(
        self,
        data: pd.DataFrame,
        y: str,
        d: str,
        z: str,
        x: Sequence[str],
        balance: Sequence[str] | None = None,
    ) -> ModerationResults:
        """Estimate the moderation effect.

        Parameters
        ----------
        data : pandas.DataFrame
            The data. Must not contain missing values in the used columns.
        y : str
            Outcome column.
        d : str
            Binary treatment column (0/1 or boolean).
        z : str
            Binary moderator column (0/1 or boolean).
        x : list of str
            Control variables (confounders). Must not include ``z``.
        balance : list of str, optional
            Balancing variables W, a subset of ``x``. If omitted, the GATE
            difference (no balancing) is estimated.

        Returns
        -------
        ModerationResults
        """
        x = list(x)
        balance = None if balance is None else list(balance)
        _validate_columns(data, y, d, z, x, balance)
        _validate_settings(self.n_folds, self.n_folds_second, self.propensity_clip)
        if self.calibration is not None and self.calibration not in CALIBRATION_METHODS:
            raise ValueError(
                f"`calibration` must be one of {CALIBRATION_METHODS} or None, "
                f"got {self.calibration!r}.")
        if not isinstance(self.calibration_splits, (int, np.integer)) or self.calibration_splits < 2:
            raise ValueError("`calibration_splits` must be an integer of at least 2.")

        yv = data[y].to_numpy(dtype=float)
        dv = _binary(data[d], d)
        zv = _binary(data[z], z)
        xv = data[x].to_numpy(dtype=float)

        rng = check_random_state(self.random_state)
        nuis = self._nuisances()
        clip = self.propensity_clip

        if balance is None:
            tau1, tau0 = gate_scores(yv, dv, zv, xv, nuis, self.n_folds, rng, clip)
            e1, v1 = mean_and_variance(tau1)
            e0, v0 = mean_and_variance(tau0)
            estimates = np.array([e1, e0, e1 - e0])
            variances = np.array([v1, v0, v1 + v0])
            scores = {"Z=1": tau1, "Z=0": tau0}
            estimand = "GATE"
        else:
            wv = data[balance].to_numpy(dtype=float)
            psi1, psi0 = bgate_scores(
                yv, dv, zv, xv, wv, nuis, self.n_folds,
                self.n_folds_second or self.n_folds, rng,
                self.normalize_weights, clip)
            diff = psi1 - psi0
            moments = [mean_and_variance(s) for s in (psi1, psi0, diff)]
            estimates = np.array([m[0] for m in moments])
            variances = np.array([m[1] for m in moments])
            scores = {"Z=1": psi1, "Z=0": psi0, "difference": diff}
            estimand = "BGATE"

        return ModerationResults(
            estimand=estimand, moderator=z, balance=balance, estimates=estimates,
            variances=variances, n_obs=len(data), scores=scores)


def _validate_columns(data, y, d, z, x, balance):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")
    if len(x) == 0:
        raise ValueError("`x` must contain at least one control variable.")
    used = [y, d, z, *x]
    missing = [c for c in used + (balance or []) if c not in data.columns]
    if missing:
        raise KeyError(f"Columns not found in data: {missing}")
    if z in x:
        raise ValueError(f"The moderator '{z}' must not be included in `x`.")
    if y in x or d in x:
        raise ValueError("`x` must not contain the outcome or the treatment.")
    if balance is not None:
        if len(balance) == 0:
            raise ValueError("`balance` is empty. Use balance=None for the GATE.")
        not_in_x = [c for c in balance if c not in x]
        if not_in_x:
            raise ValueError(f"Balancing variables must also be in `x`: {not_in_x}")
    na = data[used].isna().any()
    if na.any():
        raise ValueError(f"Missing values in columns: {list(na[na].index)}")


def _validate_settings(n_folds, n_folds_second, clip):
    for name, k in (("n_folds", n_folds), ("n_folds_second", n_folds_second)):
        if k is not None and (not isinstance(k, (int, np.integer)) or k < 2):
            raise ValueError(f"`{name}` must be an integer of at least 2.")
    if not 0 <= clip < 0.5:
        raise ValueError("`propensity_clip` must be in [0, 0.5).")


def _binary(series: pd.Series, name: str) -> np.ndarray:
    values = series.to_numpy()
    if values.dtype == bool:
        return values.astype(float)
    uniq = set(np.unique(values).tolist())
    if not uniq <= {0, 1}:
        raise ValueError(f"'{name}' must be binary (0/1), found values {sorted(uniq)[:5]}.")
    if len(uniq) < 2:
        raise ValueError(f"'{name}' takes only one value; both 0 and 1 must occur.")
    return values.astype(float)
