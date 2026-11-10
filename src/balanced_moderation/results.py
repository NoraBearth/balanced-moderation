"""Results container returned by :meth:`balanced_moderation.BGATE.fit`."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ModerationResults:
    """Estimates of a moderation effect and the group effects behind it.

    Attributes
    ----------
    estimand : str
        ``"BGATE"`` if balancing variables were given, otherwise ``"GATE"``.
    moderator : str
        Name of the moderator variable Z.
    balance : list of str or None
        The balancing variables (``None`` for the GATE).
    estimates : numpy.ndarray
        Point estimates for [effect in Z=1, effect in Z=0, difference].
    variances : numpy.ndarray
        Estimated variances of the three point estimates.
    n_obs : int
        Number of observations used.
    scores : dict of numpy.ndarray
        Observation-level scores whose means are the estimates.
    """

    estimand: str
    moderator: str
    balance: list[str] | None
    estimates: np.ndarray
    variances: np.ndarray
    n_obs: int
    scores: dict = field(repr=False, default_factory=dict)

    @property
    def labels(self) -> list[str]:
        return [
            f"{self.estimand}({self.moderator}=1)",
            f"{self.estimand}({self.moderator}=0)",
            "Difference",
        ]

    @property
    def effect(self) -> float:
        """The moderation effect: effect in Z=1 minus effect in Z=0."""
        return float(self.estimates[2])

    @property
    def std_err(self) -> float:
        """Standard error of the moderation effect."""
        return float(np.sqrt(self.variances[2]))

    @property
    def pvalue(self) -> float:
        """Two-sided p-value for H0: no moderation effect."""
        return float(2 * stats.norm.sf(abs(self.effect / self.std_err)))

    def conf_int(self, alpha: float = 0.05) -> tuple[float, float]:
        """Confidence interval for the moderation effect."""
        q = stats.norm.ppf(1 - alpha / 2)
        return (self.effect - q * self.std_err, self.effect + q * self.std_err)

    def to_frame(self, alpha: float = 0.05) -> pd.DataFrame:
        """All three estimates with standard errors, CIs and p-values."""
        se = np.sqrt(self.variances)
        q = stats.norm.ppf(1 - alpha / 2)
        level = int(round(100 * (1 - alpha)))
        return pd.DataFrame(
            {
                "estimate": self.estimates,
                "std_err": se,
                "t_stat": self.estimates / se,
                "p_value": 2 * stats.norm.sf(np.abs(self.estimates / se)),
                f"ci_{level}_lower": self.estimates - q * se,
                f"ci_{level}_upper": self.estimates + q * se,
            },
            index=self.labels,
        )

    def summary(self, alpha: float = 0.05) -> str:
        """A printable summary table."""
        if self.balance:
            balance = ", ".join(self.balance)
            title = f"Balanced GATE (BGATE) - balanced over: {balance}"
        else:
            title = "Group average treatment effects (GATE) - no balancing"
        table = self.to_frame(alpha).to_string(float_format=lambda v: f"{v:.4f}")
        width = max(len(line) for line in table.splitlines())
        rule = "=" * max(width, len(title))
        return "\n".join([
            rule, title, f"Moderator: {self.moderator}    Observations: {self.n_obs}",
            rule, table, rule,
        ])

    def __str__(self) -> str:
        return self.summary()
