"""balanced_moderation: causal machine learning for moderation effects.

Implements the balanced group average treatment effect (BGATE) of
Bearth & Lechner (2024), "Causal Machine Learning for Moderation Effects",
arXiv:2401.08290, estimated with double/debiased machine learning.
"""

from importlib.metadata import PackageNotFoundError, version

from .datasets import make_moderation_data, true_moderation_effects
from .estimator import BGATE
from .results import ModerationResults

try:
    __version__ = version("balanced-moderation")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "unknown"

__all__ = [
    "BGATE",
    "ModerationResults",
    "make_moderation_data",
    "true_moderation_effects",
    "__version__",
]
