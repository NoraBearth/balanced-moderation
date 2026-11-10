"""Statistical sanity checks against the known truth of the simulated designs."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor

from balanced_moderation import BGATE, make_moderation_data, true_moderation_effects

X = [f"x{j}" for j in range(10)]


@pytest.fixture(scope="module")
def truth():
    return true_moderation_effects(design="z_affects_x", n_draws=400_000)


def test_true_effects_differ_in_this_design(truth):
    # Z shifts the distribution of x5, so GATE and BGATE differ
    assert abs(truth["GATE"] - truth["BGATE_all"]) > 0.05


@pytest.mark.parametrize(
    "balance,key", [(None, "GATE"), (["x5"], "BGATE_x5"), (X, "BGATE_all")])
def test_estimate_close_to_truth(truth, balance, key):
    df = make_moderation_data(n=8000, design="z_affects_x", random_state=11)
    model = BGATE(
        GradientBoostingRegressor(random_state=0),
        LogisticRegression(max_iter=2000),
        n_folds=3, random_state=11)
    res = model.fit(df, "y", "d", "z", X, balance=balance)
    # within 3 standard errors of the truth
    assert abs(res.effect - truth[key]) < 3 * res.std_err, (res.effect, truth[key])
    assert res.std_err < 0.15
    assert np.isfinite(res.estimates).all()


def test_bgate_x5_truth_differs_from_gate(truth):
    assert truth["BGATE_x5"] < truth["GATE"] - 0.1
