import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from balanced_moderation import BGATE, ModerationResults, make_moderation_data

X = [f"x{j}" for j in range(10)]


@pytest.fixture(scope="module")
def df():
    return make_moderation_data(n=1500, design="z_affects_x", random_state=7)


def linear_bgate(**kwargs):
    return BGATE(LinearRegression(), LogisticRegression(max_iter=1000), **kwargs)


# Reference values computed with the original replication code
# (github.com/NoraBearth/Causal-Machine-Learning-for-Moderation-Effects,
# estimators.estimate_effect, seed 7, 3 folds), see tools/compare_with_original.py.
# The logistic regression is solved to a tight tolerance so that results do not
# depend on the scikit-learn version.
REFERENCE = [
    # balance, normalize_weights, estimate, variance
    (None, True, 0.54960490, 2.134143e-02),
    (["x5"], False, 0.34060651, 2.473072e-02),
    (["x5"], True, 0.34035122, 2.550310e-02),
    (["x0", "x1", "x5"], False, 0.83683544, 1.789335e-02),
    (["x0", "x1", "x5"], True, 0.83423665, 2.140524e-02),
]


@pytest.mark.parametrize("balance,normalize,estimate,variance", REFERENCE)
def test_matches_original_replication_code(df, balance, normalize, estimate, variance):
    res = BGATE(LinearRegression(), LogisticRegression(max_iter=10000, tol=1e-12),
                n_folds=3, normalize_weights=normalize, calibration=None,
                random_state=7).fit(
        df, "y", "d", "z", X, balance=balance)
    assert res.effect == pytest.approx(estimate, abs=1e-6)
    assert res.variances[2] == pytest.approx(variance, rel=1e-5)


def test_result_structure(df):
    res = linear_bgate(random_state=0).fit(df, "y", "d", "z", X, balance=["x5"])
    assert isinstance(res, ModerationResults)
    assert res.estimand == "BGATE"
    assert res.labels == ["BGATE(z=1)", "BGATE(z=0)", "Difference"]
    # the difference is the difference of the group effects
    assert res.estimates[2] == pytest.approx(res.estimates[0] - res.estimates[1])
    frame = res.to_frame()
    assert list(frame.columns) == [
        "estimate", "std_err", "t_stat", "p_value", "ci_95_lower", "ci_95_upper"]
    lo, hi = res.conf_int()
    assert lo < res.effect < hi
    assert 0 <= res.pvalue <= 1
    assert "Difference" in res.summary()
    assert set(res.scores) == {"Z=1", "Z=0", "difference"}


def test_gate_without_balancing(df):
    res = linear_bgate(random_state=0).fit(df, "y", "d", "z", X)
    assert res.estimand == "GATE"
    assert res.variances[2] == pytest.approx(res.variances[0] + res.variances[1])


def test_reproducible_with_seed(df):
    a = linear_bgate(random_state=3).fit(df, "y", "d", "z", X, balance=["x5"])
    b = linear_bgate(random_state=3).fit(df, "y", "d", "z", X, balance=["x5"])
    np.testing.assert_array_equal(a.estimates, b.estimates)


def test_boolean_treatment_and_moderator(df):
    df2 = df.assign(d=df.d.astype(bool), z=df.z.astype(bool))
    a = linear_bgate(random_state=3).fit(df, "y", "d", "z", X, balance=["x5"])
    b = linear_bgate(random_state=3).fit(df2, "y", "d", "z", X, balance=["x5"])
    np.testing.assert_allclose(a.estimates, b.estimates)


def test_default_models_run():
    small = make_moderation_data(n=400, random_state=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = BGATE(n_folds=2, random_state=1).fit(
            small, "y", "d", "z", X, balance=["x5"])
    assert np.all(np.isfinite(res.estimates))


def _overconfident(calibration):
    # an unregularised tree predicts probabilities of exactly 0 and 1
    return BGATE(LinearRegression(), LogisticRegression(max_iter=1000),
                 moderator_model=DecisionTreeClassifier(random_state=0),
                 n_folds=2, calibration=calibration, random_state=0)


def test_overlap_warning_without_calibration(df):
    with pytest.warns(UserWarning, match="moderator propensity"):
        _overconfident(None).fit(df, "y", "d", "z", X, balance=X)


@pytest.mark.parametrize("method", ["platt", "beta", "venn_abers"])
def test_calibration_removes_extreme_propensities(df, method):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # any overlap warning fails
        res = _overconfident(method).fit(df, "y", "d", "z", X, balance=X)
    assert np.all(np.isfinite(res.estimates))


def test_invalid_calibration(df):
    with pytest.raises(ValueError, match="calibration"):
        linear_bgate(calibration="magic").fit(df, "y", "d", "z", X)


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        (dict(x=X + ["z"]), ValueError, "moderator"),
        (dict(balance=["not_there"]), KeyError, "not found"),
        (dict(balance=["x5"], x=["x0", "x1"]), ValueError, "also be in `x`"),
        (dict(balance=[]), ValueError, "empty"),
        (dict(x=[]), ValueError, "at least one"),
    ],
)
def test_input_validation(df, kwargs, error, match):
    args = dict(data=df, y="y", d="d", z="z", x=X, balance=None)
    args.update(kwargs)
    with pytest.raises(error, match=match):
        linear_bgate().fit(**args)


def test_non_binary_moderator(df):
    with pytest.raises(ValueError, match="binary"):
        linear_bgate().fit(df.assign(z=df.z * 2), "y", "d", "z", X)


def test_missing_values(df):
    df2 = df.copy()
    df2.loc[0, "x3"] = np.nan
    with pytest.raises(ValueError, match="Missing values"):
        linear_bgate().fit(df2, "y", "d", "z", X)


def test_requires_dataframe(df):
    with pytest.raises(TypeError):
        linear_bgate().fit(df.to_numpy(), "y", "d", "z", X)


@pytest.mark.parametrize("n_folds", [1, 2.5])
def test_invalid_folds(df, n_folds):
    with pytest.raises(ValueError, match="n_folds"):
        linear_bgate(n_folds=n_folds).fit(df, "y", "d", "z", X)


def test_fold_without_group_raises():
    df = make_moderation_data(n=200, random_state=0)
    df.loc[:, "d"] = 0
    df.loc[0, "d"] = 1  # a single treated observation
    with pytest.raises(ValueError, match="no observations|only one group"):
        linear_bgate(n_folds=2, random_state=0).fit(df, "y", "d", "z", X)


def test_numpy_dtype_columns():
    df = make_moderation_data(n=300, random_state=0)
    assert isinstance(df, pd.DataFrame)
    assert set(df.d.unique()) <= {0, 1} and set(df.z.unique()) <= {0, 1}
