"""Realistic inputs: pandas types, index and column order, sklearn meta-estimators."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from balanced_moderation import BGATE, make_moderation_data

X = [f"x{j}" for j in range(10)]


@pytest.fixture(scope="module")
def df():
    return make_moderation_data(n=1000, design="z_affects_x", random_state=5)


def fit(data, x=X, balance=("x5",), **kwargs):
    model = BGATE(LinearRegression(), LogisticRegression(max_iter=1000),
                  n_folds=3, random_state=0, **kwargs)
    return model.fit(data, "y", "d", "z", list(x),
                     balance=None if balance is None else list(balance))


@pytest.mark.parametrize("calibration", [None, "venn_abers"])
@pytest.mark.parametrize("balance", [None, ("x5",)])
def test_index_and_column_order_do_not_matter(df, calibration, balance):
    base = fit(df, balance=balance, calibration=calibration)
    # string index, shuffled column order, extra unused columns
    other = df.copy()
    other.index = [f"id{i}" for i in range(len(df))][::-1]
    other["unused"] = "text"
    other = other[["unused"] + list(reversed(other.columns[:-1]))]
    res = fit(other, balance=balance, calibration=calibration)
    np.testing.assert_allclose(res.estimates, base.estimates, rtol=1e-12)


def test_pandas_nullable_dtypes(df):
    base = fit(df)
    other = df.astype({"d": "Int64", "z": "boolean", "x5": "Float64"})
    np.testing.assert_allclose(fit(other).estimates, base.estimates, rtol=1e-12)


def test_integer_and_float_coded_binary_variables(df):
    base = fit(df)
    other = df.astype({"d": float, "z": "int8"})
    np.testing.assert_allclose(fit(other).estimates, base.estimates, rtol=1e-12)


def test_missing_value_in_nullable_column(df):
    other = df.astype({"z": "Int64"})
    other.loc[3, "z"] = pd.NA
    with pytest.raises(ValueError, match="Missing values"):
        fit(other)


def test_pipelines_and_grid_search(df):
    outcome = make_pipeline(StandardScaler(), LinearRegression())
    propensity = GridSearchCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        {"logisticregression__C": [0.1, 1.0]}, cv=2)
    res = BGATE(outcome, propensity, n_folds=2, random_state=0).fit(
        df, "y", "d", "z", X, balance=["x5"])
    assert np.all(np.isfinite(res.estimates))


def test_random_state_none_runs(df):
    res = BGATE(LinearRegression(), LogisticRegression(max_iter=1000),
                n_folds=2).fit(df, "y", "d", "z", X, balance=["x5"])
    assert np.all(np.isfinite(res.estimates))


def test_n_folds_second_differs_from_n_folds(df):
    res = fit(df, n_folds_second=2)
    assert np.all(np.isfinite(res.estimates))


def test_data_not_modified(df):
    before = df.copy()
    fit(df)
    pd.testing.assert_frame_equal(df, before)


def test_estimator_is_reusable(df):
    model = BGATE(LinearRegression(), LogisticRegression(max_iter=1000),
                  n_folds=3, random_state=0)
    a = model.fit(df, "y", "d", "z", X, balance=["x5"])
    model.fit(df, "y", "d", "z", X)  # a different fit in between
    b = model.fit(df, "y", "d", "z", X, balance=["x5"])
    np.testing.assert_array_equal(a.estimates, b.estimates)


def test_bgate_balanced_on_all_x_equals_difference_of_components(df):
    res = fit(df, balance=X)
    assert res.estimates[2] == pytest.approx(res.estimates[0] - res.estimates[1])
    assert np.mean(res.scores["difference"]) == pytest.approx(res.effect)
