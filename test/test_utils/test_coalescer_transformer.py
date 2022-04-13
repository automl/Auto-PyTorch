import numpy as np

import pytest

import scipy.sparse

from autoPyTorch.utils.implementations import MinorityCoalesceTransformer


@pytest.fixture
def X1():
    # Generates an array with categories 3, 4, 5, 6, 7 and occurences of 30%,
    # 30%, 30%, 5% and 5% respectively
    X = np.vstack((
        np.ones((30, 10)) * 3,
        np.ones((30, 10)) * 4,
        np.ones((30, 10)) * 5,
        np.ones((5, 10)) * 6,
        np.ones((5, 10)) * 7,
    ))
    for col in range(X.shape[1]):
        np.random.shuffle(X[:, col])
    return X


@pytest.fixture
def X2():
    # Generates an array with categories 3, 4, 5, 6, 7 and occurences of 5%,
    # 5%, 5%, 35% and 50% respectively
    X = np.vstack((
        np.ones((5, 10)) * 3,
        np.ones((5, 10)) * 4,
        np.ones((5, 10)) * 5,
        np.ones((35, 10)) * 6,
        np.ones((50, 10)) * 7,
    ))
    for col in range(X.shape[1]):
        np.random.shuffle(X[:, col])
    return X


def test_default(X1):
    X = X1
    X_copy = np.copy(X)
    Y = MinorityCoalesceTransformer().fit_transform(X)
    np.testing.assert_array_almost_equal(Y, X_copy)
    # Assert no copies were made
    assert id(X) == id(Y)


def test_coalesce_10_percent(X1):
    X = X1
    Y = MinorityCoalesceTransformer(min_frac=.1).fit_transform(X)
    for col in range(Y.shape[1]):
        hist = np.histogram(Y[:, col], bins=np.arange(-2, 7))
        np.testing.assert_array_almost_equal(hist[0], [10, 0, 0, 0, 0, 30, 30, 30])
    # Assert no copies were made
    assert id(X) == id(Y)


def test_coalesce_10_percent_sparse(X1):
    X = scipy.sparse.csc_matrix(X1)
    Y = MinorityCoalesceTransformer(min_frac=.1).fit_transform(X)
    # Assert no copies were made
    assert id(X) == id(Y)
    Y = Y.todense()
    for col in range(Y.shape[1]):
        hist = np.histogram(Y[:, col], bins=np.arange(-2, 7))
        np.testing.assert_array_almost_equal(hist[0], [10, 0, 0, 0, 0, 30, 30, 30])


def test_invalid_X(X1):
    X = X1 - 5
    with pytest.raises(ValueError):
        MinorityCoalesceTransformer().fit_transform(X)


@pytest.mark.parametrize("min_frac", [-0.1, 1.1])
def test_invalid_min_frac(min_frac):
    with pytest.raises(ValueError):
        MinorityCoalesceTransformer(min_frac=min_frac)


def test_transform_before_fit(X1):
    with pytest.raises(RuntimeError):
        MinorityCoalesceTransformer().transform(X1)


def test_transform_after_fit(X1, X2):
    # On both X_fit and X_transf, the categories 3, 4, 5, 6, 7 are present.
    X_fit = X1  # Here categories 3, 4, 5 have ocurrence above 10%
    X_transf = X2  # Here it is the opposite, just categs 6 and 7 are above 10%

    mc = MinorityCoalesceTransformer(min_frac=.1).fit(X_fit)

    # transform() should coalesce categories as learned during fit.
    # Category distribution in X_transf should be irrelevant.
    Y = mc.transform(X_transf)
    for col in range(Y.shape[1]):
        hist = np.histogram(Y[:, col], bins=np.arange(-2, 7))
        np.testing.assert_array_almost_equal(hist[0], [85, 0, 0, 0, 0, 5, 5, 5])
