import numpy as np

import pytest

from autoPyTorch.datasets.resampling_strategy import (
    CrossValFuncs,
    CrossValTypes,
    HoldOutFuncs,
    HoldoutValTypes,
    NoResamplingStrategyTypes,
    check_resampling_strategy
)


def test_holdoutfuncs():
    split = HoldOutFuncs()
    X = np.arange(10)
    y = np.ones(10)
    # Create a minority class
    y[:2] = 0
    train, val = split.holdout_validation(0, 0.5, X, shuffle=False)
    assert len(train) == len(val) == 5

    # No shuffling
    np.testing.assert_array_equal(X, np.arange(10))

    # Make sure the stratified version splits the minority class
    train, val = split.stratified_holdout_validation(0, 0.5, X, stratify=y)
    assert 0 in y[val]
    assert 0 in y[train]


def test_crossvalfuncs():
    split = CrossValFuncs()
    X = np.arange(100)
    y = np.ones(100)
    # Create a minority class
    y[:11] = 0
    splits = split.shuffle_split_cross_validation(0, 10, X)
    assert len(splits) == 10
    assert all([len(s[1]) == 10 for s in splits])

    # Make sure the stratified version splits the minority class
    splits = split.stratified_shuffle_split_cross_validation(0, 10, X, stratify=y)
    assert len(splits) == 10
    assert all([0 in y[s[1]] for s in splits])

    #
    splits = split.stratified_k_fold_cross_validation(0, 10, X, stratify=y)
    assert len(splits) == 10
    assert all([0 in y[s[1]] for s in splits])


def test_check_resampling_strategy():
    for rs in (CrossValTypes, HoldoutValTypes, NoResamplingStrategyTypes):
        for rs_func in rs:
            check_resampling_strategy(rs_func)

    with pytest.raises(ValueError):
        check_resampling_strategy(None)
