import numpy as np

from autoPyTorch.datasets.resampling_strategy import CrossValFuncs, HoldOutFuncs


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

    # Forecasting
    n_prediction_steps = 3
    n_repeats = 1
    train, val = split.time_series_hold_out_validation(0, 0, X, n_prediction_steps=n_prediction_steps,
                                                       n_repeats=n_repeats)
    # val must start n_predictions_steps after train
    assert val[0] - train[-1] == n_prediction_steps
    assert len(val) == n_repeats

    n_prediction_steps = 2
    n_repeats = 2
    train, val = split.time_series_hold_out_validation(0, 0, X, n_prediction_steps=n_prediction_steps,
                                                       n_repeats=n_repeats)
    assert val[0] - train[-1] == n_prediction_steps
    assert len(val) == n_repeats
    # No overlapping between different splits
    assert val[1] - val[0] == n_prediction_steps

    # Failure case
    # Forecasting
    n_prediction_steps = 10
    n_repeats = 1
    train, val = split.time_series_hold_out_validation(0, 0, X, n_prediction_steps=n_prediction_steps,
                                                       n_repeats=n_repeats)
    # n_prediction steps is larger than the length of the sequence
    assert len(train) == 0
    assert val == 9

    # TODO Theoretically, this should work properly, we need to write our own spliter
    n_prediction_steps = 2
    n_repeats = 3
    train, val = split.time_series_hold_out_validation(0, 0, X, n_prediction_steps=n_prediction_steps,
                                                       n_repeats=n_repeats)
    assert len(train) == 0
    assert val == 9


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

    def eval_ts_cv(num_splits, n_prediction_steps, n_repeats):
        splits = split.time_series_cross_validation(0, num_splits, X,
                                                    n_prediction_steps=n_prediction_steps, n_repeats=n_repeats)
        assert len(splits) == num_splits
        for i, sp in enumerate(splits):
            assert len(sp[1]) == n_repeats
            assert sp[1][0] - sp[0][-1] == n_prediction_steps
            if i > 0:
                assert sp[1][0] - splits[i - 1][1][-1] == n_prediction_steps

    eval_ts_cv(2, 10, 1)
    eval_ts_cv(3, 10, 3)

    def eval_ts_sea_cv(num_splits, n_prediction_steps, n_repeats, freq_value):
        seasonality_h_value = int(np.round((n_prediction_steps // int(freq_value) + 1) * freq_value))
        splits = split.time_series_ts_cross_validation(0, num_splits=num_splits,
                                                       indices=X,
                                                       n_prediction_steps=n_prediction_steps,
                                                       n_repeats=n_repeats,
                                                       seasonality_h_value=seasonality_h_value)
        assert len(splits) == num_splits
        assert splits[0][1][-1] == len(X) - 1
        if num_splits > 1:
            for i in range(1, num_splits):
                dis_val_start_to_test = len(X) - 1 - (splits[i][1] - n_prediction_steps)
                assert np.all(dis_val_start_to_test % freq_value == 0)

    eval_ts_sea_cv(2, 10, 2, 6)
    eval_ts_sea_cv(2, 10, 1, 12)
    eval_ts_sea_cv(3, 10, 1, 6)

    n_prediction_steps = 10
    freq_value = 24
    n_repeats = 1
    num_splits = 2
    seasonality_h_value = int(np.round((n_prediction_steps // int(freq_value) + 1) * freq_value))

    sp2 = split.time_series_ts_cross_validation(0, num_splits=num_splits,
                                                indices=X[:10],
                                                n_prediction_steps=n_prediction_steps,
                                                n_repeats=n_repeats,
                                                seasonality_h_value=seasonality_h_value)
    # We cannot do a split, thus the two splits are the same

    assert np.all(sp2[1][1] == sp2[0][1])
