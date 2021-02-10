import os
import pickle

import numpy as np

import pytest


import sklearn
import sklearn.datasets
from sklearn.ensemble import VotingClassifier

import torch

from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
    HoldOutTypes,
)
from autoPyTorch.datasets.tabular_dataset import TabularDataset


# Fixtures
# ========


# Test
# ========
@pytest.mark.parametrize('openml_id', (40981, ))
@pytest.mark.parametrize('resampling_strategy', (HoldOutTypes.holdout_validation,
                                                 CrossValTypes.k_fold_cross_validation,
                                                 ))
def test_classification(openml_id, resampling_strategy, backend):

    # Get the data and check that contents of data-manager make sense
    X, y = sklearn.datasets.fetch_openml(
        data_id=int(openml_id),
        return_X_y=True, as_frame=True
    )
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=1)
    datamanager = TabularDataset(
        X=X_train, Y=y_train,
        X_test=X_test, Y_test=y_test,
        splitting_type=resampling_strategy,
        dataset_name=str(openml_id),
    )
    assert datamanager.task_type == 'tabular_classification'
    expected_num_splits = 1 if resampling_strategy == HoldOutTypes.holdout_validation else 3
    assert len(datamanager.splits) == expected_num_splits

    # Search for a good configuration
    estimator = TabularClassificationTask(backend=backend)
    estimator.search(
        dataset=datamanager,
        optimize_metric='accuracy',
        total_walltime_limit=150,
        func_eval_time_limit=50,
        traditional_per_total_budget=0
    )

    # TODO: check for budget

    # Check for the created files
    tmp_dir = estimator._backend.temporary_directory
    loaded_datamanager = estimator._backend.load_datamanager()
    assert len(loaded_datamanager.train_tensors) == len(datamanager.train_tensors)

    expected_files = [
        'smac3-output/run_1/configspace.json',
        'smac3-output/run_1/runhistory.json',
        'smac3-output/run_1/scenario.txt',
        'smac3-output/run_1/stats.json',
        'smac3-output/run_1/train_insts.txt',
        'smac3-output/run_1/trajectory.json',
        '.autoPyTorch/datamanager.pkl',
        '.autoPyTorch/ensemble_read_preds.pkl',
        '.autoPyTorch/start_time_1',
        '.autoPyTorch/ensemble_history.json',
        '.autoPyTorch/ensemble_read_scores.pkl',
        '.autoPyTorch/true_targets_ensemble.npy',
    ]
    for expected_file in expected_files:
        assert os.path.exists(os.path.join(tmp_dir, expected_file)), expected_file

    # Check that smac was able to find proper models
    succesful_runs = [run_value.status for run_value in estimator.run_history.data.values(
    ) if 'SUCCESS' in str(run_value.status)]
    assert len(succesful_runs) > 1, estimator.run_history.data.items()

    # Search for an existing run key in disc. A individual model might have
    # a timeout and hence was not written to disc
    for i, (run_key, value) in enumerate(estimator.run_history.data.items()):
        if i == 0:
            # Ignore dummy run
            continue
        if 'SUCCESS' not in str(value.status):
            continue

        run_key_model_run_dir = estimator._backend.get_numrun_directory(
            estimator.seed, run_key.config_id, run_key.budget)
        if os.path.exists(run_key_model_run_dir):
            break

    if resampling_strategy == HoldOutTypes.holdout_validation:
        model_file = os.path.join(run_key_model_run_dir,
                                  f"{estimator.seed}.{run_key.config_id}.{run_key.budget}.model")
        assert os.path.exists(model_file), model_file
        model = estimator._backend.load_model_by_seed_and_id_and_budget(
            estimator.seed, run_key.config_id, run_key.budget)
        assert isinstance(model.named_steps['network'].get_network(), torch.nn.Module)
    elif resampling_strategy == CrossValTypes.k_fold_cross_validation:
        model_file = os.path.join(
            run_key_model_run_dir,
            f"{estimator.seed}.{run_key.config_id}.{run_key.budget}.cv_model"
        )
        assert os.path.exists(model_file), model_file
        model = estimator._backend.load_cv_model_by_seed_and_id_and_budget(
            estimator.seed, run_key.config_id, run_key.budget)
        assert isinstance(model, VotingClassifier)
        assert len(model.estimators_) == 3
        assert isinstance(model.estimators_[0].named_steps['network'].get_network(),
                          torch.nn.Module)
    else:
        pytest.fail(resampling_strategy)

    # Make sure that predictions on the test data are printed and make sense
    test_prediction = os.path.join(run_key_model_run_dir,
                                   estimator._backend.get_prediction_filename(
                                       'test', estimator.seed, run_key.config_id,
                                       run_key.budget))
    assert os.path.exists(test_prediction), test_prediction
    assert np.shape(np.load(test_prediction, allow_pickle=True))[0] == np.shape(X_test)[0]

    # Also, for ensemble builder, the OOF predictions should be there and match
    # the Ground truth that is also physically printed to disk
    ensemble_prediction = os.path.join(run_key_model_run_dir,
                                       estimator._backend.get_prediction_filename(
                                           'ensemble',
                                           estimator.seed, run_key.config_id,
                                           run_key.budget))
    assert os.path.exists(ensemble_prediction), ensemble_prediction
    assert np.shape(np.load(ensemble_prediction, allow_pickle=True))[0] == np.shape(
        estimator._backend.load_targets_ensemble()
    )[0]

    # Ensemble Builder produced an ensemble
    estimator.ensemble_ is not None

    # There should be a weight for each element of the ensemble
    assert len(estimator.ensemble_.identifiers_) == len(estimator.ensemble_.weights_)

    y_pred = estimator.predict(X_test)

    assert np.shape(y_pred)[0] == np.shape(X_test)[0]

    score = estimator.score(y_pred, y_test)
    assert 'accuracy' in score

    # Check that we can pickle
    # Test pickle
    dump_file = os.path.join(estimator._backend.temporary_directory, 'dump.pkl')

    with open(dump_file, 'wb') as f:
        pickle.dump(estimator, f)

    with open(dump_file, 'rb') as f:
        restored_estimator = pickle.load(f)
    restored_estimator.predict(X_test)
