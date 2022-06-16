import json
import os
import pathlib
import pickle
import tempfile
import unittest
from test.test_api.utils import dummy_do_dummy_prediction, dummy_eval_train_function

import ConfigSpace as CS
from ConfigSpace.configuration_space import Configuration

import numpy as np

import pandas as pd

import pytest


import sklearn
import sklearn.datasets
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import VotingClassifier, VotingRegressor

from smac.runhistory.runhistory import RunHistory, RunInfo, RunValue

from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.api.tabular_regression import TabularRegressionTask
from autoPyTorch.api.time_series_forecasting import TimeSeriesForecastingTask
from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
    HoldoutValTypes,
    NoResamplingStrategyTypes,
)
from autoPyTorch.optimizer.smbo import AutoMLSMBO
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner import _traditional_learners
from autoPyTorch.pipeline.components.training.metrics.metrics import accuracy


CV_NUM_SPLITS = 2
HOLDOUT_NUM_SPLITS = 1


# Test
# ====
@unittest.mock.patch('autoPyTorch.evaluation.train_evaluator.eval_train_function',
                     new=dummy_eval_train_function)
@pytest.mark.parametrize('openml_id', (40981, ))
@pytest.mark.parametrize('resampling_strategy,resampling_strategy_args',
                         ((HoldoutValTypes.holdout_validation, None),
                          (CrossValTypes.k_fold_cross_validation, {'num_splits': CV_NUM_SPLITS})
                          ))
def test_tabular_classification(openml_id, resampling_strategy, backend, resampling_strategy_args, n_samples):

    # Get the data and check that contents of data-manager make sense
    X, y = sklearn.datasets.fetch_openml(
        data_id=int(openml_id),
        return_X_y=True, as_frame=True
    )
    X, y = X.iloc[:n_samples], y.iloc[:n_samples]

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=42)

    # Search for a good configuration
    estimator = TabularClassificationTask(
        backend=backend,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        seed=42,
    )

    with unittest.mock.patch.object(estimator, '_do_dummy_prediction', new=dummy_do_dummy_prediction):
        estimator.search(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            optimize_metric='accuracy',
            total_walltime_limit=40,
            func_eval_time_limit_secs=10,
            enable_traditional_pipeline=False,
        )

    # Internal dataset has expected settings
    assert estimator.dataset.task_type == 'tabular_classification'
    expected_num_splits = HOLDOUT_NUM_SPLITS if resampling_strategy == HoldoutValTypes.holdout_validation \
        else CV_NUM_SPLITS
    assert estimator.resampling_strategy == resampling_strategy
    assert estimator.dataset.resampling_strategy == resampling_strategy
    assert len(estimator.dataset.splits) == expected_num_splits

    # TODO: check for budget

    # Check for the created files
    tmp_dir = estimator._backend.temporary_directory
    loaded_datamanager = estimator._backend.load_datamanager()
    assert len(loaded_datamanager.train_tensors) == len(estimator.dataset.train_tensors)

    expected_files = [
        'smac3-output/run_42/configspace.json',
        'smac3-output/run_42/runhistory.json',
        'smac3-output/run_42/scenario.txt',
        'smac3-output/run_42/stats.json',
        'smac3-output/run_42/train_insts.txt',
        'smac3-output/run_42/trajectory.json',
        '.autoPyTorch/datamanager.pkl',
        '.autoPyTorch/ensemble_read_preds.pkl',
        '.autoPyTorch/start_time_42',
        '.autoPyTorch/ensemble_history.json',
        '.autoPyTorch/ensemble_read_losses.pkl',
        '.autoPyTorch/true_targets_ensemble.npy',
    ]
    for expected_file in expected_files:
        assert os.path.exists(os.path.join(tmp_dir, expected_file)), "{}/{}/{}".format(
            tmp_dir,
            [data for data in pathlib.Path(tmp_dir).glob('*')],
            expected_file,
        )

    # Check that smac was able to find proper models
    succesful_runs = [run_value.status for run_value in estimator.run_history.data.values(
    ) if 'SUCCESS' in str(run_value.status)]
    assert len(succesful_runs) > 1, [(k, v) for k, v in estimator.run_history.data.items()]

    # Search for an existing run key in disc. A individual model might have
    # a timeout and hence was not written to disc
    successful_num_run = None
    SUCCESS = False
    for i, (run_key, value) in enumerate(estimator.run_history.data.items()):
        if 'SUCCESS' in str(value.status):
            run_key_model_run_dir = estimator._backend.get_numrun_directory(
                estimator.seed, run_key.config_id + 1, run_key.budget)
            successful_num_run = run_key.config_id + 1
            if os.path.exists(run_key_model_run_dir):
                # Runkey config id is different from the num_run
                # more specifically num_run = config_id + 1(dummy)
                SUCCESS = True
                break

    assert SUCCESS, f"Successful run was not properly saved for num_run: {successful_num_run}"

    if resampling_strategy == HoldoutValTypes.holdout_validation:
        model_file = os.path.join(run_key_model_run_dir,
                                  f"{estimator.seed}.{successful_num_run}.{run_key.budget}.model")
        assert os.path.exists(model_file), model_file
        model = estimator._backend.load_model_by_seed_and_id_and_budget(
            estimator.seed, successful_num_run, run_key.budget)
    elif resampling_strategy == CrossValTypes.k_fold_cross_validation:
        model_file = os.path.join(
            run_key_model_run_dir,
            f"{estimator.seed}.{successful_num_run}.{run_key.budget}.cv_model"
        )
        assert os.path.exists(model_file), model_file

        model = estimator._backend.load_cv_model_by_seed_and_id_and_budget(
            estimator.seed, successful_num_run, run_key.budget)
        assert isinstance(model, VotingClassifier)
        assert len(model.estimators_) == CV_NUM_SPLITS
    else:
        pytest.fail(resampling_strategy)

    # Make sure that predictions on the test data are printed and make sense
    test_prediction = os.path.join(run_key_model_run_dir,
                                   estimator._backend.get_prediction_filename(
                                       'test', estimator.seed, successful_num_run,
                                       run_key.budget))
    assert os.path.exists(test_prediction), test_prediction
    assert np.shape(np.load(test_prediction, allow_pickle=True))[0] == np.shape(X_test)[0]

    # Also, for ensemble builder, the OOF predictions should be there and match
    # the Ground truth that is also physically printed to disk
    ensemble_prediction = os.path.join(run_key_model_run_dir,
                                       estimator._backend.get_prediction_filename(
                                           'ensemble',
                                           estimator.seed, successful_num_run,
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

    # Make sure that predict proba has the expected shape
    probabilites = estimator.predict_proba(X_test)
    assert np.shape(probabilites) == (np.shape(X_test)[0], 2)

    score = estimator.score(y_pred, y_test)
    assert 'accuracy' in score

    # check incumbent config and results
    incumbent_config, incumbent_results = estimator.get_incumbent_results()
    assert isinstance(incumbent_config, Configuration)
    assert isinstance(incumbent_results, dict)
    assert 'opt_loss' in incumbent_results, "run history: {}, successful_num_run: {}".format(estimator.run_history.data,
                                                                                             successful_num_run)
    assert 'train_loss' in incumbent_results

    # Check that we can pickle
    dump_file = os.path.join(estimator._backend.temporary_directory, 'dump.pkl')

    with open(dump_file, 'wb') as f:
        pickle.dump(estimator, f)

    with open(dump_file, 'rb') as f:
        restored_estimator = pickle.load(f)
    restored_estimator.predict(X_test)

    # Test refit on dummy data
    estimator.refit(dataset=backend.load_datamanager())

    # Make sure that a configuration space is stored in the estimator
    assert isinstance(estimator.get_search_space(), CS.ConfigurationSpace)


@pytest.mark.parametrize('openml_name', ("boston", ))
@unittest.mock.patch('autoPyTorch.evaluation.train_evaluator.eval_train_function',
                     new=dummy_eval_train_function)
@pytest.mark.parametrize('resampling_strategy,resampling_strategy_args',
                         ((HoldoutValTypes.holdout_validation, None),
                          (CrossValTypes.k_fold_cross_validation, {'num_splits': CV_NUM_SPLITS})
                          ))
def test_tabular_regression(openml_name, resampling_strategy, backend, resampling_strategy_args, n_samples):

    # Get the data and check that contents of data-manager make sense
    X, y = sklearn.datasets.fetch_openml(
        openml_name,
        return_X_y=True,
        as_frame=True
    )
    X, y = X.iloc[:n_samples], y.iloc[:n_samples]

    # normalize values
    y = (y - y.mean()) / y.std()

    # fill NAs for now since they are not yet properly handled
    for column in X.columns:
        if X[column].dtype.name == "category":
            X[column] = pd.Categorical(X[column],
                                       categories=list(X[column].cat.categories) + ["missing"]).fillna("missing")
        else:
            X[column] = X[column].fillna(0)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=1)

    # Search for a good configuration
    estimator = TabularRegressionTask(
        backend=backend,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        seed=42,
    )

    with unittest.mock.patch.object(estimator, '_do_dummy_prediction', new=dummy_do_dummy_prediction):
        estimator.search(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            optimize_metric='r2',
            total_walltime_limit=40,
            func_eval_time_limit_secs=10,
            enable_traditional_pipeline=False,
        )

    # Internal dataset has expected settings
    assert estimator.dataset.task_type == 'tabular_regression'
    expected_num_splits = HOLDOUT_NUM_SPLITS if resampling_strategy == HoldoutValTypes.holdout_validation\
        else CV_NUM_SPLITS
    assert estimator.resampling_strategy == resampling_strategy
    assert estimator.dataset.resampling_strategy == resampling_strategy
    assert len(estimator.dataset.splits) == expected_num_splits

    # TODO: check for budget

    # Check for the created files
    tmp_dir = estimator._backend.temporary_directory
    loaded_datamanager = estimator._backend.load_datamanager()
    assert len(loaded_datamanager.train_tensors) == len(estimator.dataset.train_tensors)

    expected_files = [
        'smac3-output/run_42/configspace.json',
        'smac3-output/run_42/runhistory.json',
        'smac3-output/run_42/scenario.txt',
        'smac3-output/run_42/stats.json',
        'smac3-output/run_42/train_insts.txt',
        'smac3-output/run_42/trajectory.json',
        '.autoPyTorch/datamanager.pkl',
        '.autoPyTorch/ensemble_read_preds.pkl',
        '.autoPyTorch/start_time_42',
        '.autoPyTorch/ensemble_history.json',
        '.autoPyTorch/ensemble_read_losses.pkl',
        '.autoPyTorch/true_targets_ensemble.npy',
    ]
    for expected_file in expected_files:
        assert os.path.exists(os.path.join(tmp_dir, expected_file)), expected_file

    # Check that smac was able to find proper models
    succesful_runs = [run_value.status for run_value in estimator.run_history.data.values(
    ) if 'SUCCESS' in str(run_value.status)]
    assert len(succesful_runs) >= 1, [(k, v) for k, v in estimator.run_history.data.items()]

    # Search for an existing run key in disc. A individual model might have
    # a timeout and hence was not written to disc
    successful_num_run = None
    SUCCESS = False
    for i, (run_key, value) in enumerate(estimator.run_history.data.items()):
        if 'SUCCESS' in str(value.status):
            run_key_model_run_dir = estimator._backend.get_numrun_directory(
                estimator.seed, run_key.config_id + 1, run_key.budget)
            successful_num_run = run_key.config_id + 1
            if os.path.exists(run_key_model_run_dir):
                # Runkey config id is different from the num_run
                # more specifically num_run = config_id + 1(dummy)
                SUCCESS = True
                break

    assert SUCCESS, f"Successful run was not properly saved for num_run: {successful_num_run}"

    if resampling_strategy == HoldoutValTypes.holdout_validation:
        model_file = os.path.join(run_key_model_run_dir,
                                  f"{estimator.seed}.{successful_num_run}.{run_key.budget}.model")
        assert os.path.exists(model_file), model_file
        model = estimator._backend.load_model_by_seed_and_id_and_budget(
            estimator.seed, successful_num_run, run_key.budget)
    elif resampling_strategy == CrossValTypes.k_fold_cross_validation:
        model_file = os.path.join(
            run_key_model_run_dir,
            f"{estimator.seed}.{successful_num_run}.{run_key.budget}.cv_model"
        )
        assert os.path.exists(model_file), model_file
        model = estimator._backend.load_cv_model_by_seed_and_id_and_budget(
            estimator.seed, successful_num_run, run_key.budget)
        assert isinstance(model, VotingRegressor)
        assert len(model.estimators_) == CV_NUM_SPLITS
    else:
        pytest.fail(resampling_strategy)

    # Make sure that predictions on the test data are printed and make sense
    test_prediction = os.path.join(run_key_model_run_dir,
                                   estimator._backend.get_prediction_filename(
                                       'test', estimator.seed, successful_num_run,
                                       run_key.budget))
    assert os.path.exists(test_prediction), test_prediction
    assert np.shape(np.load(test_prediction, allow_pickle=True))[0] == np.shape(X_test)[0]

    # Also, for ensemble builder, the OOF predictions should be there and match
    # the Ground truth that is also physically printed to disk
    ensemble_prediction = os.path.join(run_key_model_run_dir,
                                       estimator._backend.get_prediction_filename(
                                           'ensemble',
                                           estimator.seed, successful_num_run,
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
    assert 'r2' in score

    # check incumbent config and results
    incumbent_config, incumbent_results = estimator.get_incumbent_results()
    assert isinstance(incumbent_config, Configuration)
    assert isinstance(incumbent_results, dict)
    assert 'opt_loss' in incumbent_results, "run history: {}, successful_num_run: {}".format(estimator.run_history.data,
                                                                                             successful_num_run)
    assert 'train_loss' in incumbent_results, estimator.run_history.data

    # Check that we can pickle
    dump_file = os.path.join(estimator._backend.temporary_directory, 'dump.pkl')

    with open(dump_file, 'wb') as f:
        pickle.dump(estimator, f)

    with open(dump_file, 'rb') as f:
        restored_estimator = pickle.load(f)
    restored_estimator.predict(X_test)

    # Test refit on dummy data
    estimator.refit(dataset=backend.load_datamanager())

    # Make sure that a configuration space is stored in the estimator
    assert isinstance(estimator.get_search_space(), CS.ConfigurationSpace)

    representation = estimator.show_models()
    assert isinstance(representation, str)
    assert 'Weight' in representation
    assert 'Preprocessing' in representation
    assert 'Estimator' in representation


@pytest.mark.parametrize('forecasting_toy_dataset', ['uni_variant_wo_missing'], indirect=True)
@pytest.mark.parametrize('resampling_strategy,resampling_strategy_args',
                         ((HoldoutValTypes.time_series_hold_out_validation, None),
                          ))
def test_time_series_forecasting(forecasting_toy_dataset, resampling_strategy, backend, resampling_strategy_args):
    forecast_horizon = 3
    freq = '1Y'
    X, Y = forecasting_toy_dataset

    if X is not None:
        X_train = []
        X_test = []
        for x in X:
            if hasattr(x, 'iloc'):
                X_train.append(x.iloc[:-forecast_horizon].copy())
                X_test.append(x.iloc[-forecast_horizon:].copy())
            else:
                X_train.append(x[:-forecast_horizon].copy())
                X_test.append(x[-forecast_horizon:].copy())
        known_future_features = tuple(X[0].columns) if isinstance(X[0], pd.DataFrame) else \
            np.arange(X[0].shape[-1]).tolist()
    else:
        X_train = None
        X_test = None
        known_future_features = None

    y_train = []
    y_test = []

    for y in Y:
        if hasattr(y, 'iloc'):
            y_train.append(y.iloc[:-forecast_horizon].copy())
            y_test.append(y.iloc[-forecast_horizon:].copy())
        else:
            y_train.append(y[:-forecast_horizon].copy())
            y_test.append(y[-forecast_horizon:].copy())

    # Search for a good configuration
    # patch.mock  is not applied to partial func. We only test lightweight FFNN networks
    estimator = TimeSeriesForecastingTask(
        backend=backend,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        seed=42,
        include_components={'network_backbone': {'flat_encoder:MLPEncoder'}}
    )

    with unittest.mock.patch.object(estimator, '_do_dummy_prediction', new=dummy_do_dummy_prediction):
        estimator.search(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            memory_limit=None,
            optimize_metric='mean_MSE_forecasting',
            n_prediction_steps=forecast_horizon,
            freq=freq,
            total_walltime_limit=50,
            func_eval_time_limit_secs=20,
            known_future_features=known_future_features,
        )

    # Internal dataset has expected settings
    assert estimator.dataset.task_type == 'time_series_forecasting'
    expected_num_splits = HOLDOUT_NUM_SPLITS if resampling_strategy == HoldoutValTypes.time_series_hold_out_validation \
        else CV_NUM_SPLITS
    assert estimator.resampling_strategy == resampling_strategy
    assert estimator.dataset.resampling_strategy == resampling_strategy

    assert len(estimator.dataset.splits) == expected_num_splits

    # Check for the created files
    tmp_dir = estimator._backend.temporary_directory
    loaded_datamanager = estimator._backend.load_datamanager()
    assert len(loaded_datamanager.train_tensors) == len(estimator.dataset.train_tensors)

    expected_files = [
        'smac3-output/run_42/configspace.json',
        'smac3-output/run_42/runhistory.json',
        'smac3-output/run_42/scenario.txt',
        'smac3-output/run_42/stats.json',
        'smac3-output/run_42/train_insts.txt',
        'smac3-output/run_42/trajectory.json',
        '.autoPyTorch/datamanager.pkl',
        '.autoPyTorch/ensemble_read_preds.pkl',
        '.autoPyTorch/start_time_42',
        '.autoPyTorch/ensemble_history.json',
        '.autoPyTorch/ensemble_read_losses.pkl',
        '.autoPyTorch/true_targets_ensemble.npy',
    ]
    for expected_file in expected_files:
        assert os.path.exists(os.path.join(tmp_dir, expected_file)), expected_file

    # Check that smac was able to find proper models
    succesful_runs = [run_value.status for run_value in estimator.run_history.data.values(
    ) if 'SUCCESS' in str(run_value.status)]
    assert len(succesful_runs) >= 1, [(k, v) for k, v in estimator.run_history.data.items()]

    # Search for an existing run key in disc. A individual model might have
    # a timeout and hence was not written to disc
    successful_num_run = None
    SUCCESS = False
    for i, (run_key, value) in enumerate(estimator.run_history.data.items()):
        if 'SUCCESS' in str(value.status):
            run_key_model_run_dir = estimator._backend.get_numrun_directory(
                estimator.seed, run_key.config_id + 1, run_key.budget)
            successful_num_run = run_key.config_id + 1
            if os.path.exists(run_key_model_run_dir):
                # Runkey config id is different from the num_run
                # more specifically num_run = config_id + 1(dummy)
                SUCCESS = True
                break

    assert SUCCESS, f"Successful run was not properly saved for num_run: {successful_num_run}"

    if resampling_strategy == HoldoutValTypes.time_series_hold_out_validation:
        model_file = os.path.join(run_key_model_run_dir,
                                  f"{estimator.seed}.{successful_num_run}.{run_key.budget}.model")
        assert os.path.exists(model_file), model_file
        model = estimator._backend.load_model_by_seed_and_id_and_budget(
            estimator.seed, successful_num_run, run_key.budget)
    elif resampling_strategy == CrossValTypes.time_series_cross_validation:
        model_file = os.path.join(
            run_key_model_run_dir,
            f"{estimator.seed}.{successful_num_run}.{run_key.budget}.cv_model"
        )
        assert os.path.exists(model_file), model_file
        model = estimator._backend.load_cv_model_by_seed_and_id_and_budget(
            estimator.seed, successful_num_run, run_key.budget)
        assert isinstance(model, VotingRegressor)
        assert len(model.estimators_) == CV_NUM_SPLITS
    else:
        pytest.fail(resampling_strategy)

    # Make sure that predictions on the test data are printed and make sense
    test_prediction = os.path.join(run_key_model_run_dir,
                                   estimator._backend.get_prediction_filename(
                                       'test', estimator.seed, successful_num_run,
                                       run_key.budget))
    assert os.path.exists(test_prediction), test_prediction
    assert np.shape(np.load(test_prediction, allow_pickle=True))[0] == forecast_horizon * np.shape(y_test)[0]

    # Also, for ensemble builder, the OOF predictions should be there and match
    # the Ground truth that is also physically printed to disk
    ensemble_prediction = os.path.join(run_key_model_run_dir,
                                       estimator._backend.get_prediction_filename(
                                           'ensemble',
                                           estimator.seed, successful_num_run,
                                           run_key.budget))
    assert os.path.exists(ensemble_prediction), ensemble_prediction
    assert np.shape(np.load(ensemble_prediction, allow_pickle=True))[0] == np.shape(
        estimator._backend.load_targets_ensemble()
    )[0]

    # Ensemble Builder produced an ensemble
    estimator.ensemble_ is not None

    # There should be a weight for each element of the ensemble
    assert len(estimator.ensemble_.identifiers_) == len(estimator.ensemble_.weights_)

    X_test = backend.load_datamanager().generate_test_seqs()

    y_pred = estimator.predict(X_test)

    assert np.shape(y_pred) == np.shape(y_test)

    # Test refit on dummy data
    estimator.refit(dataset=backend.load_datamanager())
    # Make sure that a configuration space is stored in the estimator
    assert isinstance(estimator.get_search_space(), CS.ConfigurationSpace)


@pytest.mark.parametrize('openml_id', (
    1590,  # Adult to test NaN in categorical columns
))
def test_tabular_input_support(openml_id, backend):
    """
    Make sure we can process inputs with NaN in categorical and Object columns
    when the later is possible
    """

    # Get the data and check that contents of data-manager make sense
    X, y = sklearn.datasets.fetch_openml(
        data_id=int(openml_id),
        return_X_y=True, as_frame=True
    )

    # Make sure we are robust against objects
    X[X.columns[0]] = X[X.columns[0]].astype(object)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=1)
    # Search for a good configuration
    estimator = TabularClassificationTask(
        backend=backend,
        resampling_strategy=HoldoutValTypes.holdout_validation,
        ensemble_size=0,
    )

    estimator._do_dummy_prediction = unittest.mock.MagicMock()

    with unittest.mock.patch.object(AutoMLSMBO, 'run_smbo') as AutoMLSMBOMock:
        AutoMLSMBOMock.return_value = (RunHistory(), {}, 'epochs')
        estimator.search(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            optimize_metric='accuracy',
            total_walltime_limit=150,
            func_eval_time_limit_secs=50,
            enable_traditional_pipeline=False,
            load_models=False,
        )


@pytest.mark.parametrize("fit_dictionary_tabular", ['classification_categorical_only'], indirect=True)
def test_do_dummy_prediction(dask_client, fit_dictionary_tabular):
    backend = fit_dictionary_tabular['backend']
    estimator = TabularClassificationTask(
        backend=backend,
        resampling_strategy=HoldoutValTypes.holdout_validation,
        ensemble_size=0,
    )

    # Setup pre-requisites normally set by search()
    estimator._create_dask_client()
    estimator._metric = accuracy
    estimator._logger = estimator._get_logger('test')
    estimator._memory_limit = 5000
    estimator._time_for_task = 60
    estimator._disable_file_output = []
    estimator._all_supported_metrics = False

    with pytest.raises(ValueError, match=r".*Dummy prediction failed with run state.*"):
        with unittest.mock.patch('autoPyTorch.evaluation.tae.eval_train_function') as dummy:
            dummy.side_effect = MemoryError
            estimator._do_dummy_prediction()

    estimator._do_dummy_prediction()

    # Ensure that the dummy predictions are not in the current working
    # directory, but in the temporary directory.
    assert not os.path.exists(os.path.join(os.getcwd(), '.autoPyTorch'))
    assert os.path.exists(os.path.join(
        backend.temporary_directory, '.autoPyTorch', 'runs', '1_1_50.0',
        'predictions_ensemble_1_1_50.0.npy')
    )

    model_path = os.path.join(backend.temporary_directory,
                              '.autoPyTorch',
                              'runs', '1_1_50.0',
                              '1.1.50.0.model')

    # Make sure the dummy model complies with scikit learn
    # get/set params
    assert os.path.exists(model_path)
    with open(model_path, 'rb') as model_handler:
        clone(pickle.load(model_handler))

    estimator._close_dask_client()
    estimator._clean_logger()

    del estimator


@unittest.mock.patch('autoPyTorch.evaluation.train_evaluator.eval_train_function',
                     new=dummy_eval_train_function)
@pytest.mark.parametrize('openml_id', (40981, ))
def test_portfolio_selection(openml_id, backend, n_samples):

    # Get the data and check that contents of data-manager make sense
    X, y = sklearn.datasets.fetch_openml(
        data_id=int(openml_id),
        return_X_y=True, as_frame=True
    )
    X, y = X.iloc[:n_samples], y.iloc[:n_samples]

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=1)

    # Search for a good configuration
    estimator = TabularClassificationTask(
        backend=backend,
        resampling_strategy=HoldoutValTypes.holdout_validation,
    )

    with unittest.mock.patch.object(estimator, '_do_dummy_prediction', new=dummy_do_dummy_prediction):
        estimator.search(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            optimize_metric='accuracy',
            total_walltime_limit=30,
            func_eval_time_limit_secs=5,
            enable_traditional_pipeline=False,
            portfolio_selection=os.path.join(os.path.dirname(__file__),
                                             "../../autoPyTorch/configs/greedy_portfolio.json")
        )

    successful_config_ids = [run_key.config_id for run_key, run_value in estimator.run_history.data.items(
    ) if 'SUCCESS' in str(run_value.status)]
    successful_configs = [estimator.run_history.ids_config[id].get_dictionary() for id in successful_config_ids]
    portfolio_configs = json.load(open(os.path.join(os.path.dirname(__file__),
                                                    "../../autoPyTorch/configs/greedy_portfolio.json")))
    # check if any configs from greedy portfolio were compatible with australian
    assert any(successful_config in portfolio_configs for successful_config in successful_configs)


@unittest.mock.patch('autoPyTorch.evaluation.train_evaluator.eval_train_function',
                     new=dummy_eval_train_function)
@pytest.mark.parametrize('openml_id', (40981, ))
def test_portfolio_selection_failure(openml_id, backend, n_samples):

    # Get the data and check that contents of data-manager make sense
    X, y = sklearn.datasets.fetch_openml(
        data_id=int(openml_id),
        return_X_y=True, as_frame=True
    )
    X, y = X.iloc[:n_samples], y.iloc[:n_samples]

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=1)

    estimator = TabularClassificationTask(
        backend=backend,
        resampling_strategy=HoldoutValTypes.holdout_validation,
    )
    with pytest.raises(FileNotFoundError, match=r"The path: .+? provided for 'portfolio_selection' "
                                                r"for the file containing the portfolio configurations "
                                                r"does not exist\. Please provide a valid path"):
        estimator.search(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            optimize_metric='accuracy',
            total_walltime_limit=30,
            func_eval_time_limit_secs=5,
            enable_traditional_pipeline=False,
            portfolio_selection="random_path_to_test.json"
        )


# TODO: Make faster when https://github.com/automl/Auto-PyTorch/pull/223 is incorporated
@pytest.mark.parametrize("fit_dictionary_tabular", ['classification_categorical_only'], indirect=True)
def test_do_traditional_pipeline(fit_dictionary_tabular):
    backend = fit_dictionary_tabular['backend']
    estimator = TabularClassificationTask(
        backend=backend,
        resampling_strategy=HoldoutValTypes.holdout_validation,
        ensemble_size=0,
    )

    # Setup pre-requisites normally set by search()
    estimator._create_dask_client()
    estimator._metric = accuracy
    estimator._logger = estimator._get_logger('test')
    estimator._memory_limit = 5000
    estimator._time_for_task = 60
    estimator._disable_file_output = []
    estimator._all_supported_metrics = False

    estimator._do_traditional_prediction(time_left=60, func_eval_time_limit_secs=30)

    # The models should not be on the current directory
    assert not os.path.exists(os.path.join(os.getcwd(), '.autoPyTorch'))

    # Then we should have fitted 5 classifiers
    # Maybe some of them fail (unlikely, but we do not control external API)
    # but we want to make this test robust
    at_least_one_model_checked = False
    for i in range(2, 7):
        pred_path = os.path.join(
            backend.temporary_directory, '.autoPyTorch', 'runs', f"1_{i}_50.0",
            f"predictions_ensemble_1_{i}_50.0.npy"
        )
        if not os.path.exists(pred_path):
            continue

        model_path = os.path.join(backend.temporary_directory,
                                  '.autoPyTorch',
                                  'runs', f"1_{i}_50.0",
                                  f"1.{i}.50.0.model")

        # Make sure the dummy model complies with scikit learn
        # get/set params
        assert os.path.exists(model_path)
        with open(model_path, 'rb') as model_handler:
            model = pickle.load(model_handler)
        clone(model)
        assert model.config == list(_traditional_learners.keys())[i - 2]
        at_least_one_model_checked = True
    if not at_least_one_model_checked:
        pytest.fail("Not even one single traditional pipeline was fitted")

    estimator._close_dask_client()
    estimator._clean_logger()

    del estimator


@pytest.mark.parametrize("api_type", [TabularClassificationTask, TabularRegressionTask])
def test_unsupported_msg(api_type):
    api = api_type()
    with pytest.raises(ValueError, match=r".*is only supported after calling search. Kindly .*"):
        api.predict(np.ones((10, 10)))


@pytest.mark.parametrize("fit_dictionary_tabular", ['classification_categorical_only'], indirect=True)
@pytest.mark.parametrize("api_type", [TabularClassificationTask, TabularRegressionTask])
def test_build_pipeline(api_type, fit_dictionary_tabular):
    api = api_type()
    pipeline = api.build_pipeline(fit_dictionary_tabular['dataset_properties'])
    assert isinstance(pipeline, BaseEstimator)
    assert len(pipeline.steps) > 0


@pytest.mark.parametrize("disable_file_output", [['all'], None])
@pytest.mark.parametrize('openml_id', (40984,))
@pytest.mark.parametrize('resampling_strategy,resampling_strategy_args',
                         ((HoldoutValTypes.holdout_validation, {'val_share': 0.8}),
                          (CrossValTypes.k_fold_cross_validation, {'num_splits': 2}),
                          (NoResamplingStrategyTypes.no_resampling, {})
                          )
                         )
@pytest.mark.parametrize("budget", [15, 20])
def test_pipeline_fit(openml_id,
                      resampling_strategy,
                      resampling_strategy_args,
                      backend,
                      disable_file_output,
                      budget,
                      n_samples):
    # Get the data and check that contents of data-manager make sense
    X, y = sklearn.datasets.fetch_openml(
        data_id=int(openml_id),
        return_X_y=True, as_frame=True
    )
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X[:n_samples], y[:n_samples], random_state=1)

    # Search for a good configuration
    estimator = TabularClassificationTask(
        backend=backend,
        resampling_strategy=resampling_strategy,
        ensemble_size=0
    )

    dataset = estimator.get_dataset(X_train=X_train,
                                    y_train=y_train,
                                    X_test=X_test,
                                    y_test=y_test,
                                    resampling_strategy=resampling_strategy,
                                    resampling_strategy_args=resampling_strategy_args)

    configuration = estimator.get_search_space(dataset).get_default_configuration()
    pipeline, run_info, run_value, dataset = estimator.fit_pipeline(dataset=dataset,
                                                                    configuration=configuration,
                                                                    run_time_limit_secs=50,
                                                                    disable_file_output=disable_file_output,
                                                                    budget_type='epochs',
                                                                    budget=budget
                                                                    )
    assert isinstance(dataset, BaseDataset)
    assert isinstance(run_info, RunInfo)
    assert isinstance(run_info.config, Configuration)

    assert isinstance(run_value, RunValue)
    assert 'SUCCESS' in str(run_value.status)

    if disable_file_output is None:
        if resampling_strategy in CrossValTypes:
            assert isinstance(pipeline, BaseEstimator)
            X_test = dataset.test_tensors[0]
            preds = pipeline.predict_proba(X_test)
            assert isinstance(preds, np.ndarray)

            score = accuracy(dataset.test_tensors[1], preds)
            assert isinstance(score, float)
            assert score > 0.65
        else:
            assert isinstance(pipeline, BasePipeline)
            # To make sure we fitted the model, there should be a
            # run summary object with accuracy
            run_summary = pipeline.named_steps['trainer'].run_summary
            assert run_summary is not None
            X_test = dataset.test_tensors[0]
            preds = pipeline.predict(X_test)
            assert isinstance(preds, np.ndarray)

            score = accuracy(dataset.test_tensors[1], preds)
            assert isinstance(score, float)
            assert score > 0.65
    else:
        assert pipeline is None
        assert run_value.cost < 0.35

    # Make sure that the pipeline can be pickled
    dump_file = os.path.join(tempfile.gettempdir(), 'automl.dump.pkl')
    with open(dump_file, 'wb') as f:
        pickle.dump(pipeline, f)

    num_run_dir = estimator._backend.get_numrun_directory(
        run_info.seed, run_value.additional_info['num_run'], budget=float(budget))

    cv_model_path = os.path.join(num_run_dir, estimator._backend.get_cv_model_filename(
        run_info.seed, run_value.additional_info['num_run'], budget=float(budget)))
    model_path = os.path.join(num_run_dir, estimator._backend.get_model_filename(
        run_info.seed, run_value.additional_info['num_run'], budget=float(budget)))

    if disable_file_output:
        # No file output is expected
        assert not os.path.exists(num_run_dir)
    else:
        # We expect the model path always
        # And the cv model only on 'cv'
        assert os.path.exists(model_path)
        if resampling_strategy in CrossValTypes:
            assert os.path.exists(cv_model_path)
        elif resampling_strategy in HoldoutValTypes:
            assert not os.path.exists(cv_model_path)


@pytest.mark.parametrize('openml_id', (40984,))
@pytest.mark.parametrize('resampling_strategy,resampling_strategy_args',
                         ((HoldoutValTypes.holdout_validation, {'val_share': 0.8}),
                          )
                         )
def test_pipeline_fit_error(
    openml_id,
    resampling_strategy,
    resampling_strategy_args,
    backend,
    n_samples
):
    # Get the data and check that contents of data-manager make sense
    X, y = sklearn.datasets.fetch_openml(
        data_id=int(openml_id),
        return_X_y=True, as_frame=True
    )
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X[:n_samples], y[:n_samples], random_state=1)

    # Search for a good configuration
    estimator = TabularClassificationTask(
        backend=backend,
        resampling_strategy=resampling_strategy,
    )

    dataset = estimator.get_dataset(X_train=X_train,
                                    y_train=y_train,
                                    X_test=X_test,
                                    y_test=y_test,
                                    resampling_strategy=resampling_strategy,
                                    resampling_strategy_args=resampling_strategy_args)

    configuration = estimator.get_search_space(dataset).get_default_configuration()
    pipeline, run_info, run_value, dataset = estimator.fit_pipeline(dataset=dataset,
                                                                    configuration=configuration,
                                                                    run_time_limit_secs=7,
                                                                    )

    assert 'TIMEOUT' in str(run_value.status)
    assert pipeline is None


@pytest.mark.parametrize('openml_id', (40981, ))
def test_tabular_classification_test_evaluator(openml_id, backend, n_samples):

    # Get the data and check that contents of data-manager make sense
    X, y = sklearn.datasets.fetch_openml(
        data_id=int(openml_id),
        return_X_y=True, as_frame=True
    )
    X, y = X.iloc[:n_samples], y.iloc[:n_samples]

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=42)

    # Search for a good configuration
    estimator = TabularClassificationTask(
        backend=backend,
        resampling_strategy=NoResamplingStrategyTypes.no_resampling,
        seed=42,
        ensemble_size=0
    )

    with unittest.mock.patch.object(estimator, '_do_dummy_prediction', new=dummy_do_dummy_prediction):
        estimator.search(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            optimize_metric='accuracy',
            total_walltime_limit=50,
            func_eval_time_limit_secs=20,
            enable_traditional_pipeline=False,
        )

    # Internal dataset has expected settings
    assert estimator.dataset.task_type == 'tabular_classification'

    assert estimator.resampling_strategy == NoResamplingStrategyTypes.no_resampling
    assert estimator.dataset.resampling_strategy == NoResamplingStrategyTypes.no_resampling
    # Check for the created files
    tmp_dir = estimator._backend.temporary_directory
    loaded_datamanager = estimator._backend.load_datamanager()
    assert len(loaded_datamanager.train_tensors) == len(estimator.dataset.train_tensors)

    expected_files = [
        'smac3-output/run_42/configspace.json',
        'smac3-output/run_42/runhistory.json',
        'smac3-output/run_42/scenario.txt',
        'smac3-output/run_42/stats.json',
        'smac3-output/run_42/train_insts.txt',
        'smac3-output/run_42/trajectory.json',
        '.autoPyTorch/datamanager.pkl',
        '.autoPyTorch/start_time_42',
    ]
    for expected_file in expected_files:
        assert os.path.exists(os.path.join(tmp_dir, expected_file)), "{}/{}/{}".format(
            tmp_dir,
            [data for data in pathlib.Path(tmp_dir).glob('*')],
            expected_file,
        )

    # Check that smac was able to find proper models
    succesful_runs = [run_value.status for run_value in estimator.run_history.data.values(
    ) if 'SUCCESS' in str(run_value.status)]
    assert len(succesful_runs) > 1, [(k, v) for k, v in estimator.run_history.data.items()]

    # Search for an existing run key in disc. A individual model might have
    # a timeout and hence was not written to disc
    successful_num_run = None
    SUCCESS = False
    for i, (run_key, value) in enumerate(estimator.run_history.data.items()):
        if 'SUCCESS' in str(value.status):
            run_key_model_run_dir = estimator._backend.get_numrun_directory(
                estimator.seed, run_key.config_id + 1, run_key.budget)
            successful_num_run = run_key.config_id + 1
            if os.path.exists(run_key_model_run_dir):
                # Runkey config id is different from the num_run
                # more specifically num_run = config_id + 1(dummy)
                SUCCESS = True
                break

    assert SUCCESS, f"Successful run was not properly saved for num_run: {successful_num_run}"

    model_file = os.path.join(run_key_model_run_dir,
                              f"{estimator.seed}.{successful_num_run}.{run_key.budget}.model")
    assert os.path.exists(model_file), model_file

    # Make sure that predictions on the test data are printed and make sense
    test_prediction = os.path.join(run_key_model_run_dir,
                                   estimator._backend.get_prediction_filename(
                                       'test', estimator.seed, successful_num_run,
                                       run_key.budget))
    assert os.path.exists(test_prediction), test_prediction
    assert np.shape(np.load(test_prediction, allow_pickle=True))[0] == np.shape(X_test)[0]

    y_pred = estimator.predict(X_test)
    assert np.shape(y_pred)[0] == np.shape(X_test)[0]

    # Make sure that predict proba has the expected shape
    probabilites = estimator.predict_proba(X_test)
    assert np.shape(probabilites) == (np.shape(X_test)[0], 2)

    score = estimator.score(y_pred, y_test)
    assert 'accuracy' in score

    # check incumbent config and results
    incumbent_config, incumbent_results = estimator.get_incumbent_results()
    assert isinstance(incumbent_config, Configuration)
    assert isinstance(incumbent_results, dict)
    assert 'opt_loss' in incumbent_results, "run history: {}, successful_num_run: {}".format(estimator.run_history.data,
                                                                                             successful_num_run)
    assert 'train_loss' in incumbent_results


@pytest.mark.parametrize("ans,task_class", (
    ("continuous", TabularRegressionTask),
    ("multiclass", TabularClassificationTask))
)
def test_task_inference(ans, task_class, backend):
    # Get the data and check that contents of data-manager make sense
    X = np.random.random((6, 1))
    y = np.array([-10 ** 12, 0, 1, 2, 3, 4], dtype=np.int64) + 10 ** 12

    estimator = task_class(
        backend=backend,
        resampling_strategy=HoldoutValTypes.holdout_validation,
        resampling_strategy_args=None,
        seed=42,
    )
    dataset = estimator.get_dataset(X, y)
    assert dataset.output_type == ans

    y += 10 ** 12 + 10  # Check if the function catches overflow possibilities
    if ans == 'continuous':
        with pytest.raises(ValueError):  # ValueError due to `Too large value`
            estimator.get_dataset(X, y)
    else:
        estimator.get_dataset(X, y)
