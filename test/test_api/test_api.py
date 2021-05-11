import os
import pathlib
import pickle
import sys
import time
import unittest

import numpy as np

import pandas as pd

import pytest


import sklearn
import sklearn.datasets
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier, VotingRegressor

from smac.runhistory.runhistory import RunHistory, StatusType

import torch

from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.api.tabular_regression import TabularRegressionTask
from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
    HoldoutValTypes,
)
from autoPyTorch.evaluation.abstract_evaluator import (
    DummyClassificationPipeline,
    DummyRegressionPipeline,
    fit_and_suppress_warnings
)
from autoPyTorch.evaluation.train_evaluator import TrainEvaluator
from autoPyTorch.optimizer.smbo import AutoMLSMBO
from autoPyTorch.pipeline.components.training.metrics.metrics import accuracy
from autoPyTorch.constants import REGRESSION_TASKS


# ========
# Fixtures
# ========
class DummyTrainEvaluator(TrainEvaluator):

    def _fit_and_predict(self, pipeline, fold: int, train_indices,
                         test_indices,
                         add_pipeline_to_self
                         ):

        if self.task_type in REGRESSION_TASKS:
            pipeline = DummyRegressionPipeline(config=1)
        else:
            pipeline = DummyClassificationPipeline(config=1)

        self.indices[fold] = ((train_indices, test_indices))

        X = {'train_indices': train_indices,
             'val_indices': test_indices,
             'split_id': fold,
             'num_run': self.num_run,
             **self.fit_dictionary}  # fit dictionary
        y = None
        fit_and_suppress_warnings(self.logger, pipeline, X, y)
        self.logger.info("Model fitted, now predicting")
        (
            Y_train_pred,
            Y_opt_pred,
            Y_valid_pred,
            Y_test_pred
        ) = self._predict(
            pipeline,
            train_indices=train_indices,
            test_indices=test_indices,
        )

        if add_pipeline_to_self:
            self.pipeline = pipeline
        else:
            self.pipelines[fold] = pipeline

        return Y_train_pred, Y_opt_pred, Y_valid_pred, Y_test_pred


# create closure for evaluating an algorithm
def dummy_eval_function(
        backend,
        queue,
        metric,
        budget: float,
        config,
        seed: int,
        output_y_hat_optimization: bool,
        num_run: int,
        include,
        exclude,
        disable_file_output,
        pipeline_config=None,
        budget_type=None,
        init_params=None,
        logger_port=None,
        all_supported_metrics=True,
        search_space_updates=None,
        instance: str = None,
) -> None:
    evaluator = TrainEvaluator(
        backend=backend,
        queue=queue,
        metric=metric,
        configuration=config,
        seed=seed,
        num_run=num_run,
        output_y_hat_optimization=output_y_hat_optimization,
        include=include,
        exclude=exclude,
        disable_file_output=disable_file_output,
        init_params=init_params,
        budget=budget,
        budget_type=budget_type,
        logger_port=logger_port,
        all_supported_metrics=all_supported_metrics,
        pipeline_config=pipeline_config,
        search_space_updates=search_space_updates
    )
    evaluator.fit_predict_and_loss()


def dummy_do_dummy_prediction():
    return


# Test
# ========
@unittest.mock.patch('autoPyTorch.evaluation.train_evaluator.eval_function',
                     new=dummy_eval_function)
@pytest.mark.parametrize('openml_id', (40981, ))
@pytest.mark.parametrize('resampling_strategy,resampling_strategy_args',
                         ((HoldoutValTypes.holdout_validation, None),
                          (CrossValTypes.k_fold_cross_validation, {'num_splits': 2})
                          ))
def test_tabular_classification(openml_id, resampling_strategy, backend, resampling_strategy_args, n_samples):

    # Get the data and check that contents of data-manager make sense
    X, y = sklearn.datasets.fetch_openml(
        data_id=int(openml_id),
        return_X_y=True, as_frame=True
    )
    X, y = X.iloc[:n_samples], y.iloc[:n_samples]

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=1)

    include = None
    # for python less than 3.7, learned entity embedding
    # is not able to be stored on disk (only on CI)
    if sys.version_info < (3, 7):
        include = {'network_embedding': ['NoEmbedding']}
    # Search for a good configuration
    estimator = TabularClassificationTask(
        backend=backend,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        include_components=include
    )

    with unittest.mock.patch.object(estimator, '_do_dummy_prediction', new=dummy_do_dummy_prediction):
        estimator.search(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            optimize_metric='accuracy',
            total_walltime_limit=40,
            func_eval_time_limit_secs=5,
            enable_traditional_pipeline=False,
        )

    # Internal dataset has expected settings
    assert estimator.dataset.task_type == 'tabular_classification'
    expected_num_splits = 1 if resampling_strategy == HoldoutValTypes.holdout_validation else 2
    assert estimator.resampling_strategy == resampling_strategy
    assert estimator.dataset.resampling_strategy == resampling_strategy
    assert len(estimator.dataset.splits) == expected_num_splits

    # TODO: check for budget

    # Check for the created files
    tmp_dir = estimator._backend.temporary_directory
    loaded_datamanager = estimator._backend.load_datamanager()
    assert len(loaded_datamanager.train_tensors) == len(estimator.dataset.train_tensors)

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
        assert len(model.estimators_) == 2
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
    assert 'accuracy' in score

    # Check that we can pickle
    # Test pickle
    # This can happen on python greater than 3.6
    # as older python do not control the state of the logger
    if sys.version_info >= (3, 7):
        dump_file = os.path.join(estimator._backend.temporary_directory, 'dump.pkl')

        with open(dump_file, 'wb') as f:
            pickle.dump(estimator, f)

        with open(dump_file, 'rb') as f:
            restored_estimator = pickle.load(f)
        restored_estimator.predict(X_test)


@pytest.mark.parametrize('openml_name', ("boston", ))
@unittest.mock.patch('autoPyTorch.evaluation.train_evaluator.eval_function',
                     new=dummy_eval_function)
@pytest.mark.parametrize('resampling_strategy,resampling_strategy_args',
                         ((HoldoutValTypes.holdout_validation, None),
                          (CrossValTypes.k_fold_cross_validation, {'num_splits': 2})
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

    include = None
    # for python less than 3.7, learned entity embedding
    # is not able to be stored on disk (only on CI)
    if sys.version_info < (3, 7):
        include = {'network_embedding': ['NoEmbedding']}
    # Search for a good configuration
    estimator = TabularRegressionTask(
        backend=backend,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        include_components=include
    )

    with unittest.mock.patch.object(estimator, '_do_dummy_prediction', new=dummy_do_dummy_prediction):
        estimator.search(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            optimize_metric='r2',
            total_walltime_limit=35,
            func_eval_time_limit_secs=5,
            enable_traditional_pipeline=False,
    )

    # Internal dataset has expected settings
    assert estimator.dataset.task_type == 'tabular_regression'
    expected_num_splits = 1 if resampling_strategy == HoldoutValTypes.holdout_validation else 2
    assert estimator.resampling_strategy == resampling_strategy
    assert estimator.dataset.resampling_strategy == resampling_strategy
    assert len(estimator.dataset.splits) == expected_num_splits

    # TODO: check for budget

    # Check for the created files
    tmp_dir = estimator._backend.temporary_directory
    loaded_datamanager = estimator._backend.load_datamanager()
    assert len(loaded_datamanager.train_tensors) == len(estimator.dataset.train_tensors)

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
        assert len(model.estimators_) == 2
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

    # Check that we can pickle
    # Test pickle
    # This can happen on python greater than 3.6
    # as older python do not control the state of the logger
    if sys.version_info >= (3, 7):
        dump_file = os.path.join(estimator._backend.temporary_directory, 'dump.pkl')

        with open(dump_file, 'wb') as f:
            pickle.dump(estimator, f)

        with open(dump_file, 'rb') as f:
            restored_estimator = pickle.load(f)
        restored_estimator.predict(X_test)


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
