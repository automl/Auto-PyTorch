import json
import os
import pathlib
import pickle
import sys
import unittest
from test.test_api.utils import dummy_do_dummy_prediction, dummy_eval_function, dummy_traditional_classification

from ConfigSpace.configuration_space import Configuration

import ConfigSpace as CS

import numpy as np

import pandas as pd

import pytest


import sklearn
import sklearn.datasets
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier, VotingRegressor

from smac.runhistory.runhistory import RunHistory

from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.api.tabular_regression import TabularRegressionTask
from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
    HoldoutValTypes,
)
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.optimizer.smbo import AutoMLSMBO
from autoPyTorch.pipeline.components.setup.traditional_ml.classifier_models import _classifiers
from autoPyTorch.pipeline.components.training.metrics.metrics import accuracy


CV_NUM_SPLITS = 2
HOLDOUT_NUM_SPLITS = 1


# Test
# ====
@unittest.mock.patch('autoPyTorch.evaluation.train_evaluator.eval_function',
                     new=dummy_eval_function)
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
        include_components=include,
        seed=42,
    )

    with unittest.mock.patch.object(estimator, '_do_dummy_prediction', new=dummy_do_dummy_prediction):
        estimator.search(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            optimize_metric='accuracy',
            total_walltime_limit=30,
            func_eval_time_limit_secs=5,
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

    # Test refit on dummy data
    estimator.refit(dataset=backend.load_datamanager())

    # Make sure that a configuration space is stored in the estimator
    assert isinstance(estimator.get_search_space(), CS.ConfigurationSpace)


@pytest.mark.parametrize('openml_name', ("boston", ))
@unittest.mock.patch('autoPyTorch.evaluation.train_evaluator.eval_function',
                     new=dummy_eval_function)
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
        include_components=include,
        seed=42,
    )

    with unittest.mock.patch.object(estimator, '_do_dummy_prediction', new=dummy_do_dummy_prediction):
        estimator.search(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            optimize_metric='r2',
            total_walltime_limit=30,
            func_eval_time_limit_secs=5,
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

    original_memory_limit = estimator._memory_limit
    estimator._memory_limit = 500
    with pytest.raises(ValueError, match=r".*Dummy prediction failed with run state.*"):
        estimator._do_dummy_prediction()

    estimator._memory_limit = original_memory_limit
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


@unittest.mock.patch('autoPyTorch.evaluation.train_evaluator.eval_function',
                     new=dummy_eval_function)
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

    include = None
    # for python less than 3.7, learned entity embedding
    # is not able to be stored on disk (only on CI)
    if sys.version_info < (3, 7):
        include = {'network_embedding': ['NoEmbedding']}
    # Search for a good configuration
    estimator = TabularClassificationTask(
        backend=backend,
        resampling_strategy=HoldoutValTypes.holdout_validation,
        include_components=include
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


@unittest.mock.patch('autoPyTorch.evaluation.train_evaluator.eval_function',
                     new=dummy_eval_function)
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

    include = None
    # for python less than 3.7, learned entity embedding
    # is not able to be stored on disk (only on CI)
    if sys.version_info < (3, 7):
        include = {'network_embedding': ['NoEmbedding']}
        # Search for a good configuration
    estimator = TabularClassificationTask(
        backend=backend,
        resampling_strategy=HoldoutValTypes.holdout_validation,
        include_components=include
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


@pytest.mark.parametrize('dataset_name', ('iris',))
@pytest.mark.parametrize('include_traditional', (True, False))
def test_get_incumbent_results(dataset_name, backend, include_traditional):
    # Get the data and check that contents of data-manager make sense
    X, y = sklearn.datasets.fetch_openml(
        name=dataset_name,
        return_X_y=True, as_frame=True
    )

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=1)

    # Search for a good configuration
    estimator = TabularClassificationTask(
        backend=backend,
        resampling_strategy=HoldoutValTypes.holdout_validation,
    )

    InputValidator = TabularInputValidator(
        is_classification=True,
    )

    # Fit a input validator to check the provided data
    # Also, an encoder is fit to both train and test data,
    # to prevent unseen categories during inference
    InputValidator.fit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    dataset = TabularDataset(
        X=X_train, Y=y_train,
        X_test=X_test, Y_test=y_test,
        validator=InputValidator,
        resampling_strategy=estimator.resampling_strategy,
        resampling_strategy_args=estimator.resampling_strategy_args,
    )

    pipeline_run_history = RunHistory()
    pipeline_run_history.load_json(os.path.join(os.path.dirname(__file__), '.tmp_api/runhistory.json'),
                                   estimator.get_search_space(dataset))

    estimator._do_dummy_prediction = unittest.mock.MagicMock()

    with unittest.mock.patch.object(AutoMLSMBO, 'run_smbo') as AutoMLSMBOMock:
        with unittest.mock.patch.object(TabularClassificationTask, '_do_traditional_prediction',
                                        new=dummy_traditional_classification):
            AutoMLSMBOMock.return_value = (pipeline_run_history, {}, 'epochs')
            estimator.search(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                optimize_metric='accuracy',
                total_walltime_limit=150,
                func_eval_time_limit_secs=50,
                enable_traditional_pipeline=True,
                load_models=False,
            )
    config, results = estimator.get_incumbent_results(include_traditional=include_traditional)
    assert isinstance(config, Configuration)
    assert isinstance(results, dict)

    run_history_data = estimator.run_history.data
    costs = [run_value.cost for run_key, run_value in run_history_data.items() if run_value.additional_info is not None
             and (run_value.additional_info['configuration_origin'] != 'traditional' or include_traditional)]
    assert results['opt_loss']['accuracy'] == min(costs)

    if not include_traditional:
        assert results['configuration_origin'] != 'traditional'


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
        assert model.config == list(_classifiers.keys())[i - 2]
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
