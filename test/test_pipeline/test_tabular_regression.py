import os
import re
import unittest
import unittest.mock

from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

import pytest

import torch

from autoPyTorch.pipeline.components.setup.early_preprocessor.utils import get_preprocess_transforms
from autoPyTorch.pipeline.tabular_regression import TabularRegressionPipeline
from autoPyTorch.utils.common import FitRequirement
from autoPyTorch.utils.hyperparameter_search_space_update import (
    HyperparameterSearchSpaceUpdates,
    parse_hyperparameter_search_space_updates
)


@pytest.mark.parametrize("fit_dictionary_tabular", ["regression_numerical_only",
                                                    "regression_categorical_only",
                                                    "regression_numerical_and_categorical"
                                                    ], indirect=True)
class TestTabularRegression:
    def _assert_pipeline_search_space(self, pipeline, search_space_updates):
        config_space = pipeline.get_hyperparameter_search_space()
        for update in search_space_updates.updates:
            try:
                assert update.node_name + ':' + update.hyperparameter in config_space
                hyperparameter = config_space.get_hyperparameter(update.node_name + ':' + update.hyperparameter)
            except AssertionError:
                assert any(update.node_name + ':' + update.hyperparameter in name
                           for name in config_space.get_hyperparameter_names()), \
                    "Can't find hyperparameter: {}".format(update.hyperparameter)
                # dimension reduction in embedding starts from 0
                if 'embedding' in update.node_name:
                    hyperparameter = config_space.get_hyperparameter(
                        update.node_name + ':' + update.hyperparameter + '_0')
                else:
                    hyperparameter = config_space.get_hyperparameter(
                        update.node_name + ':' + update.hyperparameter + '_1')
            assert update.default_value == hyperparameter.default_value
            if isinstance(hyperparameter, (UniformIntegerHyperparameter, UniformFloatHyperparameter)):
                assert update.value_range[0] == hyperparameter.lower
                assert update.value_range[1] == hyperparameter.upper
                if hasattr(update, 'log'):
                    assert update.log == hyperparameter.log
            elif isinstance(hyperparameter, CategoricalHyperparameter):
                assert update.value_range == hyperparameter.choices

    def test_pipeline_fit(self, fit_dictionary_tabular):
        """This test makes sure that the pipeline is able to fit
        given random combinations of hyperparameters across the pipeline"""
        pipeline = TabularRegressionPipeline(
            dataset_properties=fit_dictionary_tabular['dataset_properties'])
        cs = pipeline.get_hyperparameter_search_space()

        config = cs.sample_configuration()
        pipeline.set_hyperparameters(config)
        pipeline.fit(fit_dictionary_tabular)

        # To make sure we fitted the model, there should be a
        # run summary object with r2
        run_summary = pipeline.named_steps['trainer'].run_summary
        assert run_summary is not None

        # Make sure that performance was properly captured
        assert run_summary.performance_tracker['train_loss'][1] > 0
        assert run_summary.total_parameter_count > 0
        assert 'r2' in run_summary.performance_tracker['train_metrics'][1]

        # Make sure a network was fit
        assert isinstance(pipeline.named_steps['network'].get_network(), torch.nn.Module)

    def test_pipeline_predict(self, fit_dictionary_tabular):
        """This test makes sure that the pipeline is able to predict
        given a random configuration"""
        X = fit_dictionary_tabular['X_train'].copy()
        pipeline = TabularRegressionPipeline(
            dataset_properties=fit_dictionary_tabular['dataset_properties'])

        cs = pipeline.get_hyperparameter_search_space()
        config = cs.sample_configuration()
        pipeline.set_hyperparameters(config)

        with unittest.mock.patch.object(pipeline.named_steps['trainer'].choice, 'train_epoch') \
             as patch_train:
            patch_train.return_value = 1, {}
            pipeline.fit(fit_dictionary_tabular)

        # we expect the output to have the same batch size as the test input,
        # and number of outputs per batch sample equal to the number of targets ("output_shape" in dataset_properties)
        expected_output_shape = (X.shape[0], fit_dictionary_tabular["dataset_properties"]["output_shape"])

        prediction = pipeline.predict(X)
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == expected_output_shape

    def test_pipeline_transform(self, fit_dictionary_tabular):
        """
        In the context of autopytorch, transform expands a fit dictionary with
        components that where previously fit. We can use this as a nice way to make sure
        that fit properly work.
        This code is added in light of components not properly added to the fit dictionary
        """

        pipeline = TabularRegressionPipeline(
            dataset_properties=fit_dictionary_tabular['dataset_properties'])
        cs = pipeline.get_hyperparameter_search_space()
        config = cs.sample_configuration()
        pipeline.set_hyperparameters(config)

        with unittest.mock.patch.object(pipeline.named_steps['trainer'].choice, 'train_epoch') \
             as patch_train:
            patch_train.return_value = 1, {}
            # We do not want to make the same early preprocessing operation to the fit dictionary
            pipeline.fit(fit_dictionary_tabular.copy())

        transformed_fit_dictionary_tabular = pipeline.transform(fit_dictionary_tabular)

        # First, we do not lose anyone! (We use a fancy subset containment check)
        assert fit_dictionary_tabular.items() <= transformed_fit_dictionary_tabular.items()

        # Then the pipeline should have added the following keys
        expected_keys = {'imputer', 'encoder', 'scaler', 'tabular_transformer',
                         'preprocess_transforms', 'network', 'optimizer', 'lr_scheduler',
                         'train_data_loader', 'val_data_loader', 'run_summary'}
        assert expected_keys.issubset(set(transformed_fit_dictionary_tabular.keys()))

        # Then we need to have transformations being created.
        assert len(get_preprocess_transforms(transformed_fit_dictionary_tabular)) > 0

        # We expect the transformations to be in the pipeline at anytime for inference
        assert 'preprocess_transforms' in transformed_fit_dictionary_tabular.keys()

    @pytest.mark.parametrize("is_small_preprocess", [True, False])
    def test_default_configuration(self, fit_dictionary_tabular, is_small_preprocess):
        """Makes sure that when no config is set, we can trust the
        default configuration from the space"""

        fit_dictionary_tabular['is_small_preprocess'] = is_small_preprocess

        pipeline = TabularRegressionPipeline(
            dataset_properties=fit_dictionary_tabular['dataset_properties'])

        with unittest.mock.patch.object(pipeline.named_steps['trainer'].choice, 'train_epoch') \
                as patch_train:
            patch_train.return_value = 1, {}
            pipeline.fit(fit_dictionary_tabular)

    def test_remove_key_check_requirements(self, fit_dictionary_tabular):
        """Makes sure that when a key is removed from X, correct error is outputted"""
        pipeline = TabularRegressionPipeline(
            dataset_properties=fit_dictionary_tabular['dataset_properties'])
        for key in ['num_run', 'device', 'split_id', 'torch_num_threads', 'dataset_properties']:
            fit_dictionary_tabular_copy = fit_dictionary_tabular.copy()
            fit_dictionary_tabular_copy.pop(key)
            with pytest.raises(ValueError, match=r"To fit .+?, expected fit dictionary to have"):
                pipeline.fit(fit_dictionary_tabular_copy)

    def test_network_optimizer_lr_handshake(self, fit_dictionary_tabular):
        """Fitting a network should put the network in the X"""
        # Create the pipeline to check. A random config should be sufficient
        pipeline = TabularRegressionPipeline(
            dataset_properties=fit_dictionary_tabular['dataset_properties'])
        cs = pipeline.get_hyperparameter_search_space()
        config = cs.sample_configuration()
        pipeline.set_hyperparameters(config)

        # Make sure that fitting a network adds a "network" to X
        assert 'network' in pipeline.named_steps.keys()
        fit_dictionary_tabular['network_embedding'] = torch.nn.Linear(3, 3)
        fit_dictionary_tabular['network_backbone'] = torch.nn.Linear(3, 4)
        fit_dictionary_tabular['network_head'] = torch.nn.Linear(4, 1)
        X = pipeline.named_steps['network'].fit(
            fit_dictionary_tabular,
            None
        ).transform(fit_dictionary_tabular)
        assert 'network' in X

        # Then fitting a optimizer should fail if no network:
        assert 'optimizer' in pipeline.named_steps.keys()
        with pytest.raises(
            ValueError,
            match=r"To fit .+?, expected fit dictionary to have 'network' but got .*"
        ):
            pipeline.named_steps['optimizer'].fit({'dataset_properties': {}}, None)

        # No error when network is passed
        X = pipeline.named_steps['optimizer'].fit(X, None).transform(X)
        assert 'optimizer' in X

        # Then fitting a optimizer should fail if no network:
        assert 'lr_scheduler' in pipeline.named_steps.keys()
        with pytest.raises(
            ValueError,
            match=r"To fit .+?, expected fit dictionary to have 'optimizer' but got .*"
        ):
            pipeline.named_steps['lr_scheduler'].fit({'dataset_properties': {}}, None)

        # No error when network is passed
        X = pipeline.named_steps['lr_scheduler'].fit(X, None).transform(X)
        assert 'optimizer' in X

    def test_get_fit_requirements(self, fit_dictionary_tabular):
        dataset_properties = {'numerical_columns': [], 'categorical_columns': [],
                              'task_type': 'tabular_regression'}
        pipeline = TabularRegressionPipeline(dataset_properties=dataset_properties)
        fit_requirements = pipeline.get_fit_requirements()

        # check if fit requirements is a list of FitRequirement named tuples
        assert isinstance(fit_requirements, list)
        for requirement in fit_requirements:
            assert isinstance(requirement, FitRequirement)

    def test_apply_search_space_updates(self, fit_dictionary_tabular, search_space_updates):
        dataset_properties = {'numerical_columns': [1], 'categorical_columns': [2],
                              'task_type': 'tabular_regression'}
        pipeline = TabularRegressionPipeline(dataset_properties=dataset_properties,
                                             search_space_updates=search_space_updates)
        self._assert_pipeline_search_space(pipeline, search_space_updates)

    def test_read_and_update_search_space(self, fit_dictionary_tabular, search_space_updates):
        import tempfile
        path = tempfile.gettempdir()
        path = os.path.join(path, 'updates.txt')
        # Write to disk
        search_space_updates.save_as_file(path=path)
        assert os.path.exists(path=path)

        # Read from disk
        file_search_space_updates = parse_hyperparameter_search_space_updates(updates_file=path)
        assert isinstance(file_search_space_updates, HyperparameterSearchSpaceUpdates)
        dataset_properties = {'numerical_columns': [1], 'categorical_columns': [2],
                              'task_type': 'tabular_regression'}
        pipeline = TabularRegressionPipeline(dataset_properties=dataset_properties,
                                             search_space_updates=file_search_space_updates)
        assert file_search_space_updates == pipeline.search_space_updates

    def test_error_search_space_updates(self, fit_dictionary_tabular, error_search_space_updates):
        dataset_properties = {'numerical_columns': [1], 'categorical_columns': [2],
                              'task_type': 'tabular_regression'}
        try:
            _ = TabularRegressionPipeline(dataset_properties=dataset_properties,
                                          search_space_updates=error_search_space_updates)
        except Exception as e:
            assert isinstance(e, ValueError)
            assert re.match(r'Unknown hyperparameter for component .*?\. Expected update '
                            r'hyperparameter to be in \[.*?\] got .+', e.args[0])

    def test_set_range_search_space_updates(self, fit_dictionary_tabular):
        dataset_properties = {'numerical_columns': [1], 'categorical_columns': [2],
                              'task_type': 'tabular_regression'}
        config_dict = TabularRegressionPipeline(dataset_properties=dataset_properties). \
            get_hyperparameter_search_space()._hyperparameters
        updates = HyperparameterSearchSpaceUpdates()
        for i, (name, hyperparameter) in enumerate(config_dict.items()):
            if '__choice__' in name:
                continue
            name = name.split(':')
            hyperparameter_name = ':'.join(name[1:])
            if '_' in hyperparameter_name:
                if any(l_.isnumeric() for l_ in hyperparameter_name.split('_')[-1]) and 'network' in name[0]:
                    hyperparameter_name = '_'.join(hyperparameter_name.split('_')[:-1])
            if isinstance(hyperparameter, CategoricalHyperparameter):
                value_range = (hyperparameter.choices[0],)
                default_value = hyperparameter.choices[0]
            else:
                value_range = (0, 1)
                default_value = 1
            updates.append(node_name=name[0], hyperparameter=hyperparameter_name,
                           value_range=value_range, default_value=default_value)
        pipeline = TabularRegressionPipeline(dataset_properties=dataset_properties,
                                             search_space_updates=updates)

        try:
            self._assert_pipeline_search_space(pipeline, updates)
        except AssertionError as e:
            # As we are setting num_layers to 1 for fully connected
            # head, units_layer does not exist in the configspace
            assert 'fully_connected:units_layer' in e.args[0]


@pytest.mark.parametrize("fit_dictionary_tabular_dummy", ['regression'], indirect=True)
def test_pipeline_score(fit_dictionary_tabular_dummy):
    """This test makes sure that the pipeline is able to achieve a decent score on dummy data
    given the default configuration"""
    # increase number of epochs to test for performance
    fit_dictionary_tabular_dummy['epochs'] = 50
    fit_dictionary_tabular_dummy['early_stopping'] = 30

    X = fit_dictionary_tabular_dummy['X_train'].copy()
    y = fit_dictionary_tabular_dummy['y_train'].copy()

    pipeline = TabularRegressionPipeline(
        dataset_properties=fit_dictionary_tabular_dummy['dataset_properties'],
        random_state=2
    )

    cs = pipeline.get_hyperparameter_search_space()
    config = cs.get_default_configuration()
    pipeline.set_hyperparameters(config)

    pipeline.fit(fit_dictionary_tabular_dummy)

    # Ensure that the network is an instance of torch Module
    assert isinstance(pipeline.named_steps['network'].get_network(), torch.nn.Module)

    r2_score = pipeline.score(X, y)

    # we should be able to get a decent score on this dummy data
    assert r2_score >= 0.8, f"Pipeline:{pipeline} Config:{config} FitDict: {fit_dictionary_tabular_dummy}, " \
                            f"{pipeline.named_steps['trainer'].run_summary.performance_tracker['train_metrics']}"
