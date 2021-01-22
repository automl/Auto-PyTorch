import copy
import unittest.mock

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

from sklearn.base import clone

import torch

import autoPyTorch.pipeline.components.setup.lr_scheduler.base_scheduler_choice as lr_components
import autoPyTorch.pipeline.components.setup.network.base_network_choice as network_components
import \
    autoPyTorch.pipeline.components.setup.network_initializer.base_network_init_choice as network_initializer_components  # noqa: E501
import autoPyTorch.pipeline.components.setup.optimizer.base_optimizer_choice as optimizer_components
from autoPyTorch.pipeline.components.setup.lr_scheduler.base_scheduler_choice import (
    BaseLRComponent,
    SchedulerChoice
)
from autoPyTorch.pipeline.components.setup.network.base_network_choice import (
    BaseNetworkComponent,
    NetworkChoice
)
from autoPyTorch.pipeline.components.setup.network_initializer.base_network_init_choice import (
    BaseNetworkInitializerComponent,
    NetworkInitializerChoice
)
from autoPyTorch.pipeline.components.setup.optimizer.base_optimizer_choice import (
    BaseOptimizerComponent,
    OptimizerChoice
)


class DummyLR(BaseLRComponent):
    def __init__(self, random_state=None):
        pass

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs

    def get_properties(dataset_properties=None):
        return {
            'shortname': 'Dummy',
            'name': 'Dummy',
        }


class DummyOptimizer(BaseOptimizerComponent):
    def __init__(self, random_state=None):
        pass

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs

    def get_properties(dataset_properties=None):
        return {
            'shortname': 'Dummy',
            'name': 'Dummy',
        }


class DummyNet(BaseNetworkComponent):
    def __init__(self, random_state=None):
        pass

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs

    def get_properties(dataset_properties=None):
        return {
            'shortname': 'Dummy',
            'name': 'Dummy',
        }


class DummyNetworkInitializer(BaseNetworkInitializerComponent):
    def __init__(self, random_state=None):
        pass

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs

    def get_properties(dataset_properties=None):
        return {
            'shortname': 'Dummy',
            'name': 'Dummy',
        }


class SchedulerTest(unittest.TestCase):
    def test_every_scheduler_is_valid(self):
        """
        Makes sure that every scheduler is a valid estimator.
        That is, we can fully create an object via get/set params.

        This also test that we can properly initialize each one
        of them
        """
        scheduler_choice = SchedulerChoice(dataset_properties={})

        # Make sure all components are returned
        self.assertEqual(len(scheduler_choice.get_components().keys()), 7)

        # For every scheduler in the components, make sure
        # that it complies with the scikit learn estimator.
        # This is important because usually components are forked to workers,
        # so the set/get params methods should recreate the same object
        for name, scheduler in scheduler_choice.get_components().items():
            config = scheduler.get_hyperparameter_search_space().sample_configuration()
            estimator = scheduler(**config)
            estimator_clone = clone(estimator)
            estimator_clone_params = estimator_clone.get_params()

            # Make sure all keys are copied properly
            for k, v in estimator.get_params().items():
                self.assertIn(k, estimator_clone_params)

            # Make sure the params getter of estimator are honored
            klass = estimator.__class__
            new_object_params = estimator.get_params(deep=False)
            for name, param in new_object_params.items():
                new_object_params[name] = clone(param, safe=False)
            new_object = klass(**new_object_params)
            params_set = new_object.get_params(deep=False)

            for name in new_object_params:
                param1 = new_object_params[name]
                param2 = params_set[name]
                self.assertEqual(param1, param2)

    def test_get_set_config_space(self):
        """Make sure that we can setup a valid choice in the scheduler
        choice"""
        scheduler_choice = SchedulerChoice(dataset_properties={})
        cs = scheduler_choice.get_hyperparameter_search_space()

        # Make sure that all hyperparameters are part of the serach space
        self.assertListEqual(
            sorted(cs.get_hyperparameter('__choice__').choices),
            sorted(list(scheduler_choice.get_components().keys()))
        )

        # Make sure we can properly set some random configs
        # Whereas just one iteration will make sure the algorithm works,
        # doing five iterations increase the confidence. We will be able to
        # catch component specific crashes
        for i in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            scheduler_choice.set_hyperparameters(config)

            self.assertEqual(scheduler_choice.choice.__class__,
                             scheduler_choice.get_components()[config_dict['__choice__']])

            # Then check the choice configuration
            selected_choice = config_dict.pop('__choice__', None)
            for key, value in config_dict.items():
                # Remove the selected_choice string from the parameter
                # so we can query in the object for it
                key = key.replace(selected_choice + ':', '')
                self.assertIn(key, vars(scheduler_choice.choice))
                self.assertEqual(value, scheduler_choice.choice.__dict__[key])

    def test_scheduler_add(self):
        """Makes sure that a component can be added to the CS"""
        # No third party components to start with
        self.assertEqual(len(lr_components._addons.components), 0)

        # Then make sure the scheduler can be added and query'ed
        lr_components.add_scheduler(DummyLR)
        self.assertEqual(len(lr_components._addons.components), 1)
        cs = SchedulerChoice(dataset_properties={}).get_hyperparameter_search_space()
        self.assertIn('DummyLR', str(cs))


class OptimizerTest(unittest.TestCase):
    def test_every_optimizer_is_valid(self):
        """
        Makes sure that every optimizer is a valid estimator.
        That is, we can fully create an object via get/set params.

        This also test that we can properly initialize each one
        of them
        """
        optimizer_choice = OptimizerChoice(dataset_properties={})

        # Make sure all components are returned
        self.assertEqual(len(optimizer_choice.get_components().keys()), 4)

        # For every optimizer in the components, make sure
        # that it complies with the scikit learn estimator.
        # This is important because usually components are forked to workers,
        # so the set/get params methods should recreate the same object
        for name, optimizer in optimizer_choice.get_components().items():
            config = optimizer.get_hyperparameter_search_space().sample_configuration()
            estimator = optimizer(**config)
            estimator_clone = clone(estimator)
            estimator_clone_params = estimator_clone.get_params()

            # Make sure all keys are copied properly
            for k, v in estimator.get_params().items():
                self.assertIn(k, estimator_clone_params)

            # Make sure the params getter of estimator are honored
            klass = estimator.__class__
            new_object_params = estimator.get_params(deep=False)
            for name, param in new_object_params.items():
                new_object_params[name] = clone(param, safe=False)
            new_object = klass(**new_object_params)
            params_set = new_object.get_params(deep=False)

            for name in new_object_params:
                param1 = new_object_params[name]
                param2 = params_set[name]
                self.assertEqual(param1, param2)

    def test_get_set_config_space(self):
        """Make sure that we can setup a valid choice in the optimizer
        choice"""
        optimizer_choice = OptimizerChoice(dataset_properties={})
        cs = optimizer_choice.get_hyperparameter_search_space()

        # Make sure that all hyperparameters are part of the serach space
        self.assertListEqual(
            sorted(cs.get_hyperparameter('__choice__').choices),
            sorted(list(optimizer_choice.get_components().keys()))
        )

        # Make sure we can properly set some random configs
        # Whereas just one iteration will make sure the algorithm works,
        # doing five iterations increase the confidence. We will be able to
        # catch component specific crashes
        for i in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            optimizer_choice.set_hyperparameters(config)

            self.assertEqual(optimizer_choice.choice.__class__,
                             optimizer_choice.get_components()[config_dict['__choice__']])

            # Then check the choice configuration
            selected_choice = config_dict.pop('__choice__', None)
            for key, value in config_dict.items():
                # Remove the selected_choice string from the parameter
                # so we can query in the object for it
                key = key.replace(selected_choice + ':', '')
                self.assertIn(key, vars(optimizer_choice.choice))
                self.assertEqual(value, optimizer_choice.choice.__dict__[key])

    def test_optimizer_add(self):
        """Makes sure that a component can be added to the CS"""
        # No third party components to start with
        self.assertEqual(len(optimizer_components._addons.components), 0)

        # Then make sure the optimizer can be added and query'ed
        optimizer_components.add_optimizer(DummyOptimizer)
        self.assertEqual(len(optimizer_components._addons.components), 1)
        cs = OptimizerChoice(dataset_properties={}).get_hyperparameter_search_space()
        self.assertIn('DummyOptimizer', str(cs))


class NetworkTest(unittest.TestCase):
    def test_every_network_is_valid(self):
        """
        Makes sure that every network is a valid estimator.
        That is, we can fully create an object via get/set params.

        This also test that we can properly initialize each one
        of them
        """
        network_choice = NetworkChoice(dataset_properties={})

        # Make sure all components are returned
        self.assertEqual(len(network_choice.get_components().keys()), 1)

        # For every network in the components, make sure
        # that it complies with the scikit learn estimator.
        # This is important because usually components are forked to workers,
        # so the set/get params methods should recreate the same object
        for name, network in network_choice.get_components().items():
            config = network.get_hyperparameter_search_space().sample_configuration()
            estimator = network(**config)
            estimator_clone = clone(estimator)
            estimator_clone_params = estimator_clone.get_params()

            # Make sure all keys are copied properly
            for k, v in estimator.get_params().items():
                self.assertIn(k, estimator_clone_params)

            # Make sure the params getter of estimator are honored
            klass = estimator.__class__
            new_object_params = estimator.get_params(deep=False)
            for name, param in new_object_params.items():
                new_object_params[name] = clone(param, safe=False)
            new_object = klass(**new_object_params)
            params_set = new_object.get_params(deep=False)

            for name in new_object_params:
                param1 = new_object_params[name]
                param2 = params_set[name]
                self.assertEqual(param1, param2)

    def test_backbone_head_net(self):
        network_choice = NetworkChoice(dataset_properties={})
        task_types = {"image_classification": ((1, 3, 64, 64), (5,)),
                      "image_regression": ((1, 3, 64, 64), (1,)),
                      "time_series_classification": ((1, 32, 6), (5,)),
                      "time_series_regression": ((1, 32, 6), (1,)),
                      "tabular_classification": ((1, 100,), (5,)),
                      "tabular_regression": ((1, 100), (1,))}

        device = torch.device("cpu")
        for task_type, (input_shape, output_shape) in task_types.items():
            cs = network_choice.get_hyperparameter_search_space(dataset_properties={"task_type": task_type},
                                                                include=["BackboneHeadNet"])
            # test 10 random configurations
            for i in range(10):
                config = cs.sample_configuration()
                network_choice.set_hyperparameters(config)
                network_choice.fit(X={"X_train": np.zeros(input_shape),
                                      "y_train": np.zeros(output_shape),
                                      'dataset_properties': {"task_type": task_type,
                                                             'input_shape': input_shape[1:],
                                                             "output_shape": output_shape,
                                                             "num_classes": output_shape[0]}}, y=None)
                self.assertNotEqual(network_choice.choice.network, None)
                network_choice.choice.to(device)
                dummy_input = torch.randn((2, *input_shape[1:]), dtype=torch.float)
                output = network_choice.choice.network(dummy_input)
                self.assertEqual(output.shape[1:], output_shape)

    def test_get_set_config_space(self):
        """Make sure that we can setup a valid choice in the network
        choice"""
        network_choice = NetworkChoice(dataset_properties={})
        cs = network_choice.get_hyperparameter_search_space()

        # Make sure that all hyperparameters are part of the search space
        self.assertListEqual(
            sorted(cs.get_hyperparameter('__choice__').choices),
            sorted(list(network_choice.get_components().keys()))
        )

        # Make sure we can properly set some random configs
        # Whereas just one iteration will make sure the algorithm works,
        # doing five iterations increase the confidence. We will be able to
        # catch component specific crashes
        for i in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            network_choice.set_hyperparameters(config)

            self.assertEqual(network_choice.choice.__class__,
                             network_choice.get_components()[config_dict['__choice__']])

            # Then check the choice configuration
            selected_choice = config_dict.pop('__choice__', None)
            self.assertNotEqual(selected_choice, None)
            for key, value in config_dict.items():
                # Remove the selected_choice string from the parameter
                # so we can query in the object for it

                key = key.replace(selected_choice + ':', '')
                # In the case of MLP, parameters are dynamic, so they exist in config
                parameters = vars(network_choice.choice)
                parameters.update(vars(network_choice.choice)['config'])
                self.assertIn(key, parameters)
                self.assertEqual(value, parameters[key])

    def test_network_add(self):
        """Makes sure that a component can be added to the CS"""
        # No third party components to start with
        self.assertEqual(len(network_components._addons.components), 0)

        # Then make sure the scheduler can be added and query'ed
        network_components.add_network(DummyNet)
        self.assertEqual(len(network_components._addons.components), 1)
        cs = NetworkChoice(dataset_properties={}).get_hyperparameter_search_space()
        self.assertIn('DummyNet', str(cs))


class NetworkInitializerTest(unittest.TestCase):
    def test_every_network_initializer_is_valid(self):
        """
        Makes sure that every network_initializer is a valid estimator.
        That is, we can fully create an object via get/set params.

        This also test that we can properly initialize each one
        of them
        """
        network_initializer_choice = NetworkInitializerChoice(dataset_properties={})

        # Make sure all components are returned
        self.assertEqual(len(network_initializer_choice.get_components().keys()), 5)

        # For every optimizer in the components, make sure
        # that it complies with the scikit learn estimator.
        # This is important because usually components are forked to workers,
        # so the set/get params methods should recreate the same object
        for name, network_initializer in network_initializer_choice.get_components().items():
            config = network_initializer.get_hyperparameter_search_space().sample_configuration()
            estimator = network_initializer(**config)
            estimator_clone = clone(estimator)
            estimator_clone_params = estimator_clone.get_params()

            # Make sure all keys are copied properly
            for k, v in estimator.get_params().items():
                self.assertIn(k, estimator_clone_params)

            # Make sure the params getter of estimator are honored
            klass = estimator.__class__
            new_object_params = estimator.get_params(deep=False)
            for name, param in new_object_params.items():
                new_object_params[name] = clone(param, safe=False)
            new_object = klass(**new_object_params)
            params_set = new_object.get_params(deep=False)

            for name in new_object_params:
                param1 = new_object_params[name]
                param2 = params_set[name]
                self.assertEqual(param1, param2)

    def test_get_set_config_space(self):
        """Make sure that we can setup a valid choice in the network_initializer
        choice"""
        network_initializer_choice = NetworkInitializerChoice(dataset_properties={})
        cs = network_initializer_choice.get_hyperparameter_search_space()

        # Make sure that all hyperparameters are part of the serach space
        self.assertListEqual(
            sorted(cs.get_hyperparameter('__choice__').choices),
            sorted(list(network_initializer_choice.get_components().keys()))
        )

        # Make sure we can properly set some random configs
        # Whereas just one iteration will make sure the algorithm works,
        # doing five iterations increase the confidence. We will be able to
        # catch component specific crashes
        for i in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            network_initializer_choice.set_hyperparameters(config)

            self.assertEqual(network_initializer_choice.choice.__class__,
                             network_initializer_choice.get_components()[config_dict['__choice__']])

            # Then check the choice configuration
            selected_choice = config_dict.pop('__choice__', None)
            for key, value in config_dict.items():
                # Remove the selected_choice string from the parameter
                # so we can query in the object for it
                key = key.replace(selected_choice + ':', '')
                self.assertIn(key, vars(network_initializer_choice.choice))
                self.assertEqual(value, network_initializer_choice.choice.__dict__[key])

    def test_network_initializer_add(self):
        """Makes sure that a component can be added to the CS"""
        # No third party components to start with
        self.assertEqual(len(network_initializer_components._addons.components), 0)

        # Then make sure the network_initializer can be added and query'ed
        network_initializer_components.add_network_initializer(DummyNetworkInitializer)
        self.assertEqual(len(network_initializer_components._addons.components), 1)
        cs = NetworkInitializerChoice(dataset_properties={}).get_hyperparameter_search_space()
        self.assertIn('DummyNetworkInitializer', str(cs))


if __name__ == '__main__':
    unittest.main()
