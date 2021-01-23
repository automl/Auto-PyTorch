import copy
import unittest.mock

from ConfigSpace.configuration_space import ConfigurationSpace

from sklearn.base import clone

import autoPyTorch.pipeline.components.setup.lr_scheduler.base_scheduler_choice as lr_components
import \
    autoPyTorch.pipeline.components.setup.network_initializer.base_network_init_choice as network_initializer_components  # noqa: E501
import autoPyTorch.pipeline.components.setup.optimizer.base_optimizer_choice as optimizer_components
from autoPyTorch.pipeline.components.setup.lr_scheduler.base_scheduler_choice import (
    BaseLRComponent,
    SchedulerChoice
)
from autoPyTorch.pipeline.components.setup.network_head.base_network_head_choice import (
    NetworkHeadChoice,
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


class NetworkHeadTest(unittest.TestCase):
    def test_every_networkHead_is_valid(self):
        """
        Makes sure that every network is a valid estimator.
        That is, we can fully create an object via get/set params.

        This also test that we can properly initialize each one
        of them
        """
        networkHead_choice = NetworkHeadChoice(dataset_properties={'task_type': 'tabular_classification'})

        # Make sure all components are returned
        self.assertEqual(len(networkHead_choice.get_components().keys()), 2)

        # For every network in the components, make sure
        # that it complies with the scikit learn estimator.
        # This is important because usually components are forked to workers,
        # so the set/get params methods should recreate the same object
        for name, networkHead in networkHead_choice.get_components().items():
            config = networkHead.get_hyperparameter_search_space().sample_configuration()
            estimator = networkHead(**config)
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
        """Make sure that we can setup a valid choice in the networkHead
        choice"""
        networkHead_choice = NetworkHeadChoice(dataset_properties={'task_type': 'tabular_classification'})
        cs = networkHead_choice.get_hyperparameter_search_space(
            dataset_properties={"task_type": 'tabular_classification'})

        # Make sure that all hyperparameters are part of the search space
        self.assertListEqual(
            sorted(cs.get_hyperparameter('__choice__').choices),
            ['fully_connected']
        )

        # Make sure we can properly set some random configs
        # Whereas just one iteration will make sure the algorithm works,
        # doing five iterations increase the confidence. We will be able to
        # catch component specific crashes
        for i in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            networkHead_choice.set_hyperparameters(config)

            self.assertEqual(networkHead_choice.choice.__class__,
                             networkHead_choice.get_components()[config_dict['__choice__']])

            # Then check the choice configuration
            selected_choice = config_dict.pop('__choice__', None)
            self.assertNotEqual(selected_choice, None)
            for key, value in config_dict.items():
                # Remove the selected_choice string from the parameter
                # so we can query in the object for it

                key = key.replace(selected_choice + ':', '')
                # In the case of MLP, parameters are dynamic, so they exist in config
                parameters = vars(networkHead_choice.choice)
                parameters.update(vars(networkHead_choice.choice)['config'])
                self.assertIn(key, parameters)
                self.assertEqual(value, parameters[key])


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
