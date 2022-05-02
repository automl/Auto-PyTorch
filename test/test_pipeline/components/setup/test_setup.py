import copy
from typing import Any, Dict, Optional, Tuple

from ConfigSpace.configuration_space import ConfigurationSpace

import pytest

from sklearn.base import clone

import torch
from torch import nn

import autoPyTorch.pipeline.components.setup.lr_scheduler as lr_components
import autoPyTorch.pipeline.components.setup.network_backbone as base_network_backbone_choice
import autoPyTorch.pipeline.components.setup.network_head as base_network_head_choice
import autoPyTorch.pipeline.components.setup.network_initializer as network_initializer_components  # noqa: E501
import autoPyTorch.pipeline.components.setup.optimizer as optimizer_components
from autoPyTorch import constants
from autoPyTorch.pipeline.components.base_component import ThirdPartyComponents
from autoPyTorch.pipeline.components.setup.lr_scheduler import (
    BaseLRComponent,
    SchedulerChoice,
)
from autoPyTorch.pipeline.components.setup.lr_scheduler.constants import (
    StepIntervalUnit,
    StepIntervalUnitChoices
)
from autoPyTorch.pipeline.components.setup.network_backbone import NetworkBackboneChoice
from autoPyTorch.pipeline.components.setup.network_backbone.ResNetBackbone import ResBlock
from autoPyTorch.pipeline.components.setup.network_backbone.ShapedResNetBackbone import ShapedResNetBackbone
from autoPyTorch.pipeline.components.setup.network_backbone.base_network_backbone import NetworkBackboneComponent
from autoPyTorch.pipeline.components.setup.network_backbone.utils import get_shaped_neuron_counts
from autoPyTorch.pipeline.components.setup.network_head import NetworkHeadChoice
from autoPyTorch.pipeline.components.setup.network_head.base_network_head import NetworkHeadComponent
from autoPyTorch.pipeline.components.setup.network_initializer import (
    BaseNetworkInitializerComponent,
    NetworkInitializerChoice
)
from autoPyTorch.pipeline.components.setup.optimizer import (
    BaseOptimizerComponent,
    OptimizerChoice
)
from autoPyTorch.utils.hyperparameter_search_space_update import (
    HyperparameterSearchSpace,
    HyperparameterSearchSpaceUpdates
)


class DummyLR(BaseLRComponent):
    def __init__(self, step_interval: StepIntervalUnit, random_state=None):
        super().__init__(step_interval=step_interval)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'Dummy',
            'name': 'Dummy',
        }


class DummyBackbone(NetworkBackboneComponent):
    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {"name": "DummyBackbone",
                "shortname": "DummyBackbone",
                "handles_tabular": True,
                "handles_image": True,
                "handles_time_series": True}

    def build_backbone(self, input_shape: Tuple[int, ...]) -> nn.Module:
        return nn.Identity()

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, str]] = None) -> ConfigurationSpace:
        return ConfigurationSpace()


class DummyHead(NetworkHeadComponent):
    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {"name": "DummyHead",
                "shortname": "DummyHead",
                "handles_tabular": True,
                "handles_image": True,
                "handles_time_series": True}

    def build_head(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> nn.Module:
        return nn.Identity()

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, str]] = None) -> ConfigurationSpace:
        return ConfigurationSpace()


class TestScheduler:
    def test_every_scheduler_is_valid(self):
        """
        Makes sure that every scheduler is a valid estimator.
        That is, we can fully create an object via get/set params.

        This also test that we can properly initialize each one
        of them
        """
        scheduler_choice = SchedulerChoice(dataset_properties={})

        # Make sure all components are returned
        assert len(scheduler_choice.get_components().keys()) == 7

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
            for k in estimator.get_params().keys():
                assert k in estimator_clone_params

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
                assert param1 == param2

    def test_get_set_config_space(self):
        """Make sure that we can setup a valid choice in the scheduler
        choice"""
        scheduler_choice = SchedulerChoice(dataset_properties={})
        cs = scheduler_choice.get_hyperparameter_search_space()

        # Make sure that all hyperparameters are part of the serach space
        assert sorted(cs.get_hyperparameter('__choice__').choices) == \
               sorted(list(scheduler_choice.get_components().keys()))

        # Make sure we can properly set some random configs
        # Whereas just one iteration will make sure the algorithm works,
        # doing five iterations increase the confidence. We will be able to
        # catch component specific crashes
        for _ in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            scheduler_choice.set_hyperparameters(config)

            assert scheduler_choice.choice.__class__ == \
                   scheduler_choice.get_components()[config_dict['__choice__']]

            # Then check the choice configuration
            selected_choice = config_dict.pop('__choice__', None)
            for key, value in config_dict.items():
                # Remove the selected_choice string from the parameter
                # so we can query in the object for it
                key = key.replace(selected_choice + ':', '')
                assert key in vars(scheduler_choice.choice)
                assert value == scheduler_choice.choice.__dict__[key]

    def test_scheduler_add(self):
        """Makes sure that a component can be added to the CS"""
        # No third party components to start with
        assert len(lr_components._addons.components) == 0

        # Then make sure the scheduler can be added and query'ed
        lr_components.add_scheduler(DummyLR)
        assert len(lr_components._addons.components) == 1
        cs = SchedulerChoice(dataset_properties={}).get_hyperparameter_search_space()
        assert 'DummyLR' in str(cs)

    def test_schduler_init(self):
        for step_interval in StepIntervalUnitChoices:
            DummyLR(step_interval=step_interval)

        for step_interval in ['Batch', 'foo']:
            try:
                DummyLR(step_interval=step_interval)
            except ValueError:
                pass
            except Exception as e:
                pytest.fail("The initialization of lr_scheduler raised an unexpected exception {}.".format(e))
            else:
                pytest.fail("The initialization of lr_scheduler did not raise an Error "
                            "although the step_unit is invalid.")


class OptimizerTest:
    def test_every_optimizer_is_valid(self):
        """
        Makes sure that every optimizer is a valid estimator.
        That is, we can fully create an object via get/set params.

        This also test that we can properly initialize each one
        of them
        """
        optimizer_choice = OptimizerChoice(dataset_properties={})

        # Make sure all components are returned
        assert len(optimizer_choice.get_components().keys()) == 4

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
            for k in estimator.get_params().keys():
                assert k in estimator_clone_params

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
                assert param1 == param2

    def test_get_set_config_space(self):
        """Make sure that we can setup a valid choice in the optimizer
        choice"""
        optimizer_choice = OptimizerChoice(dataset_properties={})
        cs = optimizer_choice.get_hyperparameter_search_space()

        # Make sure that all hyperparameters are part of the serach space
        assert sorted(cs.get_hyperparameter('__choice__').choices) == \
               sorted(list(optimizer_choice.get_components().keys()))

        # Make sure we can properly set some random configs
        # Whereas just one iteration will make sure the algorithm works,
        # doing five iterations increase the confidence. We will be able to
        # catch component specific crashes
        for _ in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            optimizer_choice.set_hyperparameters(config)

            assert optimizer_choice.choice.__class__ == optimizer_choice.get_components()[config_dict['__choice__']]

            # Then check the choice configuration
            selected_choice = config_dict.pop('__choice__', None)
            for key, value in config_dict.items():
                # Remove the selected_choice string from the parameter
                # so we can query in the object for it
                key = key.replace(selected_choice + ':', '')
                assert key == vars(optimizer_choice.choice)
                assert value == optimizer_choice.choice.__dict__[key]

    def test_optimizer_add(self):
        """Makes sure that a component can be added to the CS"""
        # No third party components to start with
        assert len(optimizer_components._addons.components) == 0

        # Then make sure the optimizer can be added and query'ed
        optimizer_components.add_optimizer(DummyOptimizer)
        assert len(optimizer_components._addons.components) == 1
        cs = OptimizerChoice(dataset_properties={}).get_hyperparameter_search_space()
        assert 'DummyOptimizer' in str(cs)


class TestNetworkBackbone:
    def test_all_backbones_available(self):
        backbone_choice = NetworkBackboneChoice(dataset_properties={})

        assert len(backbone_choice.get_components().keys()) == 6

    @pytest.mark.parametrize('task_type_input_shape', [(constants.IMAGE_CLASSIFICATION, (3, 64, 64)),
                                                       (constants.IMAGE_REGRESSION, (3, 64, 64)),
                                                       (constants.TABULAR_CLASSIFICATION, (100,)),
                                                       (constants.TABULAR_REGRESSION, (100,))])
    def test_dummy_forward_backward_pass(self, task_type_input_shape):
        network_backbone_choice = NetworkBackboneChoice(dataset_properties={})

        device = torch.device("cpu")
        # shorten search space as it causes out of memory errors in github actions
        updates = HyperparameterSearchSpaceUpdates()
        updates.append(node_name='network_backbone',
                       hyperparameter='ConvNetImageBackbone:num_layers',
                       value_range=[1, 3],
                       default_value=2)
        updates.append(node_name='network_backbone',
                       hyperparameter='ConvNetImageBackbone:conv_init_filters',
                       value_range=[8, 16],
                       default_value=8)
        updates.append(node_name='network_backbone',
                       hyperparameter='DenseNetImageBackbone:num_layers',
                       value_range=[4, 8],
                       default_value=6)
        updates.append(node_name='network_backbone',
                       hyperparameter='DenseNetImageBackbone:num_blocks',
                       value_range=[1, 2],
                       default_value=1)
        updates.apply([('network_backbone', network_backbone_choice)])

        task_type, input_shape = task_type_input_shape
        dataset_properties = {"task_type": constants.TASK_TYPES_TO_STRING[task_type]}

        cs = network_backbone_choice.get_hyperparameter_search_space(dataset_properties=dataset_properties)

        # test 10 random configurations
        for _ in range(10):
            config = cs.sample_configuration()
            network_backbone_choice.set_hyperparameters(config)
            backbone = network_backbone_choice.choice.build_backbone(input_shape=input_shape)
            assert backbone is not None
            backbone = backbone.to(device)
            dummy_input = torch.randn((2, *input_shape), dtype=torch.float)
            output = backbone(dummy_input)
            assert output.shape[1:] != output
            loss = output.sum()
            loss.backward()

    def test_every_backbone_is_valid(self):
        backbone_choice = NetworkBackboneChoice(dataset_properties={})
        assert len(backbone_choice.get_components().keys()) == 6

        for name, backbone in backbone_choice.get_components().items():
            config = backbone.get_hyperparameter_search_space().sample_configuration()
            estimator = backbone(**config)
            estimator_clone = clone(estimator)
            estimator_clone_params = estimator_clone.get_params()

            # Make sure all keys are copied properly
            for k in estimator.get_params().keys():
                assert k in estimator_clone_params

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
                assert param1 == param2

    def test_get_set_config_space(self):
        """
        Make sure that we can setup a valid choice in the network backbone choice
        """
        network_backbone_choice = NetworkBackboneChoice(dataset_properties={})
        for task_type in constants.TASK_TYPES:
            if task_type in constants.FORECASTING_TASKS:
                # Forecasting task has individual backbones
                continue
            dataset_properties = {"task_type": constants.TASK_TYPES_TO_STRING[task_type]}
            cs = network_backbone_choice.get_hyperparameter_search_space(dataset_properties)

            # Make sure we can properly set some random configs
            # Whereas just one iteration will make sure the algorithm works,
            # doing five iterations increase the confidence. We will be able to
            # catch component specific crashes
            for _ in range(5):
                config = cs.sample_configuration()
                config_dict = copy.deepcopy(config.get_dictionary())
                network_backbone_choice.set_hyperparameters(config)

                assert network_backbone_choice.choice.__class__ == \
                       network_backbone_choice.get_components()[config_dict['__choice__']]

                # Then check the choice configuration
                selected_choice = config_dict.pop('__choice__', None)
                assert selected_choice is not None
                for key, value in config_dict.items():
                    # Remove the selected_choice string from the parameter
                    # so we can query in the object for it
                    key = key.replace(selected_choice + ':', '')
                    # parameters are dynamic, so they exist in config
                    parameters = vars(network_backbone_choice.choice)
                    parameters.update(vars(network_backbone_choice.choice)['config'])
                    assert key in parameters
                    assert value == parameters[key]

    def test_add_network_backbone(self):
        """Makes sure that a component can be added to the CS"""
        # No third party components to start with
        assert len(base_network_backbone_choice._addons.components) == 0

        # Then make sure the backbone can be added
        base_network_backbone_choice.add_backbone(DummyBackbone)
        assert len(base_network_backbone_choice._addons.components) == 1

        cs = NetworkBackboneChoice(dataset_properties={}). \
            get_hyperparameter_search_space(dataset_properties={"task_type": "tabular_classification"})
        assert "DummyBackbone" in str(cs)

        # clear addons
        base_network_backbone_choice._addons = ThirdPartyComponents(NetworkBackboneComponent)

    @pytest.mark.parametrize('resnet_shape', ['funnel', 'long_funnel',
                                              'diamond', 'hexagon',
                                              'brick', 'triangle',
                                              'stairs'])
    def test_dropout(self, resnet_shape):
        # ensures that dropout is assigned to the resblock as expected
        dataset_properties = {"task_type": constants.TASK_TYPES_TO_STRING[1]}
        max_dropout = 0.5
        num_groups = 4
        config_space = ShapedResNetBackbone.get_hyperparameter_search_space(dataset_properties=dataset_properties,
                                                                            use_dropout=HyperparameterSearchSpace(
                                                                                hyperparameter='use_dropout',
                                                                                value_range=[True],
                                                                                default_value=True),
                                                                            max_dropout=HyperparameterSearchSpace(
                                                                                hyperparameter='max_dropout',
                                                                                value_range=[max_dropout],
                                                                                default_value=max_dropout),
                                                                            resnet_shape=HyperparameterSearchSpace(
                                                                                hyperparameter='resnet_shape',
                                                                                value_range=[resnet_shape],
                                                                                default_value=resnet_shape),
                                                                            num_groups=HyperparameterSearchSpace(
                                                                                hyperparameter='num_groups',
                                                                                value_range=[num_groups],
                                                                                default_value=num_groups),
                                                                            blocks_per_group=HyperparameterSearchSpace(
                                                                                hyperparameter='blocks_per_group',
                                                                                value_range=[1],
                                                                                default_value=1
                                                                            )
                                                                            )

        config = config_space.sample_configuration().get_dictionary()
        resnet_backbone = ShapedResNetBackbone(**config)
        backbone = resnet_backbone.build_backbone((100, 5))
        dropout_probabilites = [resnet_backbone.config[key] for key in resnet_backbone.config if 'dropout_' in key]
        dropout_shape = get_shaped_neuron_counts(
            shape=resnet_shape,
            in_feat=0,
            out_feat=0,
            max_neurons=max_dropout,
            layer_count=num_groups + 1,
        )[:-1]
        blocks_dropout = []
        for block in backbone:
            if isinstance(block, torch.nn.Sequential):
                for inner_block in block:
                    if isinstance(inner_block, ResBlock):
                        blocks_dropout.append(inner_block.dropout)
        assert dropout_probabilites == dropout_shape == blocks_dropout


class TestNetworkHead:
    def test_all_heads_available(self):
        network_head_choice = NetworkHeadChoice(dataset_properties={})

        assert len(network_head_choice.get_components().keys()) == 2

    @pytest.mark.parametrize('task_type_input_output_shape', [(constants.IMAGE_CLASSIFICATION, (3, 64, 64), (5,)),
                                                              (constants.IMAGE_REGRESSION, (3, 64, 64), (1,)),
                                                              (constants.TABULAR_CLASSIFICATION, (100,), (5,)),
                                                              (constants.TABULAR_REGRESSION, (100,), (1,))])
    def test_dummy_forward_backward_pass(self, task_type_input_output_shape):
        network_head_choice = NetworkHeadChoice(dataset_properties={})

        task_type, input_shape, output_shape = task_type_input_output_shape
        device = torch.device("cpu")

        dataset_properties = {"task_type": constants.TASK_TYPES_TO_STRING[task_type]}
        if task_type in constants.CLASSIFICATION_TASKS:
            dataset_properties["num_classes"] = output_shape[0]

        cs = network_head_choice.get_hyperparameter_search_space(dataset_properties=dataset_properties)
        # test 10 random configurations
        for _ in range(10):
            config = cs.sample_configuration()
            network_head_choice.set_hyperparameters(config)
            head = network_head_choice.choice.build_head(input_shape=input_shape,
                                                         output_shape=output_shape)
            assert head is not None
            head = head.to(device)
            dummy_input = torch.randn((2, *input_shape), dtype=torch.float)
            output = head(dummy_input)
            assert output.shape[1:] == output_shape
            loss = output.sum()
            loss.backward()

    def test_every_head_is_valid(self):
        """
        Makes sure that every network is a valid estimator.
        That is, we can fully create an object via get/set params.

        This also test that we can properly initialize each one
        of them
        """
        network_head_choice = NetworkHeadChoice(dataset_properties={'task_type': 'tabular_classification'})

        # For every network in the components, make sure
        # that it complies with the scikit learn estimator.
        # This is important because usually components are forked to workers,
        # so the set/get params methods should recreate the same object
        for name, network_head in network_head_choice.get_components().items():
            config = network_head.get_hyperparameter_search_space().sample_configuration()
            estimator = network_head(**config)
            estimator_clone = clone(estimator)
            estimator_clone_params = estimator_clone.get_params()

            # Make sure all keys are copied properly
            for k in estimator.get_params().keys():
                assert k in estimator_clone_params

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
                assert param1 == param2

    def test_get_set_config_space(self):
        """
        Make sure that we can setup a valid choice in the network head choice
        """
        network_head_choice = NetworkHeadChoice(dataset_properties={})
        for task_type in constants.TASK_TYPES:
            dataset_properties = {"task_type": constants.TASK_TYPES_TO_STRING[task_type]}
            cs = network_head_choice.get_hyperparameter_search_space(dataset_properties)

            # Make sure we can properly set some random configs
            # Whereas just one iteration will make sure the algorithm works,
            # doing five iterations increase the confidence. We will be able to
            # catch component specific crashes
            for _ in range(5):
                config = cs.sample_configuration()
                config_dict = copy.deepcopy(config.get_dictionary())
                network_head_choice.set_hyperparameters(config)

                assert network_head_choice.choice.__class__ == \
                       network_head_choice.get_components()[config_dict['__choice__']]

                # Then check the choice configuration
                selected_choice = config_dict.pop('__choice__', None)
                assert selected_choice is not None
                for key, value in config_dict.items():
                    # Remove the selected_choice string from the parameter
                    # so we can query in the object for it
                    key = key.replace(selected_choice + ':', '')
                    # parameters are dynamic, so they exist in config
                    parameters = vars(network_head_choice.choice)
                    parameters.update(vars(network_head_choice.choice)['config'])
                    assert key in parameters
                    assert value == parameters[key]

    def test_add_network_head(self):
        """Makes sure that a component can be added to the CS"""
        # No third party components to start with
        assert len(base_network_head_choice._addons.components) == 0

        # Then make sure the head can be added
        base_network_head_choice.add_head(DummyHead)
        assert len(base_network_head_choice._addons.components) == 1

        cs = NetworkHeadChoice(dataset_properties={}). \
            get_hyperparameter_search_space(dataset_properties={"task_type": "tabular_classification"})
        assert "DummyHead" in str(cs)

        # clear addons
        base_network_head_choice._addons = ThirdPartyComponents(NetworkHeadComponent)


class TestNetworkInitializer:
    def test_every_network_initializer_is_valid(self):
        """
        Makes sure that every network_initializer is a valid estimator.
        That is, we can fully create an object via get/set params.

        This also test that we can properly initialize each one
        of them
        """
        network_initializer_choice = NetworkInitializerChoice(dataset_properties={})

        # Make sure all components are returned
        assert len(network_initializer_choice.get_components().keys()) == 5

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
            for k in estimator.get_params().keys():
                assert k in estimator_clone_params

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
                assert param1 == param2

    def test_get_set_config_space(self):
        """Make sure that we can setup a valid choice in the network_initializer
        choice"""
        network_initializer_choice = NetworkInitializerChoice(dataset_properties={})
        cs = network_initializer_choice.get_hyperparameter_search_space()

        # Make sure that all hyperparameters are part of the serach space
        assert sorted(cs.get_hyperparameter('__choice__').choices) == \
               sorted(list(network_initializer_choice.get_components().keys()))

        # Make sure we can properly set some random configs
        # Whereas just one iteration will make sure the algorithm works,
        # doing five iterations increase the confidence. We will be able to
        # catch component specific crashes
        for _ in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            network_initializer_choice.set_hyperparameters(config)

            assert network_initializer_choice.choice.__class__ == \
                   network_initializer_choice.get_components()[config_dict['__choice__']]

            # Then check the choice configuration
            selected_choice = config_dict.pop('__choice__', None)
            for key, value in config_dict.items():
                # Remove the selected_choice string from the parameter
                # so we can query in the object for it
                key = key.replace(selected_choice + ':', '')
                assert key in vars(network_initializer_choice.choice)
                assert value == network_initializer_choice.choice.__dict__[key]

    def test_network_initializer_add(self):
        """Makes sure that a component can be added to the CS"""
        # No third party components to start with
        assert len(network_initializer_components._addons.components) == 0

        # Then make sure the network_initializer can be added and query'ed
        network_initializer_components.add_network_initializer(DummyNetworkInitializer)
        assert len(network_initializer_components._addons.components) == 1
        cs = NetworkInitializerChoice(dataset_properties={}).get_hyperparameter_search_space()
        assert 'DummyNetworkInitializer' in str(cs)
