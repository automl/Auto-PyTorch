from typing import Any, Dict, Optional, Tuple, Union

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter

import numpy as np

from torch import nn

from autoPyTorch.pipeline.components.setup.network_head.base_network_head import NetworkHeadComponent
from autoPyTorch.pipeline.components.setup.network_head.utils import _activations
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


class FullyConnectedHead(NetworkHeadComponent):
    """
    Head consisting of a number of fully connected layers.
    Flattens any input in a array of shape [B, prod(input_shape)].
    """

    def build_head(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> nn.Module:
        layers = [nn.Flatten()]
        in_features = np.prod(input_shape).item()
        for i in range(1, self.config["num_layers"]):
            layers.append(nn.Linear(in_features=in_features,
                                    out_features=self.config[f"units_layer_{i}"]))
            layers.append(_activations[self.config["activation"]]())
            in_features = self.config[f"units_layer_{i}"]
        out_features = np.prod(output_shape).item()
        layers.append(nn.Linear(in_features=in_features,
                                out_features=out_features))
        return nn.Sequential(*layers)

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'FullyConnectedHead',
            'name': 'FullyConnectedHead',
            'handles_tabular': True,
            'handles_image': True,
            'handles_time_series': True,
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None,
        num_layers: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="num_layers",
                                                                          value_range=(1, 4),
                                                                          default_value=2),
        units_layer: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="units_layer",
                                                                           value_range=(64, 512),
                                                                           default_value=128),
        activation: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="activation",
                                                                          value_range=tuple(_activations.keys()),
                                                                          default_value=list(_activations.keys())[0]),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        min_num_layers, max_num_layers = num_layers.value_range
        num_layers_hp = get_hyperparameter(num_layers, UniformIntegerHyperparameter)

        activation_hp = get_hyperparameter(activation, CategoricalHyperparameter)

        cs.add_hyperparameters([num_layers_hp, activation_hp])
        cs.add_condition(CS.GreaterThanCondition(activation_hp, num_layers_hp, 1))

        for i in range(1, max_num_layers):
            num_units_search_space = HyperparameterSearchSpace(hyperparameter=f"units_layer_{i}",
                                                               value_range=units_layer.value_range,
                                                               default_value=units_layer.default_value,
                                                               log=units_layer.log)
            num_units_hp = get_hyperparameter(num_units_search_space, UniformIntegerHyperparameter)
            cs.add_hyperparameter(num_units_hp)

            if i >= min_num_layers:
                cs.add_condition(CS.GreaterThanCondition(num_units_hp, num_layers_hp, i))

        return cs
