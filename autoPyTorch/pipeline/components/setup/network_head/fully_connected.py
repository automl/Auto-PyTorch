from typing import Any, Dict, Optional, Tuple, Union

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter

import numpy as np

from torch import nn

from autoPyTorch.pipeline.components.setup.network_head.base_network_head import (
    NetworkHeadComponent,
)

_activations: Dict[str, nn.Module] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid
}


class FullyConnectedHead(NetworkHeadComponent):
    """
    Standard head consisting of a number of fully connected layers.
    Flattens any input in a array of shape [B, prod(input_shape)].
    """
    supported_tasks = {"tabular_classification", "tabular_regression",
                       "image_classification", "image_regression",
                       "time_series_classification", "time_series_regression"}

    def build_head(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> nn.Module:
        layers = [nn.Flatten()]
        in_features = np.prod(input_shape).item()
        for i in range(1, self.config["num_layers"]):
            layers.append(nn.Linear(in_features=in_features,
                                    out_features=self.config[f"layer_{i}_units"]))
            layers.append(_activations[self.config["activation"]]())
            in_features = self.config[f"layer_{i}_units"]
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
            'handles_image': False,
            'handles_time_series': False,
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, str]] = None,
                                        min_num_layers: int = 1,
                                        max_num_layers: int = 4,
                                        min_num_units: int = 64,
                                        max_num_units: int = 512) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        num_layers_hp = UniformIntegerHyperparameter("num_layers",
                                                     lower=min_num_layers,
                                                     upper=max_num_layers)

        activation_hp = CategoricalHyperparameter("activation",
                                                  choices=list(_activations.keys()))

        cs.add_hyperparameters([num_layers_hp, activation_hp])
        cs.add_condition(CS.GreaterThanCondition(activation_hp, num_layers_hp, 1))

        for i in range(1, max_num_layers):

            num_units_hp = UniformIntegerHyperparameter(f"layer_{i}_units",
                                                        lower=min_num_units,
                                                        upper=max_num_units)
            cs.add_hyperparameter(num_units_hp)
            if i >= min_num_layers:
                cs.add_condition(CS.GreaterThanCondition(num_units_hp, num_layers_hp, i))

        return cs
