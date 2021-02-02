from typing import Any, Dict, Optional, Tuple, Union

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter

import numpy as np

from torch import nn

from autoPyTorch.pipeline.components.setup.network_head.base_network_head import NetworkHeadComponent
from autoPyTorch.pipeline.components.setup.network_head.utils import _activations


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
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, str]] = None,
                                        num_layers: Tuple[Tuple, int] = ((1, 4), 2),
                                        units_layer: Tuple[Tuple, int] = ((64, 512), 128),
                                        activation: Tuple[Tuple, str] = (tuple(_activations.keys()),
                                                                         list(_activations.keys())[0])
                                        ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        min_num_layers, max_num_layers = num_layers[0]
        num_layers_hp = UniformIntegerHyperparameter("num_layers",
                                                     lower=min_num_layers,
                                                     upper=max_num_layers,
                                                     default_value=num_layers[1]
                                                     )

        activation_hp = CategoricalHyperparameter(
            "activation", choices=activation[0],
            default_value=activation[1]
        )

        cs.add_hyperparameters([num_layers_hp, activation_hp])
        cs.add_condition(CS.GreaterThanCondition(activation_hp, num_layers_hp, 1))

        for i in range(1, max_num_layers):

            num_units_hp = UniformIntegerHyperparameter(f"units_layer_{i}",
                                                        lower=units_layer[0][0],
                                                        upper=units_layer[0][1],
                                                        default_value=units_layer[1])
            cs.add_hyperparameter(num_units_hp)
            if i >= min_num_layers:
                cs.add_condition(CS.GreaterThanCondition(num_units_hp, num_layers_hp, i))

        return cs
