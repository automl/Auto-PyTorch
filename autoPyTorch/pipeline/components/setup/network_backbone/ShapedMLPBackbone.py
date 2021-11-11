from typing import Dict, List, Optional, Tuple, Union

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

from torch import nn

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_backbone.base_network_backbone import NetworkBackboneComponent
from autoPyTorch.pipeline.components.setup.network_backbone.utils import (
    _activations,
    get_shaped_neuron_counts,
)
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


class ShapedMLPBackbone(NetworkBackboneComponent):
    """
    Implementation of a Shaped MLP -- an MLP with the number of units
    arranged so that a given shape is honored
    """

    def build_backbone(self, input_shape: Tuple[int, ...]) -> nn.Module:
        layers: List[nn.Module] = list()
        in_features = input_shape[0]
        out_features = self.config["output_dim"]
        neuron_counts = get_shaped_neuron_counts(self.config['mlp_shape'],
                                                 in_features,
                                                 out_features,
                                                 self.config['max_units'],
                                                 self.config['num_groups'])
        if self.config["use_dropout"] and self.config["max_dropout"] > 0.05:
            dropout_shape = get_shaped_neuron_counts(
                self.config['mlp_shape'], 0, 0, 1000, self.config['num_groups']
            )

        previous = in_features
        for i in range(self.config['num_groups'] - 1):
            if i >= len(neuron_counts):
                break
            if self.config["use_dropout"] and self.config["max_dropout"] > 0.05:
                dropout = dropout_shape[i] / 1000 * self.config["max_dropout"]
            else:
                dropout = 0.0
            self._add_layer(layers, previous, neuron_counts[i], dropout)
            previous = neuron_counts[i]
        layers.append(nn.Linear(previous, out_features))

        backbone = nn.Sequential(*layers)
        self.backbone = backbone
        return backbone

    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return (self.config["output_dim"],)

    def _add_layer(self, layers: List[nn.Module],
                   in_features: int, out_features: int, dropout: float
                   ) -> None:
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.BatchNorm1d(out_features))
        layers.append(_activations[self.config["activation"]]())
        if self.config["use_dropout"] and self.config["max_dropout"] > 0.05:
            layers.append(nn.Dropout(dropout))

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'ShapedMLPBackbone',
            'name': 'ShapedMLPBackbone',
            'handles_tabular': True,
            'handles_image': False,
            'handles_time_series': False,
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        num_groups: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="num_groups",
                                                                          value_range=(1, 15),
                                                                          default_value=5,
                                                                          ),
        max_dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="max_dropout",
                                                                           value_range=(0, 1),
                                                                           default_value=0.5,
                                                                           ),
        use_dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="use_dropout",
                                                                           value_range=(True, False),
                                                                           default_value=False,
                                                                           ),
        max_units: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="max_units",
                                                                         value_range=(10, 1024),
                                                                         default_value=200,
                                                                         ),
        output_dim: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="output_dim",
                                                                          value_range=(10, 1024),
                                                                          default_value=200,
                                                                          ),
        mlp_shape: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="mlp_shape",
                                                                         value_range=('funnel', 'long_funnel',
                                                                                      'diamond', 'hexagon',
                                                                                      'brick', 'triangle',
                                                                                      'stairs'),
                                                                         default_value='funnel',
                                                                         ),
        activation: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="activation",
                                                                          value_range=tuple(_activations.keys()),
                                                                          default_value=list(_activations.keys())[0],
                                                                          ),

    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()

        # The number of groups that will compose the resnet. That is,
        # a group can have N Resblock. The M number of this N resblock
        # repetitions is num_groups
        add_hyperparameter(cs, num_groups, UniformIntegerHyperparameter)
        add_hyperparameter(cs, mlp_shape, CategoricalHyperparameter)
        add_hyperparameter(cs, activation, CategoricalHyperparameter)
        add_hyperparameter(cs, max_units, UniformIntegerHyperparameter)
        add_hyperparameter(cs, output_dim, UniformIntegerHyperparameter)

        # We can have dropout in the network for
        # better generalization
        use_dropout = get_hyperparameter(use_dropout, CategoricalHyperparameter)
        max_dropout = get_hyperparameter(max_dropout, UniformFloatHyperparameter)

        cs.add_hyperparameters([use_dropout, max_dropout])
        cs.add_condition(CS.EqualsCondition(max_dropout, use_dropout, True))

        return cs
