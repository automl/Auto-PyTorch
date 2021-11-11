from typing import Dict, List, Optional, Tuple, Union

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

import torch

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_backbone.ResNetBackbone import ResNetBackbone
from autoPyTorch.pipeline.components.setup.network_backbone.utils import (
    _activations,
    get_shaped_neuron_counts,
)
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


class ShapedResNetBackbone(ResNetBackbone):
    """
    Implementation of a Residual Network builder with support
    for shaped number of units per group.
    """

    def build_backbone(self, input_shape: Tuple[int, ...]) -> torch.nn.Sequential:
        layers: List[torch.nn.Module] = list()
        in_features = input_shape[0]
        out_features = self.config["output_dim"]

        # use the get_shaped_neuron_counts to update the number of units
        neuron_counts = get_shaped_neuron_counts(self.config['resnet_shape'],
                                                 in_features,
                                                 out_features,
                                                 self.config['max_units'],
                                                 self.config['num_groups'] + 2)[:-1]
        self.config.update(
            {"num_units_%d" % (i): num for i, num in enumerate(neuron_counts)}
        )
        if self.config['use_dropout']:
            # the last dropout ("neuron") value is skipped since it will be equal
            # to output_feat, which is 0. This is also skipped when getting the
            # n_units for the architecture, since, it is mostly implemented for the
            # output layer, which is part of the head and not of the backbone.
            dropout_shape = get_shaped_neuron_counts(
                shape=self.config['resnet_shape'],
                in_feat=0,
                out_feat=0,
                max_neurons=self.config["max_dropout"],
                layer_count=self.config['num_groups'] + 1,
            )[:-1]

            self.config.update(
                {"dropout_%d" % (i + 1): dropout for i, dropout in enumerate(dropout_shape)}
            )
        layers.append(torch.nn.Linear(in_features, self.config["num_units_0"]))

        # build num_groups-1 groups each consisting of blocks_per_group ResBlocks
        # the output features of each group is defined by num_units_i
        for i in range(1, self.config['num_groups'] + 1):
            layers.append(
                self._add_group(
                    in_features=self.config["num_units_%d" % (i - 1)],
                    out_features=self.config["num_units_%d" % i],
                    blocks_per_group=self.config["blocks_per_group"],
                    last_block_index=(i - 1) * self.config["blocks_per_group"],
                    dropout=self.config[f'dropout_{i}'] if self.config['use_dropout'] else None
                )
            )

        layers.append(torch.nn.BatchNorm1d(self.config["num_units_%i" % self.config['num_groups']]))
        backbone = torch.nn.Sequential(*layers)
        return backbone

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'ShapedResNetBackbone',
            'name': 'ShapedResidualNetworkBackbone',
            'handles_tabular': True,
            'handles_image': False,
            'handles_time_series': False,
        }

    @staticmethod
    def get_hyperparameter_search_space(  # type: ignore[override]
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        resnet_shape: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="resnet_shape",
                                                                            value_range=('funnel', 'long_funnel',
                                                                                         'diamond', 'hexagon',
                                                                                         'brick', 'triangle',
                                                                                         'stairs'),
                                                                            default_value='funnel',
                                                                            ),
        output_dim: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="output_dim",
                                                                          value_range=(10, 1024),
                                                                          default_value=200,
                                                                          ),
        num_groups: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="num_groups",
                                                                          value_range=(1, 15),
                                                                          default_value=5,
                                                                          ),
        use_dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="use_dropout",
                                                                           value_range=(True, False),
                                                                           default_value=False,
                                                                           ),
        max_units: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="max_units",
                                                                         value_range=(10, 1024),
                                                                         default_value=200),
        activation: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="activation",
                                                                          value_range=tuple(_activations.keys()),
                                                                          default_value=list(_activations.keys())[0]),
        blocks_per_group: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="blocks_per_group",
                                                                                value_range=(1, 4),
                                                                                default_value=2),
        max_dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="max_dropout",
                                                                           value_range=(0, 0.8),
                                                                           default_value=0.5),
        use_shake_shake: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="use_shake_shake",
                                                                               value_range=(True, False),
                                                                               default_value=True),
        use_shake_drop: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="use_shake_drop",
                                                                              value_range=(True, False),
                                                                              default_value=True),
        max_shake_drop_probability: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="max_shake_drop_probability",
            value_range=(0, 1),
            default_value=0.5),
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()

        # Support for different shapes
        add_hyperparameter(cs, resnet_shape, CategoricalHyperparameter)

        # The number of groups that will compose the resnet. That is,
        # a group can have N Resblock. The M number of this N resblock
        # repetitions is num_groups
        add_hyperparameter(cs, num_groups, UniformIntegerHyperparameter)
        add_hyperparameter(cs, blocks_per_group, UniformIntegerHyperparameter)

        add_hyperparameter(cs, activation, CategoricalHyperparameter)
        add_hyperparameter(cs, output_dim, UniformIntegerHyperparameter)

        use_shake_shake = get_hyperparameter(use_shake_shake, CategoricalHyperparameter)
        use_shake_drop = get_hyperparameter(use_shake_drop, CategoricalHyperparameter)
        shake_drop_prob = get_hyperparameter(max_shake_drop_probability, UniformFloatHyperparameter)
        cs.add_hyperparameters([use_shake_shake, use_shake_drop, shake_drop_prob])
        cs.add_condition(CS.EqualsCondition(shake_drop_prob, use_shake_drop, True))

        add_hyperparameter(cs, max_units, UniformIntegerHyperparameter)

        use_dropout = get_hyperparameter(use_dropout, CategoricalHyperparameter)
        max_dropout = get_hyperparameter(max_dropout, UniformFloatHyperparameter)

        cs.add_hyperparameters([use_dropout])
        cs.add_hyperparameters([max_dropout])
        cs.add_condition(CS.EqualsCondition(max_dropout, use_dropout, True))

        return cs
