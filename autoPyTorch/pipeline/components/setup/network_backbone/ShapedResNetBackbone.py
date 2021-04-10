from typing import Any, Dict, List, Optional, Tuple, Union

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

import torch

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

    def build_backbone(self, input_shape: Tuple[int, ...]) -> None:
        layers = list()  # type: List[torch.nn.Module]
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
        if self.config['use_dropout'] and self.config["max_dropout"] > 0.05:
            dropout_shape = get_shaped_neuron_counts(
                self.config['resnet_shape'], 0, 0, 1000, self.config['num_groups']
            )

            dropout_shape = [
                dropout / 1000 * self.config["max_dropout"] for dropout in dropout_shape
            ]

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
                    dropout=self.config['use_dropout']
                )
            )
        if self.config['use_batch_norm']:
            layers.append(torch.nn.BatchNorm1d(self.config["num_units_%i" % self.config['num_groups']]))
        backbone = torch.nn.Sequential(*layers)
        self.backbone = backbone
        return backbone

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'ShapedResNetBackbone',
            'name': 'ShapedResidualNetworkBackbone',
            'handles_tabular': True,
            'handles_image': False,
            'handles_time_series': False,
        }

    @staticmethod
    def get_hyperparameter_search_space(  # type: ignore[override]
        dataset_properties: Optional[Dict] = None,
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
                                                                          log=True
                                                                          ),
        num_groups: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="num_groups",
                                                                          value_range=(1, 15),
                                                                          default_value=5,
                                                                          ),
        use_dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="use_dropout",
                                                                           value_range=(True, False),
                                                                           default_value=False,
                                                                           ),
        use_batch_norm: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="use_batch_norm",
                                                                              value_range=(True, False),
                                                                              default_value=False,
                                                                              ),
        use_skip_connection: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="use_skip_connection",
                                                                                   value_range=(True, False),
                                                                                   default_value=True,
                                                                                   ),
        multi_branch_choice: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="mb_choice",
                                                                                   value_range=('None', 'shake-shake',
                                                                                                'shake-drop'),
                                                                                   default_value='shake-drop',
                                                                                   ),
        max_units: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="max_units",
                                                                         value_range=(10, 1024),
                                                                         default_value=200,
                                                                         log=True
                                                                         ),
        activation: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="activation",
                                                                          value_range=tuple(_activations.keys()),
                                                                          default_value=list(_activations.keys())[0]),
        blocks_per_group: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="blocks_per_group",
                                                                                value_range=(1, 4),
                                                                                default_value=2),
        max_dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="max_dropout",
                                                                           value_range=(0, 0.8),
                                                                           default_value=0.5),
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
        add_hyperparameter(cs, max_units, UniformIntegerHyperparameter)
        add_hyperparameter(cs, activation, CategoricalHyperparameter)
        # activation controlled batch normalization
        add_hyperparameter(cs, use_batch_norm, CategoricalHyperparameter)
        add_hyperparameter(cs, output_dim, UniformIntegerHyperparameter)

        use_dropout = get_hyperparameter(use_dropout, CategoricalHyperparameter)
        max_dropout = get_hyperparameter(max_dropout, UniformFloatHyperparameter)
        cs.add_hyperparameters([use_dropout, max_dropout])
        cs.add_condition(CS.EqualsCondition(max_dropout, use_dropout, True))

        use_sc = get_hyperparameter(use_skip_connection, CategoricalHyperparameter)
        mb_choice = get_hyperparameter(multi_branch_choice, CategoricalHyperparameter)
        shake_drop_prob = get_hyperparameter(max_shake_drop_probability, UniformFloatHyperparameter)
        cs.add_hyperparameters([use_sc, mb_choice, shake_drop_prob])
        cs.add_condition(CS.EqualsCondition(mb_choice, use_sc, True))
        # TODO check if shake_drop is as an option in mb_choice
        # Incomplete work
        cs.add_condition(CS.EqualsCondition(shake_drop_prob, mb_choice, "shake-drop"))

        return cs
