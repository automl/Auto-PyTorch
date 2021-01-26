from typing import Any, Dict, List, Optional, Tuple, Union

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

import torch

from autoPyTorch.pipeline.components.setup.network_backbone.ResNetBackbone import ResNetBackbone
from autoPyTorch.pipeline.components.setup.network_backbone.utils import (
    _activations,
    get_shaped_neuron_counts,
)


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
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None,
                                        min_num_gropus: int = 1,
                                        max_num_groups: int = 9,
                                        min_blocks_per_groups: int = 1,
                                        max_blocks_per_groups: int = 4,
                                        min_num_units: int = 10,
                                        max_num_units: int = 1024,
                                        ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        # Support for different shapes
        resnet_shape = CategoricalHyperparameter(
            'resnet_shape',
            choices=[
                'funnel',
                'long_funnel',
                'diamond',
                'hexagon',
                'brick',
                'triangle',
                'stairs'
            ]
        )
        cs.add_hyperparameter(resnet_shape)

        # The number of groups that will compose the resnet. That is,
        # a group can have N Resblock. The M number of this N resblock
        # repetitions is num_groups
        num_groups = UniformIntegerHyperparameter(
            "num_groups", lower=min_num_gropus, upper=max_num_groups, default_value=5)

        blocks_per_group = UniformIntegerHyperparameter(
            "blocks_per_group", lower=min_blocks_per_groups, upper=max_blocks_per_groups)

        activation = CategoricalHyperparameter(
            "activation", choices=list(_activations.keys())
        )

        output_dim = UniformIntegerHyperparameter(
            "output_dim",
            lower=min_num_units,
            upper=max_num_units
        )

        cs.add_hyperparameters([num_groups, blocks_per_group, activation, output_dim])

        # We can have dropout in the network for
        # better generalization
        use_dropout = CategoricalHyperparameter(
            "use_dropout", choices=[True, False])
        cs.add_hyperparameters([use_dropout])

        use_shake_shake = CategoricalHyperparameter("use_shake_shake", choices=[True, False])
        use_shake_drop = CategoricalHyperparameter("use_shake_drop", choices=[True, False])
        shake_drop_prob = UniformFloatHyperparameter(
            "max_shake_drop_probability", lower=0.0, upper=1.0)
        cs.add_hyperparameters([use_shake_shake, use_shake_drop, shake_drop_prob])
        cs.add_condition(CS.EqualsCondition(shake_drop_prob, use_shake_drop, True))

        max_units = UniformIntegerHyperparameter(
            "max_units",
            lower=min_num_units,
            upper=max_num_units,
        )
        cs.add_hyperparameters([max_units])

        max_dropout = UniformFloatHyperparameter(
            "max_dropout", lower=0.0, upper=1.0
        )
        cs.add_hyperparameters([max_dropout])
        cs.add_condition(CS.EqualsCondition(max_dropout, use_dropout, True))

        return cs
