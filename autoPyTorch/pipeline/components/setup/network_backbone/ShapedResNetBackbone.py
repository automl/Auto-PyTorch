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
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None,  # type: ignore[override]
                                        num_groups: Tuple[Tuple, int] = ((1, 15), 5),
                                        use_batch_norm: Tuple[Tuple, bool] = ((True, False), True),
                                        use_dropout: Tuple[Tuple, bool] = ((True, False), False),
                                        use_skip_connection: Tuple[Tuple, bool] = ((True, False), True),
                                        max_units: Tuple[Tuple, int] = ((10, 1024), 200),
                                        blocks_per_group: Tuple[Tuple, int] = ((1, 4), 2),
                                        max_dropout: Tuple[Tuple, float] = ((0, 0.8), 0.5),
                                        multi_branch_choice: Tuple[Tuple, str] = (('None', 'shake-shake',
                                                                                   'shake-drop'), 'shake-drop'),
                                        max_shake_drop_probability: Tuple[Tuple, float] = ((0, 1), 0.5),
                                        resnet_shape: Tuple[Tuple, str] = (('funnel', 'long_funnel',
                                                                            'diamond', 'hexagon',
                                                                            'brick', 'triangle', 'stairs'), 'funnel'),
                                        activation: Tuple[Tuple, str] = (
                                        tuple(_activations.keys()), list(_activations.keys())[0]),
                                        output_dim: Tuple[Tuple, int] = ((10, 1024), 200),
                                        ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        # Support for different shapes
        resnet_shape = CategoricalHyperparameter(
            'resnet_shape',
            choices=resnet_shape[0],
            default_value=resnet_shape[1]
        )
        cs.add_hyperparameter(resnet_shape)

        # The number of groups that will compose the resnet. That is,
        # a group can have N Resblock. The M number of this N resblock
        # repetitions is num_groups
        num_groups = UniformIntegerHyperparameter(
            "num_groups", lower=num_groups[0][0], upper=num_groups[0][1], default_value=num_groups[1])

        blocks_per_group = UniformIntegerHyperparameter(
            "blocks_per_group", lower=blocks_per_group[0][0],
            upper=blocks_per_group[0][1],
            default_value=blocks_per_group[1])

        activation = CategoricalHyperparameter(
            "activation", choices=activation[0],
            default_value=activation[1]
        )
        (min_num_units, max_num_units), default_units = max_units[:2]
        output_dim = UniformIntegerHyperparameter(
            "output_dim",
            lower=output_dim[0][0],
            upper=output_dim[0][1],
            default_value=output_dim[1]
        )

        cs.add_hyperparameters([num_groups, blocks_per_group, activation, output_dim])

        use_sc = CategoricalHyperparameter(
            "use_skip_connection",
            choices=use_skip_connection[0],
            default_value=use_skip_connection[1],
        )

        mb_choice = CategoricalHyperparameter(
            "multi_branch_choice",
            choices=multi_branch_choice[0],
            default_value=multi_branch_choice[1],
        )

        shake_drop_prob = UniformFloatHyperparameter(
            "max_shake_drop_probability",
            lower=max_shake_drop_probability[0][0],
            upper=max_shake_drop_probability[0][1],
            default_value=max_shake_drop_probability[1])

        cs.add_hyperparameters([use_sc, mb_choice, shake_drop_prob])
        cs.add_condition(CS.EqualsCondition(mb_choice, use_sc, True))
        cs.add_condition(
            CS.AndConjunction(
                CS.EqualsCondition(shake_drop_prob, use_sc, True),
                CS.EqualsCondition(shake_drop_prob, mb_choice, "shake-drop"),
            )
        )

        max_units = UniformIntegerHyperparameter(
            "max_units",
            lower=min_num_units,
            upper=max_num_units,
            default_value=default_units
        )
        cs.add_hyperparameters([max_units])

        use_dropout = CategoricalHyperparameter(
            "use_dropout", choices=use_dropout[0], default_value=use_dropout[1])
        use_batch_normalization = CategoricalHyperparameter(
            "use_batch_norm", choices=use_batch_norm[0], default_value=use_batch_norm[1])

        max_dropout = UniformFloatHyperparameter("max_dropout", lower=max_dropout[0][0], upper=max_dropout[0][1],
                                                 default_value=max_dropout[1])
        cs.add_hyperparameters([use_batch_normalization])
        cs.add_hyperparameters([use_dropout])
        cs.add_hyperparameters([max_dropout])
        cs.add_condition(CS.EqualsCondition(max_dropout, use_dropout, True))

        return cs
