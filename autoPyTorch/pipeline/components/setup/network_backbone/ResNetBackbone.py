from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

import torch
from torch import nn

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_backbone.base_network_backbone import NetworkBackboneComponent
from autoPyTorch.pipeline.components.setup.network_backbone.utils import (
    _activations,
    shake_drop,
    shake_drop_get_bl,
    shake_get_alpha_beta,
    shake_shake
)
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


class ResNetBackbone(NetworkBackboneComponent):
    """
    Implementation of a Residual Network backbone
    """

    def build_backbone(self, input_shape: Tuple[int, ...]) -> torch.nn.Sequential:
        layers: List[nn.Module] = list()
        in_features = input_shape[0]
        layers.append(nn.Linear(in_features, self.config["num_units_0"]))

        # build num_groups-1 groups each consisting of blocks_per_group ResBlocks
        # the output features of each group is defined by num_units_i
        for i in range(1, self.config['num_groups'] + 1):
            layers.append(
                self._add_group(
                    in_features=self.config["num_units_%d" % (i - 1)],
                    out_features=self.config["num_units_%d" % i],
                    blocks_per_group=self.config["blocks_per_group_%d" % i],
                    last_block_index=(i - 1) * self.config["blocks_per_group_%d" % i],
                    dropout=self.config[f'dropout_{i}'] if self.config['use_dropout'] else None,
                )
            )

        layers.append(nn.BatchNorm1d(self.config["num_units_%i" % self.config['num_groups']]))
        layers.append(_activations[self.config["activation"]]())
        backbone = nn.Sequential(*layers)
        return backbone

    def _add_group(self, in_features: int, out_features: int,
                   blocks_per_group: int, last_block_index: int, dropout: Optional[float]
                   ) -> nn.Module:
        """
        Adds a group into the main backbone.
        In the case of ResNet a group is a set of blocks_per_group
        ResBlocks

        Args:
            in_features (int): number of inputs for the current block
            out_features (int): output dimensionality for the current block
            blocks_per_group (int): Number of ResNet per group
            last_block_index (int): block index for shake regularization
            dropout (bool): whether or not use dropout
        """
        blocks = list()
        for i in range(blocks_per_group):
            blocks.append(
                ResBlock(
                    config=self.config,
                    in_features=in_features,
                    out_features=out_features,
                    blocks_per_group=blocks_per_group,
                    block_index=last_block_index + i,
                    dropout=dropout,
                    activation=_activations[self.config["activation"]]
                )
            )
            in_features = out_features
        return nn.Sequential(*blocks)

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'ResNetBackbone',
            'name': 'ResidualNetworkBackbone',
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
        use_dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="use_dropout",
                                                                           value_range=(True, False),
                                                                           default_value=False,
                                                                           ),
        num_units: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="num_units",
                                                                         value_range=(10, 1024),
                                                                         default_value=200,
                                                                         ),
        activation: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="activation",
                                                                          value_range=tuple(_activations.keys()),
                                                                          default_value=list(_activations.keys())[0],
                                                                          ),
        blocks_per_group: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="blocks_per_group",
                                                                                value_range=(1, 4),
                                                                                default_value=2,
                                                                                ),
        dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="dropout",
                                                                       value_range=(0, 0.8),
                                                                       default_value=0.5,
                                                                       ),
        use_shake_shake: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="use_shake_shake",
                                                                               value_range=(True, False),
                                                                               default_value=True,
                                                                               ),
        use_shake_drop: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="use_shake_drop",
                                                                              value_range=(True, False),
                                                                              default_value=True,
                                                                              ),
        max_shake_drop_probability: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="max_shake_drop_probability",
            value_range=(0, 1),
            default_value=0.5),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        # The number of groups that will compose the resnet. That is,
        # a group can have N Resblock. The M number of this N resblock
        # repetitions is num_groups
        min_num_gropus, max_num_groups = num_groups.value_range
        num_groups = get_hyperparameter(num_groups, UniformIntegerHyperparameter)

        add_hyperparameter(cs, activation, CategoricalHyperparameter)
        cs.add_hyperparameters([num_groups])

        # We can have dropout in the network for
        # better generalization
        use_dropout = get_hyperparameter(use_dropout, CategoricalHyperparameter)
        cs.add_hyperparameters([use_dropout])

        use_shake_shake = get_hyperparameter(use_shake_shake, CategoricalHyperparameter)
        use_shake_drop = get_hyperparameter(use_shake_drop, CategoricalHyperparameter)
        shake_drop_prob = get_hyperparameter(max_shake_drop_probability, UniformFloatHyperparameter)
        cs.add_hyperparameters([use_shake_shake, use_shake_drop, shake_drop_prob])
        cs.add_condition(CS.EqualsCondition(shake_drop_prob, use_shake_drop, True))

        # It is the upper bound of the nr of groups,
        # since the configuration will actually be sampled.
        for i in range(0, int(max_num_groups) + 1):

            n_units_search_space = HyperparameterSearchSpace(hyperparameter='num_units_%d' % i,
                                                             value_range=num_units.value_range,
                                                             default_value=num_units.default_value,
                                                             log=num_units.log)
            n_units_hp = get_hyperparameter(n_units_search_space, UniformIntegerHyperparameter)

            blocks_per_group_search_space = HyperparameterSearchSpace(hyperparameter='blocks_per_group_%d' % i,
                                                                      value_range=blocks_per_group.value_range,
                                                                      default_value=blocks_per_group.default_value,
                                                                      log=blocks_per_group.log)
            blocks_per_group_hp = get_hyperparameter(blocks_per_group_search_space, UniformIntegerHyperparameter)
            cs.add_hyperparameters([n_units_hp, blocks_per_group_hp])

            if i > 1:
                cs.add_condition(CS.GreaterThanCondition(n_units_hp, num_groups, i - 1))
                cs.add_condition(CS.GreaterThanCondition(blocks_per_group_hp, num_groups, i - 1))

            dropout_search_space = HyperparameterSearchSpace(hyperparameter='dropout_%d' % i,
                                                             value_range=dropout.value_range,
                                                             default_value=dropout.default_value,
                                                             log=dropout.log)
            dropout_hp = get_hyperparameter(dropout_search_space, UniformFloatHyperparameter)
            cs.add_hyperparameter(dropout_hp)

            dropout_condition_1 = CS.EqualsCondition(dropout_hp, use_dropout, True)

            if i > 1:

                dropout_condition_2 = CS.GreaterThanCondition(dropout_hp, num_groups, i - 1)

                cs.add_condition(CS.AndConjunction(dropout_condition_1, dropout_condition_2))
            else:
                cs.add_condition(dropout_condition_1)
        return cs


class ResBlock(nn.Module):
    """
    __author__ = "Max Dippel, Michael Burkart and Matthias Urban"
    """

    def __init__(
        self,
        config: Dict[str, Any],
        in_features: int,
        out_features: int,
        blocks_per_group: int,
        block_index: int,
        dropout: Optional[float],
        activation: nn.Module
    ):
        super(ResBlock, self).__init__()
        self.config = config
        self.dropout = dropout
        self.activation = activation

        self.shortcut = None
        self.start_norm: Optional[Callable] = None

        # if in != out the shortcut needs a linear layer to match the result dimensions
        # if the shortcut needs a layer we apply batchnorm and activation to the shortcut
        # as well (start_norm)
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
            self.start_norm = nn.Sequential(
                nn.BatchNorm1d(in_features),
                self.activation()
            )

        self.block_index = block_index
        self.num_blocks = blocks_per_group * self.config["num_groups"]
        self.layers = self._build_block(in_features, out_features)

        if config["use_shake_shake"]:
            self.shake_shake_layers = self._build_block(in_features, out_features)

    # each bloack consists of two linear layers with batch norm and activation
    def _build_block(self, in_features: int, out_features: int) -> nn.Module:
        layers = list()

        if self.start_norm is None:
            layers.append(nn.BatchNorm1d(in_features))
            layers.append(self.activation())
        layers.append(nn.Linear(in_features, out_features))

        layers.append(nn.BatchNorm1d(out_features))
        layers.append(self.activation())

        if self.config["use_dropout"]:
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(out_features, out_features))

        return nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        residual = x

        # if shortcut is not none we need a layer such that x matches the output dimension
        if self.shortcut is not None and self.start_norm is not None:
            # in this case self.start_norm is also != none
            # apply start_norm to x in order to have batchnorm+activation
            # in front of shortcut and layers. Note that in this case layers
            # does not start with batchnorm+activation but with the first linear layer
            # (see _build_block). As a result if in_features == out_features
            # -> result = x + W(~D(A(BN(W(A(BN(x))))))
            # if in_features != out_features
            # -> result = W_shortcut(A(BN(x))) + W_2(~D(A(BN(W_1(A(BN(x))))))
            x = self.start_norm(x)
            residual = self.shortcut(x)

        if self.config["use_shake_shake"]:
            x1 = self.layers(x)
            x2 = self.shake_shake_layers(x)
            alpha, beta = shake_get_alpha_beta(self.training, x.is_cuda)
            x = shake_shake(x1, x2, alpha, beta)
        else:
            x = self.layers(x)

        if self.config["use_shake_drop"]:
            alpha, beta = shake_get_alpha_beta(self.training, x.is_cuda)
            bl = shake_drop_get_bl(
                self.block_index,
                1 - self.config["max_shake_drop_probability"],
                self.num_blocks,
                self.training,
                x.is_cuda
            )
            x = shake_drop(x, alpha, beta, bl)

        x = x + residual
        return x
