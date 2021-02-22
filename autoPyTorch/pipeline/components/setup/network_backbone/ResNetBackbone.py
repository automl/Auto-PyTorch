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

from autoPyTorch.pipeline.components.setup.network_backbone.base_network_backbone import NetworkBackboneComponent
from autoPyTorch.pipeline.components.setup.network_backbone.utils import (
    _activations,
    shake_drop,
    shake_drop_get_bl,
    shake_get_alpha_beta,
    shake_shake
)


class ResNetBackbone(NetworkBackboneComponent):
    """
    Implementation of a Residual Network backbone
    """

    def build_backbone(self, input_shape: Tuple[int, ...]) -> None:
        layers = list()  # type: List[nn.Module]
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
                    dropout=self.config['use_dropout']
                )
            )
        if self.config['use_batch_norm']:
            layers.append(nn.BatchNorm1d(self.config["num_units_%i" % self.config['num_groups']]))
        layers.append(_activations[self.config["activation"]]())
        backbone = nn.Sequential(*layers)
        self.backbone = backbone
        return backbone

    def _add_group(self, in_features: int, out_features: int,
                   blocks_per_group: int, last_block_index: int, dropout: bool
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
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'ResNetBackbone',
            'name': 'ResidualNetworkBackbone',
            'handles_tabular': True,
            'handles_image': False,
            'handles_time_series': False,
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None,
                                        num_groups: Tuple[Tuple, int] = ((1, 15), 5),
                                        use_batch_norm: Tuple[Tuple, bool] = ((True, False), True),
                                        use_dropout: Tuple[Tuple, bool] = ((True, False), False),
                                        use_skip_connection: Tuple[Tuple, bool] = ((True, False), True),
                                        num_units: Tuple[Tuple, int] = ((10, 1024), 200),
                                        activation: Tuple[Tuple, str] = (tuple(_activations.keys()),
                                                                         list(_activations.keys())[0]),
                                        blocks_per_group: Tuple[Tuple, int] = ((1, 4), 2),
                                        dropout: Tuple[Tuple, float] = ((0, 0.8), 0.5),
                                        multi_branch_choice: Tuple[Tuple, str] = (('shake-drop', 'shake-shake',
                                                                                   'none'), 'shake-drop'),
                                        max_shake_drop_probability: Tuple[Tuple, float] = ((0, 1), 0.5)
                                        ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        # The number of groups that will compose the resnet. That is,
        # a group can have N Resblock. The M number of this N resblock
        # repetitions is num_groups
        min_num_gropus, max_num_groups = num_groups[0]
        num_groups = UniformIntegerHyperparameter(
            "num_groups", lower=min_num_gropus, upper=max_num_groups, default_value=num_groups[1])

        activation = CategoricalHyperparameter(
            "activation", choices=activation[0],
            default_value=activation[1]
        )
        cs.add_hyperparameters([num_groups, activation])

        # We can have dropout in the network for
        # better generalization
        use_dropout = CategoricalHyperparameter("use_dropout", choices=use_dropout[0], default_value=use_dropout[1])
        use_batch_normalization = CategoricalHyperparameter(
            "use_batch_norm", choices=use_batch_norm[0], default_value=use_batch_norm[1])
        cs.add_hyperparameters([use_dropout])
        cs.add_hyperparameters([use_batch_normalization])

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
        #TODO check if shake_drop is as an option in mb_choice
        # Incomplete work
        cs.add_condition(CS.EqualsCondition(shake_drop_prob, mb_choice, "shake-drop"))

        # It is the upper bound of the nr of groups,
        # since the configuration will actually be sampled.
        (min_blocks_per_group, max_blocks_per_group), default_blocks_per_group = blocks_per_group[:2]
        for i in range(0, max_num_groups + 1):

            n_units = UniformIntegerHyperparameter(
                "num_units_%d" % i,
                lower=num_units[0][0],
                upper=num_units[0][1],
                default_value=num_units[1]
            )
            blocks_per_group = UniformIntegerHyperparameter(
                "blocks_per_group_%d" % i,
                lower=min_blocks_per_group,
                upper=max_blocks_per_group,
                default_value=default_blocks_per_group)

            cs.add_hyperparameters([n_units, blocks_per_group])

            if i > 1:
                cs.add_condition(CS.GreaterThanCondition(n_units, num_groups, i - 1))
                cs.add_condition(CS.GreaterThanCondition(blocks_per_group, num_groups, i - 1))

            this_dropout = UniformFloatHyperparameter(
                "dropout_%d" % i,
                lower=dropout[0][0],
                upper=dropout[0][1],
                default_value=dropout[1]
            )
            cs.add_hyperparameters([this_dropout])

            dropout_condition_1 = CS.EqualsCondition(this_dropout, use_dropout, True)

            if i > 1:

                dropout_condition_2 = CS.GreaterThanCondition(this_dropout, num_groups, i - 1)

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
            dropout: bool,
            activation: nn.Module
    ):
        super(ResBlock, self).__init__()
        self.config = config
        self.dropout = dropout
        self.activation = activation

        self.shortcut = None
        self.start_norm = None  # type: Optional[Callable]

        # if in != out the shortcut needs a linear layer to match the result dimensions
        # if the shortcut needs a layer we apply batchnorm and activation to the shortcut
        # as well (start_norm)
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
            initial_normalization = list()
            if self.config['use_batch_norm']:
                initial_normalization.append(
                    nn.BatchNorm1d(in_features)
                )
            initial_normalization.append(
                self.activation()
            )
            self.start_norm = nn.Sequential(
                *initial_normalization
            )

        self.block_index = block_index
        self.num_blocks = blocks_per_group * self.config["num_groups"]
        self.layers = self._build_block(in_features, out_features)

        if self.config["use_skip_connection"]:
            if config["multi_branch_choice"] == 'shake-shake':
                self.shake_shake_layers = self._build_block(in_features, out_features)

    # each block consists of two linear layers with batch norm and activation
    def _build_block(self, in_features: int, out_features: int) -> nn.Module:
        layers = list()

        if self.start_norm is None:
            if self.config['use_batch_norm']:
                layers.append(nn.BatchNorm1d(in_features))
            layers.append(self.activation())
        layers.append(nn.Linear(in_features, out_features))

        if self.config['use_batch_norm']:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(self.activation())

        if self.config["use_dropout"]:
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(out_features, out_features))

        return nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:

        if self.config["use_skip_connection"]:
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
            if self.config["use_skip_connection"]:
                residual = self.shortcut(x)

        if self.config["use_skip_connection"]:
            if self.config["multi_branch_choice"] == 'shake-shake':
                x1 = self.layers(x)
                x2 = self.shake_shake_layers(x)
                alpha, beta = shake_get_alpha_beta(self.training, x.is_cuda)
                x = shake_shake(x1, x2, alpha, beta)
            else:
                x = self.layers(x)
        else:
            x = self.layers(x)

        if self.config["use_skip_connection"]:
            if self.config["multi_branch_choice"] == 'shake-drop':
                alpha, beta = shake_get_alpha_beta(self.training, x.is_cuda)
                bl = shake_drop_get_bl(
                    self.block_index,
                    1 - self.config["max_shake_drop_probability"],
                    self.num_blocks,
                    self.training,
                    x.is_cuda,
                )
                x = shake_drop(x, alpha, beta, bl)

        if self.config["use_skip_connection"]:
            x = x + residual

        return x
