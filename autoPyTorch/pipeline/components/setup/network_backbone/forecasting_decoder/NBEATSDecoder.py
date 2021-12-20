from typing import Any, Dict, List, Optional, Union, Tuple

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, \
    UniformFloatHyperparameter
from ConfigSpace.conditions import GreaterThanCondition, InCondition, EqualsCondition, AndConjunction
from ConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenAndConjunction

from typing import Dict, Optional, Tuple, Union, Any

from torch import nn


from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_head.utils import _activations
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_decoder.base_forecasting_decoder import \
    BaseForecastingDecoder

# TODO we need to rewrite NBEATS part to make it neater!!!


class NBEATSBLock(nn.Module):
    def __init__(self,
                 n_in_features: int,
                 stack_idx: int,
                 config: Dict,
                 ):
        super().__init__()
        self.n_in_features = n_in_features
        self.stack_idx = stack_idx

        self.weight_sharing = config['weight_sharing_%d' % self.stack_idx]
        self.num_blocks = config['num_blocks_%d' % self.stack_idx]
        self.stack_type = config['stack_type_%d' % self.stack_idx]
        if self.stack_type == 'generic':
            self.expansion_coefficient_length = config['expansion_coefficient_length_generic_%d' % self.stack_idx]
        else:
            self.expansion_coefficient_length = config['expansion_coefficient_length_interpretable_%d' % self.stack_idx]

        self.num_layers = config['num_layers_%d' % self.stack_idx]
        self.width = config['width_%d' % self.stack_idx]
        self.normalization = config['normalization']
        self.activation = config['activation']
        self.use_dropout = config['use_dropout']
        self.dropout_rate = config.get('dropout_%d' % self.stack_idx, None)

        self.backbone = nn.Sequential(*self.build_backbone())

        self.backcast_head = None
        self.forecast_head = None

    def build_backbone(self):
        layers: List[nn.Module] = list()
        for _ in range(self.num_layers):
            self._add_layer(layers, self.n_in_features)
        return layers

    def _add_layer(self, layers: List[nn.Module], in_features: int) -> None:
        layers.append(nn.Linear(in_features, self.width))
        if self.normalization == 'BN':
            layers.append(nn.BatchNorm1d(self.width))
        elif self.normalization == 'LN':
            layers.append(nn.LayerNorm(self.width))
        layers.append(_activations[self.activation]())
        if self.use_dropout:
            layers.append(nn.Dropout(self.dropout_rate))

    def forward(self, x):
        if self.backcast_head is None and self.forecast_head is None:
            # used to compute head dimensions
            return self.backbone(x)
        else:
            x = self.backbone(x)
            forecast = self.forecast_head(x)
            backcast = self.backcast_head(x)
            return backcast, forecast


class NBEATSDecoder(BaseForecastingDecoder):
    _fixed_seq_length = True
    window_size = 1
    fill_lower_resolution_seq = False
    fill_kwargs = {}

    def decoder_properties(self):
        decoder_properties = super().decoder_properties()
        decoder_properties.update({
            'multi_blocks': True
        })
        return decoder_properties

    def _build_decoder(self, input_shape: Tuple[int, ...], n_prediction_heads: int,
                       dataset_properties:Dict) -> Tuple[nn.Module, int]:
        in_features = input_shape[-1]
        stacks = [[] for _ in range(self.config['num_stacks'])]
        for stack_idx in range(1, self.config['num_stacks'] + 1):
            for block_idx in range(self.config['num_blocks_%d' % stack_idx]):
                if self.config['weight_sharing_%d' % stack_idx] and block_idx > 0:
                    # for weight sharing, we only create one instance
                    break
                stacks[stack_idx - 1].append(NBEATSBLock(in_features, stack_idx=stack_idx, config=self.config))
        return stacks, stacks[-1][-1].width

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'NBEATSDecoder',
            'name': 'NBEATSDecoder',
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X.update({'backcast_loss_ratio': self.config['backcast_loss_ratio']})
        return super().transform(X)

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
            num_stacks: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="num_stacks",
                value_range=(1, 4),
                default_value=2
            ),
            num_blocks: HyperparameterSearchSpace = HyperparameterSearchSpace(
                'num_blocks',
                value_range=(1, 5),
                default_value=3
            ),
            num_layers: HyperparameterSearchSpace = HyperparameterSearchSpace(
                'num_layers',
                value_range=(1, 5),
                default_value=3
            ),
            width: HyperparameterSearchSpace = HyperparameterSearchSpace(
                'width',
                value_range=(256, 2048),
                default_value=512,
                log=True
            ),
            weight_sharing: HyperparameterSearchSpace = HyperparameterSearchSpace(
                'weight_sharing',
                value_range=(True, False),
                default_value=False,
            ),
            stack_type: HyperparameterSearchSpace = HyperparameterSearchSpace(
                'stack_type',
                value_range=('generic', 'seasonality', 'trend'),
                default_value='generic'),
            expansion_coefficient_length_generic: HyperparameterSearchSpace = HyperparameterSearchSpace(
                'expansion_coefficient_length_generic',
                value_range=(1, 4),
                default_value=3,
            ),
            expansion_coefficient_length_interpretable: HyperparameterSearchSpace = HyperparameterSearchSpace(
                'expansion_coefficient_length_interpretable',
                value_range=(16, 64),
                default_value=32,
                log=True
            ),
            activation: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="activation",
                value_range=tuple(_activations.keys()),
                default_value=list(_activations.keys())[0],
            ),
            use_dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="use_dropout",
                value_range=(True, False),
                default_value=False,
            ),
            normalization: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="normalization",
                value_range=('BN', 'LN', 'NoNorm'),
                default_value='BN'
            ),
            dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="dropout",
                value_range=(0, 0.8),
                default_value=0.5,
            ),
            backcast_loss_ratio: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="backcast_loss_ratio",
                value_range=(0., 1.),
                default_value=1.,
            )
    ) -> ConfigurationSpace:
        """
        Configuration for N-BEATS. The network is composed of several stacks, each stack is composed of several block,
        we follow the implementation from N-BEATS: blocks are only composed of fully-connected layers with the same
        width
        The design of the configuration space follows pytorch-forecasting:
        https://github.com/jdb78/pytorch-forecasting/tree/master/pytorch_forecasting/models/nbeats
        Args:
            dataset_properties:
            num_stacks: number of stacks
            num_blocks: number of blocks per stack
            num_layers: number of fc layers per block, this value is the same across all the blocks within one stack
            width: fc layer width, this value is the same across all the blocks within one stack
            weight_sharing: if weights are shared inside one block
            stack_type: stack type, used to define the final output
            expansion_coefficient_length_generic: expansion_coefficient_length, activate if stack_type is 'generic'
            expansion_coefficient_length_interpretable: expansion_coefficient_length, activate if stack_type is 'trend'
            or 'seasonality' (in this case n_dim is expansion_coefficient_length_interpretable * n_prediciton_steps)
             the expansion coefficient) or trend (in this case, it corresponds to the degree of the polynomial)
            activation: activation function across fc layers
            use_dropout: if dropout is applied
            normalization: if normalization is applied
            dropout: dropout value, if use_dropout is set as True
            backcast_loss_ratio: weight of backcast in comparison to forecast when calculating the loss.
                A weight of 1.0 means that forecast and backcast loss is weighted the same (regardless of backcast and
                forecast lengths). Defaults to 0.0, i.e. no weight.
        Returns:
            Configuration Space
        """

        cs = ConfigurationSpace()
        min_num_stacks, max_num_stacks = num_stacks.value_range

        num_stacks = get_hyperparameter(num_stacks, UniformIntegerHyperparameter)

        add_hyperparameter(cs, activation, CategoricalHyperparameter)
        add_hyperparameter(cs, normalization, CategoricalHyperparameter)
        add_hyperparameter(cs, backcast_loss_ratio, UniformFloatHyperparameter)

        # We can have dropout in the network for
        # better generalization
        use_dropout = get_hyperparameter(use_dropout, CategoricalHyperparameter)
        cs.add_hyperparameters([num_stacks, use_dropout])

        for stack_idx in range(1, int(max_num_stacks) + 1):
            num_blocks_search_space = HyperparameterSearchSpace(hyperparameter='num_blocks_%d' % stack_idx,
                                                                value_range=num_blocks.value_range,
                                                                default_value=num_blocks.default_value,
                                                                log=num_blocks.log)
            num_layers_search_space = HyperparameterSearchSpace(hyperparameter='num_layers_%d' % stack_idx,
                                                                value_range=num_layers.value_range,
                                                                default_value=num_layers.default_value,
                                                                log=num_layers.log)
            width_search_space = HyperparameterSearchSpace(hyperparameter='width_%d' % stack_idx,
                                                           value_range=width.value_range,
                                                           default_value=width.default_value,
                                                           log=width.log)
            weight_sharing_search_sapce = HyperparameterSearchSpace(hyperparameter='weight_sharing_%d' % stack_idx,
                                                                    value_range=weight_sharing.value_range,
                                                                    default_value=weight_sharing.default_value,
                                                                    log=weight_sharing.log)
            stack_type_search_space = HyperparameterSearchSpace(hyperparameter='stack_type_%d' % stack_idx,
                                                                value_range=stack_type.value_range,
                                                                default_value=stack_type.default_value,
                                                                log=stack_type.log)
            expansion_coefficient_length_generic_search_space = HyperparameterSearchSpace(
                hyperparameter='expansion_coefficient_length_generic_%d' % stack_idx,
                value_range=expansion_coefficient_length_generic.value_range,
                default_value=expansion_coefficient_length_generic.default_value,
                log=expansion_coefficient_length_generic.log
            )
            expansion_coefficient_length_interpretable_search_space = HyperparameterSearchSpace(
                hyperparameter='expansion_coefficient_length_interpretable_%d' % stack_idx,
                value_range=expansion_coefficient_length_interpretable.value_range,
                default_value=expansion_coefficient_length_interpretable.default_value,
                log=expansion_coefficient_length_interpretable.log
            )

            num_blocks_hp = get_hyperparameter(num_blocks_search_space, UniformIntegerHyperparameter)
            num_layers_hp = get_hyperparameter(num_layers_search_space, UniformIntegerHyperparameter)
            width_hp = get_hyperparameter(width_search_space, UniformIntegerHyperparameter)
            weight_sharing_hp = get_hyperparameter(weight_sharing_search_sapce, CategoricalHyperparameter)
            stack_type_hp = get_hyperparameter(stack_type_search_space, CategoricalHyperparameter)

            expansion_coefficient_length_generic_hp = get_hyperparameter(
                expansion_coefficient_length_generic_search_space,
                UniformIntegerHyperparameter
            )
            expansion_coefficient_length_interpretable_hp = get_hyperparameter(
                expansion_coefficient_length_interpretable_search_space,
                UniformIntegerHyperparameter
            )

            hps = [num_blocks_hp, num_layers_hp, width_hp, stack_type_hp, weight_sharing_hp]
            cs.add_hyperparameters([*hps, expansion_coefficient_length_generic_hp,
                                    expansion_coefficient_length_interpretable_hp])

            if stack_idx > int(min_num_stacks):
                # The units of layer i should only exist
                # if there are at least i layers
                for hp in hps:
                    cs.add_condition(GreaterThanCondition(hp, num_stacks, stack_idx - 1))
                cond_ecl_generic = AndConjunction(
                    GreaterThanCondition(expansion_coefficient_length_generic_hp, num_stacks, stack_idx -1),
                    EqualsCondition(expansion_coefficient_length_generic_hp, stack_type_hp, 'generic')
                )
                cond_ecl_interpretable = AndConjunction(
                    GreaterThanCondition(expansion_coefficient_length_interpretable_hp, num_stacks, stack_idx - 1),
                    InCondition(expansion_coefficient_length_interpretable_hp, stack_type_hp, ('seasonality', 'trend'))
                )
                cs.add_conditions([cond_ecl_generic, cond_ecl_interpretable])

            dropout_search_space = HyperparameterSearchSpace(hyperparameter='dropout_%d' % stack_idx,
                                                             value_range=dropout.value_range,
                                                             default_value=dropout.default_value,
                                                             log=dropout.log)

            dropout_hp = get_hyperparameter(dropout_search_space, UniformFloatHyperparameter)
            cs.add_hyperparameter(dropout_hp)

            dropout_condition_1 = EqualsCondition(dropout_hp, use_dropout, True)

            if stack_idx > int(min_num_stacks):
                dropout_condition_2 = GreaterThanCondition(dropout_hp, num_stacks, stack_idx - 1)
                cs.add_condition(AndConjunction(dropout_condition_1, dropout_condition_2))
            else:
                cs.add_condition(dropout_condition_1)


        return cs
