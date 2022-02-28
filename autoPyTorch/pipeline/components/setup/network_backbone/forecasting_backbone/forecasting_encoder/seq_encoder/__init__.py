import os
from collections import OrderedDict
from typing import Dict, Optional, List, Any, Union
import numpy as np
from sklearn.pipeline import Pipeline

from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter
)
from ConfigSpace.configuration_space import ConfigurationSpace, Configuration
from ConfigSpace.conditions import (
    EqualsCondition, OrConjunction, GreaterThanCondition, NotEqualsCondition, AndConjunction
)
from ConfigSpace.forbidden import ForbiddenInClause, ForbiddenEqualsClause, ForbiddenAndConjunction

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType

from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    autoPyTorchComponent,
    find_components,
)
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder import \
    AbstractForecastingEncoderChoice

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder. \
    base_forecasting_encoder import BaseForecastingEncoder, ForecastingNetworkStructure

directory = os.path.split(__file__)[0]
_encoders = find_components(__package__,
                            directory,
                            BaseForecastingEncoder)
_addons = ThirdPartyComponents(BaseForecastingEncoder)


def add_encoder(encoder: BaseForecastingEncoder) -> None:
    _addons.add_component(encoder)


class SeqForecastingEncoderChoice(AbstractForecastingEncoderChoice):
    deepAR_decoder_name = 'MLPDecoder'
    deepAR_decoder_prefix = 'block_1'

    def get_components(self) -> Dict[str, autoPyTorchComponent]:
        """Returns the available backbone components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all basebackbone components available
                as choices for learning rate scheduling
        """
        components = OrderedDict()
        components.update(_encoders)
        components.update(_addons.components)
        return components

    def get_hyperparameter_search_space(
            self,
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
            num_blocks: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="num_blocks",
                                                                              value_range=(1, 1),
                                                                              default_value=1),
            variable_selection: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="variable_selection",
                value_range=(True, False),
                default_value=False
            ),
            decoder_auto_regressive: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="decoder_auto_regressive",
                value_range=(True, False),
                default_value=False,
            ),
            skip_connection: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="skip_connection",
                                                                                   value_range=(True, False),
                                                                                   default_value=False),
            skip_connection_type: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="skip_connection_type",
                value_range=("add", "gate_add_norm"),
                default_value="gate_add_norm",
            ),
            grn_use_dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="grn_use_dropout",
                                                                                   value_range=(True, False),
                                                                                   default_value=True),
            grn_dropout_rate: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='grn_dropout_rate',
                                                                                    value_range=(0.0, 0.8),
                                                                                    default_value=0.1),
            default: Optional[str] = None,
            include: Optional[List[str]] = None,
            exclude: Optional[List[str]] = None,
    ) -> ConfigurationSpace:
        """Returns the configuration space of the current chosen components

        Args:
            dataset_properties (Optional[Dict[str, str]]): Describes the dataset to work on
            num_blocks (HyperparameterSearchSpace): number of encoder-decoder structure blocks
            variable_selection (HyperparameterSearchSpace): if variable selection is applied, if True, then the first
                block will be attached with a variable selection block while the following will be enriched with static
                features.
            skip_connection: HyperparameterSearchSpace: if skip connection is applied
            skip_connection_type (HyperparameterSearchSpace): skip connection type, it could be directly added or a grn
                network (
                Lim et al, Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting:
                https://arxiv.org/abs/1912.09363) TODO consider hidden size of grn as a new HP
            grn_use_dropout (HyperparameterSearchSpace): if dropout layer is applied to grn
            grn_dropout_rate (HyperparameterSearchSpace): dropout rate of grn
            decoder_auto_regressive: HyperparameterSearchSpace: if decoder is auto_regressive, e.g., if the decoder
                receives the output as its input, this only works for  auto_regressive decoder models
            default (Optional[str]): Default backbone to use
            include: Optional[Dict[str, Any]]: what components to include. It is an exhaustive
                list, and will exclusively use this components.
            exclude: Optional[Dict[str, Any]]: which components to skip

        Returns:
            ConfigurationSpace: the configuration space of the hyper-parameters of the
                 chosen component
        """
        if dataset_properties is None:
            dataset_properties = {}

        # TODO
        static_features_shape = dataset_properties.get("static_features_shape", 0)
        future_feature_shapes = dataset_properties.get("future_feature_shapes", (0,))

        cs = ConfigurationSpace()

        min_num_blocks, max_num_blocks = num_blocks.value_range

        variable_selection = get_hyperparameter(variable_selection, CategoricalHyperparameter)
        decoder_auto_regressive = get_hyperparameter(decoder_auto_regressive, CategoricalHyperparameter)
        num_blocks = get_hyperparameter(num_blocks, UniformIntegerHyperparameter)

        skip_connection = get_hyperparameter(skip_connection, CategoricalHyperparameter)

        hp_network_structures = [num_blocks, decoder_auto_regressive, variable_selection, skip_connection]
        cond_skip_connections = []
        if True in skip_connection.choices:
            skip_connection_type = get_hyperparameter(skip_connection_type, CategoricalHyperparameter)
            hp_network_structures.append(skip_connection_type)
            cond_skip_connections.append(EqualsCondition(skip_connection_type, skip_connection, True))
            if 'grn' in skip_connection_type.choices:
                grn_use_dropout = get_hyperparameter(grn_use_dropout, CategoricalHyperparameter)
                hp_network_structures.append(grn_use_dropout)
                cond_skip_connections.append(EqualsCondition(grn_use_dropout, skip_connection_type, "grn"))
                if True in grn_use_dropout.choices:
                    grn_dropout_rate = get_hyperparameter(grn_dropout_rate, UniformFloatHyperparameter)
                    hp_network_structures.append(grn_dropout_rate)
                    cond_skip_connections.append(EqualsCondition(grn_dropout_rate, grn_use_dropout, True))

        cs.add_hyperparameters(hp_network_structures)
        if cond_skip_connections:
            cs.add_conditions(cond_skip_connections)

        if static_features_shape + future_feature_shapes[-1] == 0:
            if False in variable_selection.choices and True in decoder_auto_regressive.choices:
                if variable_selection.num_choices == 1 and decoder_auto_regressive.num_choices == 1:
                    raise ValueError("When no future information is available, it is not possible to disable variable"
                                     "selection and enable auto-regressive decoder model")
                cs.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(variable_selection, False),
                    ForbiddenEqualsClause(decoder_auto_regressive, True)
                ))

        # Compile a list of legal preprocessors for this problem
        available_encoders = self.get_available_components(
            dataset_properties=dataset_properties,
            include=include, exclude=exclude)

        available_decoders = self.get_available_components(
            dataset_properties=dataset_properties,
            include=None, exclude=None,
            components=self.get_decoder_components())

        if len(available_encoders) == 0:
            raise ValueError("No Encoder found")
        if len(available_decoders) == 0:
            raise ValueError("No Decoder found")

        if default is None:
            defaults = self._defaults_network
            for default_ in defaults:
                if default_ in available_encoders:
                    default = default_
                    break
        updates_choice = self._get_search_space_updates()

        forbiddens_decoder_auto_regressive = []

        if False in decoder_auto_regressive.choices:
            forbidden_decoder_ar = ForbiddenEqualsClause(decoder_auto_regressive, True)
        else:
            forbidden_decoder_ar = None

        for i in range(1, int(max_num_blocks) + 1):
            block_prefix = f'block_{i}:'

            if '__choice__' in updates_choice.keys():
                choice_hyperparameter = updates_choice['__choice__']
                if not set(choice_hyperparameter.value_range).issubset(available_encoders):
                    raise ValueError("Expected given update for {} to have "
                                     "choices in {} got {}".format(self.__class__.__name__,
                                                                   available_encoders,
                                                                   choice_hyperparameter.value_range))
                hp_encoder = CategoricalHyperparameter(block_prefix + '__choice__',
                                                       choice_hyperparameter.value_range,
                                                       default_value=choice_hyperparameter.default_value)
            else:
                hp_encoder = CategoricalHyperparameter(
                    block_prefix + '__choice__',
                    list(available_encoders.keys()),
                    default_value=default
                )
            cs.add_hyperparameter(hp_encoder)
            if i > int(min_num_blocks):
                cs.add_condition(
                    GreaterThanCondition(hp_encoder, num_blocks, i - 1)
                )

            decoder2encoder = {key: [] for key in available_decoders.keys()}
            encoder2decoder = {}
            for encoder_name in hp_encoder.choices:
                updates = self._get_search_space_updates(prefix=block_prefix + encoder_name)
                config_space = available_encoders[encoder_name].get_hyperparameter_search_space(dataset_properties,
                                                                                                # type: ignore
                                                                                                **updates)
                parent_hyperparameter = {'parent': hp_encoder, 'value': encoder_name}
                cs.add_configuration_space(
                    block_prefix + encoder_name,
                    config_space,
                    parent_hyperparameter=parent_hyperparameter
                )

                allowed_decoders = available_encoders[encoder_name].allowed_decoders()
                if len(allowed_decoders) > 1:
                    if 'decoder_type' not in config_space:
                        raise ValueError('When a specific encoder has more than one allowed decoder, its ConfigSpace'
                                         'must contain the hyperparameter "decoder_type" ! Please check your encoder '
                                         'setting!')
                    hp_decoder_choice = config_space.get_hyperparameter('decoder_type').choices
                    if not set(hp_decoder_choice).issubset(allowed_decoders):
                        raise ValueError(
                            'The encoder hyperparameter decoder_type must be a subset of the allowed_decoders')
                    allowed_decoders = hp_decoder_choice
                for decoder_name in allowed_decoders:
                    decoder2encoder[decoder_name].append(encoder_name)
                encoder2decoder[encoder_name] = allowed_decoders

            for decoder_name in available_decoders.keys():
                if not decoder2encoder[decoder_name]:
                    continue
                updates = self._get_search_space_updates(prefix=block_prefix + decoder_name)
                if i == 1 and decoder_name == self.deepAR_decoder_name:
                    # TODO this is only a temporary solution, a fix on ConfigSpace needs to be implemented
                    updates['can_be_auto_regressive'] = True
                if decoder_name == "MLPDecoder" and i < int(max_num_blocks):
                    updates['has_local_layer'] = HyperparameterSearchSpace('has_local_layer',
                                                                           value_range=(True,),
                                                                           default_value=True)
                config_space = available_decoders[decoder_name].get_hyperparameter_search_space(dataset_properties,
                                                                                                # type: ignore
                                                                                                **updates)
                compatible_encoders = decoder2encoder[decoder_name]
                encoders_with_multi_decoder = []
                encoder_with_single_decoder = []
                for encoder in compatible_encoders:
                    if len(encoder2decoder[encoder]) > 1:
                        encoders_with_multi_decoder.append(encoder)
                    else:
                        encoder_with_single_decoder.append(encoder)

                cs.add_configuration_space(
                    block_prefix + decoder_name,
                    config_space,
                    # parent_hyperparameter=parent_hyperparameter
                )
                if not available_decoders[decoder_name].decoder_properties().recurrent:
                    hp_encoder_choice = cs.get_hyperparameter(block_prefix + '__choice__')
                    for encoder_single in encoder_with_single_decoder:
                        if encoder_single in hp_encoder_choice.choices:
                            forbiddens_decoder_auto_regressive.append(ForbiddenAndConjunction(
                                forbidden_decoder_ar,
                                ForbiddenEqualsClause(hp_encoder_choice, encoder_single)
                            ))
                    for encode_multi in encoders_with_multi_decoder:
                        hp_decoder_type = cs.get_hyperparameter(f"{block_prefix}{encode_multi}:decoder_type")
                        forbiddens_decoder_auto_regressive.append(ForbiddenAndConjunction(
                            forbidden_decoder_ar,
                            ForbiddenEqualsClause(hp_decoder_type, decoder_name)
                        ))

                hps = cs.get_hyperparameters()  # type: List[CSH.Hyperparameter]
                conditions_to_add = []
                for hp in hps:
                    # TODO consider if this will raise any unexpected behavior
                    if hp.name.startswith(block_prefix + decoder_name):
                        # From the implementation of ConfigSpace
                        # Only add a condition if the parameter is a top-level
                        # parameter of the new configuration space (this will be some
                        #  kind of tree structure).
                        if cs.get_parents_of(hp):
                            continue
                        or_cond = []
                        for encoder_single in encoder_with_single_decoder:
                            or_cond.append(EqualsCondition(hp,
                                                           hp_encoder,
                                                           encoder_single))
                        for encode_multi in encoders_with_multi_decoder:
                            hp_decoder_type = cs.get_hyperparameter(f'{block_prefix + encode_multi}:decoder_type')
                            or_cond.append(EqualsCondition(hp, hp_decoder_type, decoder_name))
                        if len(or_cond) == 0:
                            continue
                        elif len(or_cond) > 1:
                            conditions_to_add.append(OrConjunction(*or_cond))
                        else:
                            conditions_to_add.append(or_cond[0])

                cs.add_conditions(conditions_to_add)

        cs.add_forbidden_clauses(forbiddens_decoder_auto_regressive)
        if self.deepAR_decoder_name in available_decoders:
            deep_ar_hp = ':'.join([self.deepAR_decoder_prefix, self.deepAR_decoder_name, 'auto_regressive'])
            if deep_ar_hp in cs:
                deep_ar_hp = cs.get_hyperparameter(deep_ar_hp)
                forbidden_ar = ForbiddenEqualsClause(deep_ar_hp, True)
                if min_num_blocks == 1:
                    if max_num_blocks > 1:
                        if max_num_blocks - min_num_blocks > 1:
                            forbidden = ForbiddenAndConjunction(
                                ForbiddenInClause(num_blocks, list(range(1, max_num_blocks))),
                                forbidden_ar
                            )
                        else:
                            forbidden = ForbiddenAndConjunction(ForbiddenEqualsClause(num_blocks, 2), forbidden_ar)
                        cs.add_forbidden_clause(forbidden)
                if 'RNNEncoder' in available_encoders:
                    for i in range(min_num_blocks, max_num_blocks + 1):
                        rnn_bidirectional_hp = ':'.join([f'block_{min_num_blocks}',
                                                         'RNNEncoder',
                                                         'bidirectional'])
                        if rnn_bidirectional_hp in cs:
                            rnn_bidirectional_hp = cs.get_hyperparameter(rnn_bidirectional_hp)
                            if 'True' in rnn_bidirectional_hp.choices:
                                forbidden = ForbiddenAndConjunction(
                                    ForbiddenEqualsClause(rnn_bidirectional_hp, True),
                                    deep_ar_hp
                                )
                                cs.add_forbidden_clause(forbidden)
        return cs

    def set_hyperparameters(self,
                            configuration: Configuration,
                            init_params: Optional[Dict[str, Any]] = None
                            ) -> 'autoPyTorchChoice':
        """
        Applies a configuration to the given component.
        This method translate a hierarchical configuration key,
        to an actual parameter of the autoPyTorch component.

        Args:
            configuration (Configuration):
                Which configuration to apply to the chosen component
            init_params (Optional[Dict[str, any]]):
                Optional arguments to initialize the chosen component

        Returns:
            self: returns an instance of self
        """
        new_params = {}

        params = configuration.get_dictionary()
        num_blocks = params['num_blocks']
        decoder_auto_regressive = params['decoder_auto_regressive']
        forecasting_structure_kwargs = dict(num_blocks=num_blocks,
                                            variable_selection=params['variable_selection'],
                                            skip_connection=params['skip_connection'])

        del params['num_blocks']
        del params['variable_selection']
        del params['skip_connection']
        del params['decoder_auto_regressive']

        if 'skip_connection_type' in params:
            forecasting_structure_kwargs['skip_connection_type'] = params['skip_connection_type']
            del params['skip_connection_type']
            if 'grn_use_dropout' in params:
                del params['grn_use_dropout']
                if 'grn_dropout_rate' in params:
                    forecasting_structure_kwargs['grn_dropout_rate'] = params['grn_dropout_rate']
                    del params['grn_dropout_rate']
                else:
                    forecasting_structure_kwargs['grn_dropout_rate'] = 0.0

        pipeline_steps = [('net_structure', ForecastingNetworkStructure(**forecasting_structure_kwargs))]
        self.encoder_choice = []
        self.decoder_choice = []

        decoder_components = self.get_decoder_components()

        for i in range(1, num_blocks + 1):
            block_prefix = f'block_{i}:'
            choice = params[block_prefix + '__choice__']
            del params[block_prefix + '__choice__']

            for param, value in params.items():
                if param.startswith(block_prefix):
                    param = param.replace(block_prefix + choice + ':', '')
                    new_params[param] = value

            if init_params is not None:
                for param, value in init_params.items():
                    if param.startswith(block_prefix):
                        param = param.replace(block_prefix + choice + ':', '')
                        new_params[param] = value

            decoder_type = None

            decoder_params = {}
            decoder_params_names = []
            for param, value in new_params.items():
                if decoder_type is None:
                    for decoder_component in decoder_components.keys():
                        if param.startswith(block_prefix + decoder_component):
                            decoder_type = decoder_component
                            decoder_params_names.append(param)
                            param = param.replace(block_prefix + decoder_type + ':', '')
                            decoder_params[param] = value
                else:
                    if param.startswith(block_prefix + decoder_type):
                        decoder_params_names.append(param)
                        param = param.replace(block_prefix + decoder_type + ':', '')
                        decoder_params[param] = value

            for param_name in decoder_params_names:
                del new_params[param_name]
            new_params['random_state'] = self.random_state
            new_params['block_number'] = i
            decoder_params['random_state'] = self.random_state
            decoder_params['block_number'] = i
            # for mlp decoder, to avoid decoder's auto_regressive being overwritten by decoder_auto_regressive
            if 'auto_regressive' not in decoder_params:
                decoder_params['auto_regressive'] = decoder_auto_regressive
            encoder = self.get_components()[choice](**new_params)
            decoder = decoder_components[decoder_type](**decoder_params)
            pipeline_steps.extend([(f'encoder_{i}', encoder), (f'decoder_{i}', decoder)])
            self.encoder_choice.append(encoder)
            self.decoder_choice.append(decoder)

        self.pipeline = Pipeline(pipeline_steps)
        self.choice = self.encoder_choice[0]
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'SeqEncoder',
            'name': 'SeqEncoder',
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }
