import inspect
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import ConfigSpace as CS
from ConfigSpace.conditions import (
    EqualsCondition,
    GreaterThanCondition,
    OrConjunction
)
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.forbidden import (
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    ForbiddenInClause
)
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    Hyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter
)

from sklearn.pipeline import Pipeline

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents, autoPyTorchComponent, find_components)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.components_util import \
    ForecastingNetworkStructure
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder. \
    base_forecasting_decoder import BaseForecastingDecoder
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder import \
    AbstractForecastingEncoderChoice
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder. \
    base_forecasting_encoder import BaseForecastingEncoder
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.other_components. \
    TemporalFusion import TemporalFusion
from autoPyTorch.utils.common import (
    HyperparameterSearchSpace,
    get_hyperparameter
)

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
    tf_prefix = "temporal_fusion"

    def get_components(self) -> Dict[str, Type[autoPyTorchComponent]]:  # type: ignore[override]
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

    def get_hyperparameter_search_space(  # type: ignore[override]
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
            variable_selection_use_dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="variable_selection_use_dropout",
                value_range=(True, False),
                default_value=False,
            ),
            variable_selection_dropout_rate: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="variable_selection_dropout_rate",
                value_range=(0.0, 0.8),
                default_value=0.1,
            ),

            share_single_variable_networks: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="share_single_variable_networks",
                value_range=(True, False),
                default_value=False,
            ),
            use_temporal_fusion: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter='use_temporal_fusion',
                value_range=(True, False),
                default_value=False,
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
            dataset_properties (Optional[Dict[str, str]]):
                Describes the dataset to work on
            num_blocks (int):
                number of encoder-decoder structure blocks
            variable_selection (bool):
                if variable selection is applied, if True, then the first block will be attached with a variable
                 selection block while the following will be enriched with static features.
            variable_selection_use_dropout (bool):
                if variable selection network uses dropout
            variable_selection_dropout_rate (float):
                dropout rate of variable selection network
            share_single_variable_networks (bool):
                if single variable networks are shared between encoder and decoder
            skip_connection (int):
                if skip connection is applied
            use_temporal_fusion (int):
                if temporal fusion layer is applied
            skip_connection_type (str):
                skip connection type, it could be directly added or a GRN network
                (Lim et al, Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting:
                https://arxiv.org/abs/1912.09363) TODO consider hidden size of grn as a new HP
            grn_use_dropout (bool):
                if dropout layer is applied to GRN, since variable selection network also contains GRN,
                this parameter also influence variable selection network
            grn_dropout_rate (float):
                dropout rate of GRN, same as above, this variable also influence variable selection network
            decoder_auto_regressive (int):
                if decoder is auto_regressive, e.g., if the decoder receives the output as its input,
                 this only works for  auto_regressive decoder models
            default (Optional[str]):
                Default backbone to use
            include: Optional[Dict[str, Any]]:
                what components to include. It is an exhaustive list, and will exclusively use this components.
            exclude: Optional[Dict[str, Any]]:
                which components to skip

        Returns:
            ConfigurationSpace: the configuration space of the hyper-parameters of the
                 chosen component
        """
        if dataset_properties is None:
            dataset_properties = {}

        static_features_shape: int = dataset_properties.get("static_features_shape", 0)  # type: ignore[assignment]
        future_feature_shapes: Tuple[int] = dataset_properties.get("future_feature_shapes",  # type: ignore[assignment]
                                                                   (0,))

        cs = ConfigurationSpace()

        min_num_blocks: int = num_blocks.value_range[0]   # type: ignore[assignment]
        max_num_blocks: int = num_blocks.value_range[1]   # type: ignore[assignment]

        variable_selection_hp: CategoricalHyperparameter = get_hyperparameter(  # type: ignore[assignment]
            variable_selection, CategoricalHyperparameter)
        share_single_variable_networks = get_hyperparameter(share_single_variable_networks, CategoricalHyperparameter)

        decoder_auto_regressive_hp: CategoricalHyperparameter = get_hyperparameter(  # type: ignore[assignment]
            decoder_auto_regressive, CategoricalHyperparameter
        )

        if min_num_blocks == max_num_blocks:
            num_blocks = Constant(num_blocks.hyperparameter, num_blocks.value_range[0])
        else:
            num_blocks = OrdinalHyperparameter(
                num_blocks.hyperparameter,
                sequence=list(range(min_num_blocks, max_num_blocks + 1))
            )

        skip_connection_hp: CategoricalHyperparameter = get_hyperparameter(skip_connection,  # type: ignore[assignment]
                                                                           CategoricalHyperparameter)

        hp_network_structures = [num_blocks, decoder_auto_regressive_hp, variable_selection_hp,
                                 skip_connection_hp]
        cond_skip_connections = []

        if True in skip_connection_hp.choices:
            skip_connection_type_hp: CategoricalHyperparameter = get_hyperparameter(  # type: ignore[assignment]
                skip_connection_type, CategoricalHyperparameter
            )
            hp_network_structures.append(skip_connection_type_hp)
            cond_skip_connections.append(EqualsCondition(skip_connection_type_hp, skip_connection_hp, True))
            if 'gate_add_norm' in skip_connection_type_hp.choices:
                grn_use_dropout_hp: CategoricalHyperparameter = get_hyperparameter(  # type: ignore[assignment]
                    grn_use_dropout, CategoricalHyperparameter
                )
                hp_network_structures.append(grn_use_dropout_hp)
                if True in variable_selection_hp.choices:
                    cond_skip_connections.append(
                        EqualsCondition(grn_use_dropout_hp, skip_connection_type_hp, "gate_add_norm")
                    )
                else:
                    cond_skip_connections.append(
                        EqualsCondition(grn_use_dropout_hp, skip_connection_type_hp, "gate_add_norm"))
                if True in grn_use_dropout_hp.choices:
                    grn_dropout_rate_hp = get_hyperparameter(grn_dropout_rate, UniformFloatHyperparameter)
                    hp_network_structures.append(grn_dropout_rate_hp)
                    cond_skip_connections.append(EqualsCondition(grn_dropout_rate_hp, grn_use_dropout_hp, True))
        cs.add_hyperparameters(hp_network_structures)
        if cond_skip_connections:
            cs.add_conditions(cond_skip_connections)

        if True in variable_selection_hp.choices:
            variable_selection_use_dropout_hp = get_hyperparameter(variable_selection_use_dropout,
                                                                   CategoricalHyperparameter)
            variable_selection_dropout_rate_hp = get_hyperparameter(variable_selection_dropout_rate,
                                                                    UniformFloatHyperparameter)
            cs.add_hyperparameters([variable_selection_use_dropout_hp, variable_selection_dropout_rate_hp])

            cond_vs_dropout = EqualsCondition(variable_selection_use_dropout_hp, variable_selection_hp, True)
            cond_vs_dropoutrate = EqualsCondition(variable_selection_dropout_rate_hp,
                                                  variable_selection_use_dropout_hp,
                                                  True)
            cs.add_conditions([cond_vs_dropout, cond_vs_dropoutrate])

        if True in variable_selection_hp.choices:
            cs.add_hyperparameter(share_single_variable_networks)
            cs.add_condition(EqualsCondition(share_single_variable_networks, variable_selection_hp, True))

        # Compile a list of legal preprocessors for this problem
        available_encoders: Dict[str, BaseForecastingEncoder] = self.get_available_components(  # type: ignore
            dataset_properties=dataset_properties,
            include=include, exclude=exclude)

        available_decoders: Dict[str, BaseForecastingDecoder] = self.get_available_components(  # type: ignore
            dataset_properties=dataset_properties,
            include=None, exclude=exclude,
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

        # TODO this is only a temporary solution, needs to be updated when ConfigSpace allows more complex conditions!
        # General Idea to work with auto-regressive decoders:
        # decoder cannot be auto-regressive if it is not recurrent
        #   decoder_auto_regressive is conditioned on the HPs that allow recurrent decoders:
        #     encoders that only have recurrent decoders -> EqCond(dar, encoder, en_name)
        #     decoder_types of Encoders that contain recurrent decoders -> EqCond(dar, encoder:de_type, de_name)
        #
        # When no future data can be fed to the decoder (no future features), decoder must be auto-regressive:
        #   disable the recurrent decoders without auto-regressive or variable selection
        #   this is judged by add_forbidden_for_non_ar_recurrent_decoder

        if True in decoder_auto_regressive_hp.choices:
            forbidden_decoder_ar: Optional[ForbiddenEqualsClause] = ForbiddenEqualsClause(decoder_auto_regressive_hp,
                                                                                          True)
        else:
            forbidden_decoder_ar = None

        add_forbidden_for_non_ar_recurrent_decoder = False
        if static_features_shape + future_feature_shapes[-1] == 0:
            if False in decoder_auto_regressive_hp.choices and False in variable_selection_hp.choices:
                add_forbidden_for_non_ar_recurrent_decoder = True

        if len(decoder_auto_regressive_hp.choices) == 1 and True in decoder_auto_regressive_hp.choices:
            conds_decoder_ar: Optional[List[CS.conditions.ConditionComponent]] = None
        else:
            conds_decoder_ar = []

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
            if conds_decoder_ar is None:
                # In this case we only allow encoders that has recurrent decoders
                available_encoders_w_recurrent_decoder = []
                for encoder_name in hp_encoder.choices:
                    decoders = available_encoders[encoder_name].allowed_decoders()
                    for decoder_name in decoders:
                        if available_decoders[decoder_name].decoder_properties().recurrent:
                            available_encoders_w_recurrent_decoder.append(encoder_name)
                            break
                if not available_encoders_w_recurrent_decoder:
                    raise ValueError('If only auto-regressive decoder is allowed, at least one encoder must contain '
                                     'recurrent decoder!')
                hp_encoder = CategoricalHyperparameter(
                    block_prefix + '__choice__',
                    available_encoders_w_recurrent_decoder,
                    default_value=available_encoders_w_recurrent_decoder[0])

            cs.add_hyperparameter(hp_encoder)
            if i > int(min_num_blocks):
                cs.add_condition(
                    GreaterThanCondition(hp_encoder, num_blocks, i - 1)
                )

            decoder2encoder: Dict[str, List[str]] = {key: [] for key in available_decoders.keys()}
            encoder2decoder = {}
            for encoder_name in hp_encoder.choices:
                updates = self._get_search_space_updates(prefix=block_prefix + encoder_name)
                config_space = available_encoders[encoder_name].get_hyperparameter_search_space(  # type: ignore
                    dataset_properties,
                    **updates  # type: ignore[call-arg]
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
                    recurrent_decoders = []
                    for decoder_name in allowed_decoders:
                        if available_decoders[decoder_name].decoder_properties().recurrent:
                            recurrent_decoders.append(decoder_name)
                    if conds_decoder_ar is None:
                        if recurrent_decoders:
                            updates['decoder_type'] = HyperparameterSearchSpace('decoder_type',
                                                                                tuple(recurrent_decoders),
                                                                                recurrent_decoders[0]
                                                                                )
                            ecd = available_encoders[encoder_name]
                            config_space = ecd.get_hyperparameter_search_space(  # type:ignore
                                dataset_properties,
                                **updates  # type: ignore[call-arg]
                            )
                            hp_decoder_choice = recurrent_decoders
                        else:
                            cs.add_forbidden_clause(ForbiddenEqualsClause(hp_encoder, encoder_name))

                    allowed_decoders = hp_decoder_choice
                valid_decoders = []
                for decoder_name in allowed_decoders:
                    if decoder_name in decoder2encoder:
                        valid_decoders.append(decoder_name)
                        decoder2encoder[decoder_name].append(encoder_name)
                encoder2decoder[encoder_name] = allowed_decoders
                if len(allowed_decoders) > 1:

                    if len(valid_decoders) < len(config_space.get_hyperparameter('decoder_type').choices):
                        updates['decoder_type'] = HyperparameterSearchSpace(hyperparameter='decoder_type',
                                                                            value_range=tuple(valid_decoders),
                                                                            default_value=valid_decoders[0])
                        config_space = available_encoders[encoder_name].get_hyperparameter_search_space(  # type:ignore
                            dataset_properties,
                            **updates  # type: ignore[call-arg]
                        )
                parent_hyperparameter = {'parent': hp_encoder, 'value': encoder_name}
                cs.add_configuration_space(
                    block_prefix + encoder_name,
                    config_space,
                    parent_hyperparameter=parent_hyperparameter
                )

            for decoder_name in available_decoders.keys():
                if not decoder2encoder[decoder_name]:
                    continue
                updates = self._get_search_space_updates(prefix=block_prefix + decoder_name)
                if i == 1 and decoder_name == self.deepAR_decoder_name:
                    # TODO this is only a temporary solution, a fix on ConfigSpace needs to be implemented
                    updates['can_be_auto_regressive'] = True  # type: ignore[assignment]

                config_space = available_decoders[decoder_name].get_hyperparameter_search_space(  # type: ignore
                    dataset_properties,
                    **updates  # type: ignore[call-arg]
                )
                compatible_encoders = decoder2encoder[decoder_name]
                encoders_with_multi_decoder_l = []
                encoder_with_single_decoder_l = []
                for encoder in compatible_encoders:
                    if len(encoder2decoder[encoder]) > 1:
                        encoders_with_multi_decoder_l.append(encoder)
                    else:
                        encoder_with_single_decoder_l.append(encoder)
                encoders_with_multi_decoder = set(encoders_with_multi_decoder_l)
                encoder_with_single_decoder = set(encoder_with_single_decoder_l)

                cs.add_configuration_space(
                    block_prefix + decoder_name,
                    config_space,
                    # parent_hyperparameter=parent_hyperparameter
                )

                hps = cs.get_hyperparameters()  # type: List[Hyperparameter]
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

            if conds_decoder_ar is not None or forbidden_decoder_ar is not None:
                forbiddens_ar_non_recurrent: List[CS.forbidden.AbstractForbiddenClause] = []
                for encoder in hp_encoder.choices:
                    if len(encoder2decoder[encoder]) == 1:
                        if available_decoders[encoder2decoder[encoder][0]].decoder_properties().recurrent:
                            # conds_decoder_ar is not None: False can be in decoder_auto_regressive. In this case,
                            # if hp_encoder selects encoder, then decoder_auto_regressive becomes inactiavte
                            # (indicates a default decoder_auto_regressive=False, thus we need to add another
                            # forbidden incase add_forbidden_for_non_ar_recurrent_decoder is required)
                            # forbidden_decoder_ar is not None: only False in decoder_auto_regressive
                            # add_forbidden_for_non_ar_recurrent_decoder is True:False in decoder_auto_regressive
                            if conds_decoder_ar is not None:
                                conds_decoder_ar.append(
                                    EqualsCondition(decoder_auto_regressive_hp, hp_encoder, encoder)
                                )
                                if add_forbidden_for_non_ar_recurrent_decoder:
                                    forbiddens_decoder_auto_regressive.append(
                                        ForbiddenAndConjunction(
                                            ForbiddenEqualsClause(variable_selection_hp, False),
                                            ForbiddenEqualsClause(hp_encoder, encoder)
                                        )
                                    )
                            else:
                                if add_forbidden_for_non_ar_recurrent_decoder:
                                    forbiddens_decoder_auto_regressive.append(
                                        ForbiddenAndConjunction(
                                            ForbiddenAndConjunction(
                                                ForbiddenEqualsClause(variable_selection_hp, False),
                                                ForbiddenEqualsClause(decoder_auto_regressive_hp, False)
                                            ),
                                            ForbiddenEqualsClause(hp_encoder, encoder)
                                        )
                                    )

                    elif len(encoder2decoder[encoder]) > 1:
                        hp_decoder_type = cs.get_hyperparameter(f'{block_prefix + encoder}:decoder_type')
                        for decoder in hp_decoder_type.choices:
                            if not available_decoders[decoder].decoder_properties().recurrent:
                                # TODO this is a temporary solution: currently ConfigSpace is not able to correctly
                                # activate/deactivate a complex nested configspace; Too many forbiddens might also rise
                                # errors. Thus we only allow decoder_ar to be conditioned on the top layer hps and
                                # put forbiddenclauses here
                                if forbidden_decoder_ar is not None:
                                    forbiddens_decoder_auto_regressive.append(
                                        ForbiddenAndConjunction(
                                            forbidden_decoder_ar,
                                            ForbiddenEqualsClause(hp_decoder_type, decoder)
                                        )
                                    )
                            else:
                                if add_forbidden_for_non_ar_recurrent_decoder:
                                    forbiddens_decoder_auto_regressive.append(
                                        ForbiddenAndConjunction(
                                            ForbiddenAndConjunction(
                                                ForbiddenEqualsClause(variable_selection_hp, False),
                                                ForbiddenEqualsClause(decoder_auto_regressive_hp, False)
                                            ),
                                            ForbiddenEqualsClause(hp_decoder_type, decoder)
                                        )
                                    )

                    if forbiddens_ar_non_recurrent:
                        cs.add_forbidden_clauses(forbiddens_ar_non_recurrent)
        if conds_decoder_ar:
            cs.add_condition(OrConjunction(*conds_decoder_ar))

        use_temporal_fusion_hp = get_hyperparameter(use_temporal_fusion, CategoricalHyperparameter)
        cs.add_hyperparameter(use_temporal_fusion_hp)
        if True in use_temporal_fusion_hp.choices:
            update = self._get_search_space_updates(prefix=self.tf_prefix)
            cs_tf = TemporalFusion.get_hyperparameter_search_space(dataset_properties,
                                                                   **update)
            parent_hyperparameter = {'parent': use_temporal_fusion_hp, 'value': True}
            cs.add_configuration_space(
                self.tf_prefix,
                cs_tf,
                parent_hyperparameter=parent_hyperparameter
            )

        for encoder_name, encoder in available_encoders.items():
            encoder_is_casual = encoder.encoder_properties().is_casual
            if not encoder_is_casual:
                # we do not allow non-casual encoder to appear in the lower layer of the network. e.g, if we have an
                # encoder with 3 blocks, then non_casual encoder is only allowed to appear in the third layer
                for i in range(max(min_num_blocks, 2), max_num_blocks + 1):
                    for j in range(1, i):
                        choice_hp = cs.get_hyperparameter(f"block_{j}:__choice__")
                        if encoder_name in choice_hp.choices:
                            forbidden_encoder_uncasual = [ForbiddenEqualsClause(num_blocks, i),
                                                          ForbiddenEqualsClause(choice_hp, encoder_name)]
                            if forbidden_decoder_ar is not None:
                                forbidden_encoder_uncasual.append(forbidden_decoder_ar)
                            forbiddens_decoder_auto_regressive.append(
                                ForbiddenAndConjunction(*forbidden_encoder_uncasual)
                            )

        cs.add_forbidden_clauses(forbiddens_decoder_auto_regressive)

        if self.deepAR_decoder_name in available_decoders:
            deep_ar_hp_name = ':'.join([self.deepAR_decoder_prefix, self.deepAR_decoder_name, 'auto_regressive'])
            if deep_ar_hp_name in cs:
                deep_ar_hp = cs.get_hyperparameter(deep_ar_hp_name)
                if True in deep_ar_hp.choices:
                    forbidden_deep_ar = ForbiddenEqualsClause(deep_ar_hp, True)
                    if min_num_blocks == 1:
                        if max_num_blocks > 1:
                            forbidden = ForbiddenAndConjunction(
                                ForbiddenInClause(num_blocks, list(range(2, max_num_blocks + 1))),
                                forbidden_deep_ar
                            )
                            cs.add_forbidden_clause(forbidden)
                    else:
                        cs.add_forbidden_clause(forbidden_deep_ar)

                    forbidden_deep_ars = []

                    hps_forbidden_deep_ar = [use_temporal_fusion_hp]
                    for hp_forbidden_deep_ar in hps_forbidden_deep_ar:
                        if True in hp_forbidden_deep_ar.choices:
                            forbidden_deep_ars.append(ForbiddenAndConjunction(
                                ForbiddenEqualsClause(hp_forbidden_deep_ar, True),
                                forbidden_deep_ar
                            ))
                    if True in skip_connection_hp.choices:
                        forbidden_deep_ars.append(ForbiddenAndConjunction(
                            ForbiddenEqualsClause(skip_connection_hp, True),
                            forbidden_deep_ar
                        ))
                    if forbidden_deep_ars:
                        cs.add_forbidden_clauses(forbidden_deep_ars)

        forbidden_mlp_local_layer = []
        for i in range(1, max_num_blocks + 1):
            hp_mlp_has_local_layer = f"block_{i}:MLPDecoder:has_local_layer"
            if hp_mlp_has_local_layer in cs:
                hp_mlp_has_local_layer = cs.get_hyperparameter(hp_mlp_has_local_layer)
                if i < max_num_blocks:
                    forbidden_mlp_local_layer.append(ForbiddenAndConjunction(
                        ForbiddenEqualsClause(hp_mlp_has_local_layer, False),
                        ForbiddenInClause(num_blocks, list(range(i + 1, max_num_blocks + 1))),
                    ))
                c1 = isinstance(skip_connection_hp, CategoricalHyperparameter) and True in skip_connection_hp.choices
                c2 = isinstance(skip_connection_hp, Constant) and skip_connection_hp.value
                if c1 or c2:
                    if True in skip_connection_hp.choices:
                        forbidden_mlp_local_layer.append(ForbiddenAndConjunction(
                            ForbiddenEqualsClause(hp_mlp_has_local_layer, False),
                            ForbiddenEqualsClause(skip_connection_hp, True),
                        ))
                c1 = isinstance(
                    use_temporal_fusion_hp, CategoricalHyperparameter
                ) and True in use_temporal_fusion_hp.choices
                c2 = isinstance(use_temporal_fusion_hp, Constant) and skip_connection_hp.value
                if c1 or c2:
                    if True in use_temporal_fusion_hp.choices:
                        forbidden_mlp_local_layer.append(ForbiddenAndConjunction(
                            ForbiddenEqualsClause(hp_mlp_has_local_layer, False),
                            ForbiddenEqualsClause(use_temporal_fusion_hp, True),
                        ))

        cs.add_forbidden_clauses(forbidden_mlp_local_layer)
        return cs

    @property
    def _defaults_network(self) -> List[str]:
        return ['RNNEncoder', 'NBEATSEncoder']

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

        params = configuration.get_dictionary()
        decoder_auto_regressive = params.pop('decoder_auto_regressive', False)
        net_structure_default_kwargs = inspect.signature(ForecastingNetworkStructure.__init__).parameters

        forecasting_structure_kwargs = {
            key: params.pop(key, value.default) for key, value in net_structure_default_kwargs.items()
            if key != 'self'
        }
        if not params.pop('grn_use_dropout', False):
            forecasting_structure_kwargs['grn_dropout_rate'] = 0.0

        num_blocks = forecasting_structure_kwargs['num_blocks']
        use_temporal_fusion = forecasting_structure_kwargs['use_temporal_fusion']

        pipeline_steps = [('net_structure', ForecastingNetworkStructure(**forecasting_structure_kwargs))]
        self.encoder_choice: Union[List[BaseForecastingEncoder], List[()]] = []
        self.decoder_choice: Union[List[BaseForecastingDecoder], List[()]] = []

        decoder_components = self.get_decoder_components()

        for i in range(1, num_blocks + 1):
            new_params = {}

            block_prefix = f'block_{i}:'
            choice = params.pop(block_prefix + '__choice__')

            for param, value in params.items():
                if param.startswith(block_prefix):
                    param = param.replace(block_prefix + choice + ':', '')
                    new_params[param] = value

            if init_params is not None:
                for param, value in init_params.items():
                    if param.startswith(block_prefix):
                        param = param.replace(block_prefix + choice + ':', '')
                        new_params[param] = value

            decoder_type: Optional[str] = None

            decoder_params = {}
            decoder_params_names = []
            for param, value in new_params.items():
                if decoder_type is None:
                    for decoder_component in decoder_components.keys():
                        if param.startswith(block_prefix + decoder_component):
                            decoder_type: str = decoder_component  # type:ignore[no-redef]
                            decoder_params_names.append(param)
                            param = param.replace(block_prefix + decoder_type + ':', '')  # type:ignore[operator]
                            decoder_params[param] = value
                else:
                    if param.startswith(block_prefix + decoder_type):
                        decoder_params_names.append(param)
                        param = param.replace(block_prefix + decoder_type + ':', '')
                        decoder_params[param] = value
            assert decoder_type is not None, 'Decoder must be given to initialize a forecasting backbone!'

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

        new_params = {}
        if use_temporal_fusion:
            for param, value in params.items():
                if param.startswith(self.tf_prefix):
                    param = param.replace(self.tf_prefix + ':', '')
                    new_params[param] = value
            temporal_fusion = TemporalFusion(self.random_state,
                                             **new_params)
            pipeline_steps.extend([('temporal_fusion', temporal_fusion)])

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
