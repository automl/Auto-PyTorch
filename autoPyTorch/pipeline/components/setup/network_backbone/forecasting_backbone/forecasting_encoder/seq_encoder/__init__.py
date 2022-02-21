import os
from collections import OrderedDict
from typing import Dict, Optional, List, Any, Union
import numpy as np
from sklearn.pipeline import Pipeline

from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.configuration_space import ConfigurationSpace, Configuration
from ConfigSpace.conditions import (
    EqualsCondition, OrConjunction, GreaterThanCondition, NotEqualsCondition, AndConjunction
)

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType

from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    autoPyTorchComponent,
    find_components,
)
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.setup.network_backbone import NetworkBackboneChoice
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.base_forecasting_encoder import (
    BaseForecastingEncoder,
)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder import \
    decoders, decoder_addons, add_decoder
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder import \
    AbstractForecastingEncoderChoice

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder. \
    base_forecasting_encoder import BaseForecastingEncoder

directory = os.path.split(__file__)[0]
_encoders = find_components(__package__,
                            directory,
                            BaseForecastingEncoder)
_addons = ThirdPartyComponents(BaseForecastingEncoder)


def add_encoder(encoder: BaseForecastingEncoder) -> None:
    _addons.add_component(encoder)


class ForecastingNetworkStructure(autoPyTorchComponent):
    def __init__(self, random_state: Optional[np.random.RandomState] = None,
                 num_blocks: int = 1,
                 variable_selection: bool = False,
                 skip_connection: bool = False,
                 auto_regressive: str = 'not_applied') -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.random_state = random_state
        self.variable_selection = variable_selection
        self.skip_connection = skip_connection
        self.auto_regressive = auto_regressive

    def fit(self, X: Dict[str, Any], y: Any = None) -> "ForecastingNetworkStructure":
        self.check_requirements(X, y)
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X.update({
            'num_blocks': self.num_blocks,
            'variable_selection': self.variable_selection,
            'skip_connection': self.skip_connection,
            'auto_regressive': self.auto_regressive,
        })
        return X

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
            **kwargs: Any
    ) -> ConfigurationSpace:
        return ConfigurationSpace()

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'EarlyPreprocessing',
            'name': 'Early Preprocessing Node',
        }

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.__class__.__name__
        return string


class SeqForecastingEncoderChoice(AbstractForecastingEncoderChoice):
    deepAR_decoder_name = 'MLPDecoder'
    deepAR_decoder_prefix = 'deepAR_decoder'
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
                                                                              value_range=(1, 2),
                                                                              default_value=1),
            variable_selection: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="variable_selection",
                value_range=(True, False),
                default_value=False
            ),
            skip_connection: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="skip_connection",
                                                                                   value_range=(True, False),
                                                                                   default_value=False),
            auto_regressive: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="auto_regressive",
                value_range=('encoder', 'decoder'),
                default_value='encoder',
            ),
            default: Optional[str] = None,
            include: Optional[List[str]] = None,
            exclude: Optional[List[str]] = None,
    ) -> ConfigurationSpace:
        """Returns the configuration space of the current chosen components

        Args:
            dataset_properties (Optional[Dict[str, str]]): Describes the dataset to work on
            num_blocks: HyperparameterSearchSpace: number of encoder-decoder structure blocks
            variable_selection: HyperparameterSearchSpace: if variable selection is applied, if True, then the first
            block will be attached with a variable selection block while the following will be enriched with static
            features.
            skip_connection: HyperparameterSearchSpace: if skip connection is applied
            auto_regressive: HyperparameterSearchSpace: if auto-regressive is applied, depending on the choice,
            auto-regressive strategy is applied to either encoder (DeepAR model), decoder or not applied (in which case
            )
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

        cs = ConfigurationSpace()
        add_hyperparameter(cs, variable_selection, CategoricalHyperparameter)
        add_hyperparameter(cs, skip_connection, CategoricalHyperparameter)

        min_num_blocks, max_num_blcoks = num_blocks.value_range

        num_blocks = get_hyperparameter(num_blocks, UniformIntegerHyperparameter)
        auto_regressive = get_hyperparameter(auto_regressive, CategoricalHyperparameter)
        cs.add_hyperparameters([num_blocks, auto_regressive])

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

        for i in range(1, int(max_num_blcoks) + 1):
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

            if len(auto_regressive.choices) > 1 or auto_regressive.choices[0] != 'encoder':
                for decoder_name in available_decoders.keys():
                    if not decoder2encoder[decoder_name]:
                        continue
                    updates = self._get_search_space_updates(prefix=block_prefix + decoder_name)
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
                            if len(or_cond) == 1:
                                conditions_to_add.append(
                                    AndConjunction(or_cond[0],
                                                   NotEqualsCondition(hp, auto_regressive, 'encoder'))
                                )
                            elif len(or_cond) > 1:
                                conditions_to_add.append(
                                    AndConjunction(OrConjunction(*or_cond),
                                                   NotEqualsCondition(hp, auto_regressive, 'encoder'))
                                )

                    cs.add_conditions(conditions_to_add)

        if 'encoder' in auto_regressive.choices:
            decoder_name = self.deepAR_decoder_name
            if decoder_name in available_decoders:
                updates = self._get_search_space_updates(prefix=self.deepAR_decoder_prefix + decoder_name)
                updates['auto_regressive'] = True
                config_space = available_decoders[decoder_name].get_hyperparameter_search_space(dataset_properties,
                                                                                                # type: ignore
                                                                                                **updates)
                parent_hyperparameter = {'parent': auto_regressive, 'value': 'encoder'}
                cs.add_configuration_space(
                    self.deepAR_decoder_prefix + decoder_name,
                    config_space,
                    parent_hyperparameter=parent_hyperparameter
                )

            self.configuration_space_ = cs
            self.dataset_properties_ = dataset_properties

        cs.get_parent_conditions_of('block_1:MLPDecoder:has_local_layer')

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
        variable_selection = params['variable_selection']
        skip_connection = params['skip_connection']
        auto_regressive = params['auto_regressive']
        del params['num_blocks']
        del params['variable_selection']
        del params['skip_connection']

        pipeline_steps = [('net_structure', ForecastingNetworkStructure(random_state=self.random_state,
                                                                        num_blocks=num_blocks,
                                                                        variable_selection=variable_selection,
                                                                        skip_connection=skip_connection,
                                                                        auto_regressive=auto_regressive,))]
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

            if auto_regressive != 'encoder':
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
                decoder_params['random_state'] = self.random_state
                encoder = self.get_components()[choice](**new_params)
                decoder = decoder_components[decoder_type](**decoder_params)
                pipeline_steps.extend([(f'encoder_{i}', encoder), (f'decoder_{i}', decoder)])
                self.encoder_choice.append(encoder)
                self.decoder_choice.append(decoder)
            else:
                new_params['random_state'] = self.random_state
                encoder = self.get_components()[choice](**new_params)
                pipeline_steps.extend([(f'encoder_{i}', encoder)])
                self.encoder_choice.append(encoder)

        if auto_regressive == 'encoder':
            decoder_params = {}
            for param, value in new_params.items():
                if param.startswith(self.deepAR_decoder_prefix + self.deepAR_decoder_name):
                    param = param.replace(self.deepAR_decoder_prefix + self.deepAR_decoder_name + ':', '')
                    decoder_params[param] = value
            decoder_params['random_state'] = self.random_state
            decoder = decoder_components[self.deepAR_decoder_name](**decoder_params)
            pipeline_steps.extend([(f'decoder', decoder)])
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
