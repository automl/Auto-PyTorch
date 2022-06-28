import os
import warnings
from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Type, Union

import ConfigSpace.hyperparameters as CSH
from ConfigSpace.conditions import EqualsCondition, OrConjunction
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

from sklearn.pipeline import Pipeline

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    autoPyTorchComponent,
    find_components
)
from autoPyTorch.pipeline.components.setup.network_backbone import NetworkBackboneChoice
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.components_util import \
    ForecastingNetworkStructure
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder import (
    decoder_addons, decoders)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder.\
    base_forecasting_decoder import BaseForecastingDecoder
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.\
    base_forecasting_encoder import BaseForecastingEncoder

directory = os.path.split(__file__)[0]
_encoders = find_components(__package__,
                            directory,
                            BaseForecastingEncoder)
_addons = ThirdPartyComponents(BaseForecastingEncoder)


class AbstractForecastingEncoderChoice(autoPyTorchChoice):
    """
    A network is composed of an encoder and decoder. In most of the case, the choice of decoder is heavily dependent on
    the choice of encoder. Thus here "choice" indicates the choice of encoder, then decoder will be determined by
    the encoder.
    """

    def __init__(self,
                 **kwargs: Any,
                 ):
        super().__init__(**kwargs)
        self.pipeline: Optional[Pipeline] = None
        self.decoder_choice: Optional[List[BaseForecastingDecoder]] = None

    @abstractmethod
    def get_components(self) -> Dict[str, Type[autoPyTorchComponent]]:  # type: ignore[override]
        """Returns the available backbone components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all basebackbone components available
                as choices for learning rate scheduling
        """
        raise NotImplementedError

    def get_decoder_components(self) -> Dict[str, Type[autoPyTorchComponent]]:
        components = OrderedDict()
        components.update(decoders)
        components.update(decoder_addons.components)
        return components

    @property
    def additional_components(self) -> List[Callable]:
        # This function is deigned to add additional components rather than the components in __choice__
        return [self.get_decoder_components]

    def get_available_components(  # type: ignore[override]
            self,
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
            include: List[str] = None,
            exclude: List[str] = None,
            components: Optional[Dict[str, Type[autoPyTorchComponent]]] = None
    ) -> Dict[str, Type[autoPyTorchComponent]]:
        """Filters out components based on user provided
        include/exclude directives, as well as the dataset properties

        Args:
         include (Optional[Dict[str, Any]]): what hyper-parameter configurations
            to honor when creating the configuration space
         exclude (Optional[Dict[str, Any]]): what hyper-parameter configurations
             to remove from the configuration space
         dataset_properties (Optional[Dict[str, Union[str, int]]]): Caracteristics
             of the dataset to guide the pipeline choices of components
         components (Optional[Dict[str, Type[autoPyTorchComponent]]]): components

        Returns:
            Dict[str, autoPyTorchComponent]: A filtered dict of learning
                rate backbones

        """
        if dataset_properties is None:
            dataset_properties = {}

        if include is not None and exclude is not None:
            raise ValueError(
                "The argument include and exclude cannot be used together.")

        if components is None:
            available_comp = self.get_components()
        else:
            available_comp = components

        if include is not None:
            for incl in include:
                if incl not in available_comp:
                    warnings.warn("Trying to include unknown component: ""%s" % incl)

        components_dict = OrderedDict()
        for name in available_comp:
            if include is not None and name not in include:
                continue
            elif exclude is not None and name in exclude:
                continue

            entry = available_comp[name]

            # Exclude itself to avoid infinite loop
            if entry == NetworkBackboneChoice or hasattr(entry, 'get_components'):
                continue

            task_type = str(dataset_properties['task_type'])
            properties = entry.get_properties()
            if 'tabular' in task_type and not bool(properties['handles_tabular']):
                continue
            elif 'image' in task_type and not bool(properties['handles_image']):
                continue
            elif 'time_series' in task_type and not bool(properties['handles_time_series']):
                continue

            # target_type = dataset_properties['target_type']
            # Apply some automatic filtering here for
            # backbones based on the dataset!
            # TODO: Think if there is any case where a backbone
            # is not recommended for a certain dataset

            components_dict[name] = entry

        return components_dict

    def get_hyperparameter_search_space(
            self,
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
            default: Optional[str] = None,
            include: Optional[List[str]] = None,
            exclude: Optional[List[str]] = None,
    ) -> ConfigurationSpace:
        """Returns the configuration space of the current chosen components

        Args:
            dataset_properties (Optional[Dict[str, str]]):
                Describes the dataset to work on
            default (Optional[str]):
                Default encoder to use
            include: Optional[Dict[str, Any]]:
                what components to include. It is an exhaustive list, and will exclusively use this components. It
                allows nested encoder such as flat_encoder:MLPEncoder
            exclude: Optional[Dict[str, Any]]:
                which components to skip. It allows nested encoder as such flat_encoder:MLPEncoder

        Returns:
            ConfigurationSpace: the configuration space of the hyper-parameters of the
                 chosen component
        """
        cs = ConfigurationSpace()

        if dataset_properties is None:
            dataset_properties = {}

        # Compile a list of legal components for this problem
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
        updates = self._get_search_space_updates()
        if '__choice__' in updates.keys():
            choice_hyperparameter = updates['__choice__']
            if not set(choice_hyperparameter.value_range).issubset(available_encoders):
                raise ValueError("Expected given update for {} to have "
                                 "choices in {} got {}".format(self.__class__.__name__,
                                                               available_encoders,
                                                               choice_hyperparameter.value_range))
            hp_encoder = CSH.CategoricalHyperparameter('__choice__',
                                                       choice_hyperparameter.value_range,
                                                       default_value=choice_hyperparameter.default_value)
        else:
            hp_encoder = CSH.CategoricalHyperparameter(
                '__choice__',
                list(available_encoders.keys()),
                default_value=default
            )
        cs.add_hyperparameter(hp_encoder)

        decoder2encoder: Dict[str, List[str]] = {key: [] for key in available_decoders.keys()}
        encoder2decoder: Dict[str, List[str]] = {}
        for encoder_name in hp_encoder.choices:
            updates = self._get_search_space_updates(prefix=encoder_name)
            config_space = available_encoders[encoder_name].get_hyperparameter_search_space(  # type: ignore[call-arg]
                dataset_properties,
                **updates   # type: ignore[call-arg]
            )
            parent_hyperparameter = {'parent': hp_encoder, 'value': encoder_name}
            cs.add_configuration_space(
                encoder_name,
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
                    raise ValueError('The encoder hyperparameter decoder_type must be a subset of the allowed_decoders')
                allowed_decoders = hp_decoder_choice
            for decoder_name in allowed_decoders:
                decoder2encoder[decoder_name].append(encoder_name)
            encoder2decoder[encoder_name] = allowed_decoders

        for decoder_name in available_decoders.keys():
            if not decoder2encoder[decoder_name]:
                continue
            updates = self._get_search_space_updates(prefix=decoder_name)
            config_space = available_decoders[decoder_name].get_hyperparameter_search_space(  # type: ignore[call-arg]
                dataset_properties,
                **updates   # type: ignore[call-arg]
            )
            compatible_encoders = decoder2encoder[decoder_name]
            encoders_with_multi_decoder = []
            encoder_with_uni_decoder = []

            for encoder in compatible_encoders:
                if len(encoder2decoder[encoder]) > 1:
                    encoders_with_multi_decoder.append(encoder)
                else:
                    encoder_with_uni_decoder.append(encoder)

            cs.add_configuration_space(
                decoder_name,
                config_space,
                # parent_hyperparameter=parent_hyperparameter
            )
            hps = cs.get_hyperparameters()  # type: List[CSH.Hyperparameter]
            conditions_to_add = []
            for hp in hps:
                # TODO consider if this will raise any unexpected behavior
                if hp.name.startswith(decoder_name):
                    # From the implementation of ConfigSpace
                    # Only add a condition if the parameter is a top-level
                    # parameter of the new configuration space (this will be some
                    #  kind of tree structure).
                    if cs.get_parents_of(hp):
                        continue
                    or_cond = []
                    for encoder_uni in encoder_with_uni_decoder:
                        or_cond.append(EqualsCondition(hp,
                                                       hp_encoder,
                                                       encoder_uni))
                    for encode_multi in encoders_with_multi_decoder:
                        hp_decoder_type = cs.get_hyperparameter(f'{encode_multi}:decoder_type')
                        or_cond.append(EqualsCondition(hp,
                                                       hp_decoder_type,
                                                       decoder_name))
                    if len(or_cond) == 0:
                        continue
                    elif len(or_cond) > 1:
                        conditions_to_add.append(OrConjunction(*or_cond))
                    else:
                        conditions_to_add.append(or_cond[0])
            cs.add_conditions(conditions_to_add)
        self.configuration_space_ = cs
        self.dataset_properties_ = dataset_properties
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
        choice = params.pop('__choice__')

        for param, value in params.items():
            param = param.replace(choice + ':', '')
            new_params[param] = value

        if init_params is not None:
            for param, value in init_params.items():
                param = param.replace(choice + ':', '')
                new_params[param] = value

        decoder_components = self.get_decoder_components()

        decoder_type: Optional[str] = None

        decoder_params = {}
        decoder_params_names = []
        for param, value in new_params.items():
            if decoder_type is None:
                for decoder_component in decoder_components.keys():
                    if param.startswith(decoder_component):
                        decoder_type = decoder_component
                        decoder_params_names.append(param)
                        param = param.replace(decoder_type + ':', '')
                        decoder_params[param] = value
            else:
                if param.startswith(decoder_type):
                    decoder_params_names.append(param)
                    param = param.replace(decoder_type + ':', '')
                    decoder_params[param] = value
        assert decoder_type is not None, 'Decoder must be given to initialize a network backbone'

        for param_name in decoder_params_names:
            del new_params[param_name]

        new_params['random_state'] = self.random_state
        decoder_params['random_state'] = self.random_state

        self.new_params = new_params
        self.choice = self.get_components()[choice](**new_params)
        self.decoder_choice = decoder_components[decoder_type](**decoder_params)

        self.pipeline = Pipeline([('net_structure', ForecastingNetworkStructure()),
                                  ('encoder', self.choice),
                                  ('decoder', self.decoder_choice)])
        return self

    @property
    def _defaults_network(self) -> List[str]:
        return ['MLPEncoder', 'RNNEncoder', 'NBEATSEncoder']

    def fit(self, X: Dict[str, Any], y: Any = None) -> Pipeline:  # type: ignore[override]
        """Handy method to check if a component is fitted

        Args:
            X (X: Dict[str, Any]):
                Dependencies needed by current component to perform fit
            y (Any):
                not used. To comply with sklearn API
        """
        # Allows to use check_is_fitted on the choice object
        self.fitted_ = True
        assert self.pipeline is not None, "Cannot call fit without initializing the component"
        return self.pipeline.fit(X, y)

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        assert self.pipeline is not None, "Cannot call transform before the object is initialized"
        return self.pipeline.transform(X)   # type: ignore[no-any-return]

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        raise NotImplementedError
