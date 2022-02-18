import os
from collections import OrderedDict
from typing import Dict, Optional, List, Any, Union, Tuple
from sklearn.pipeline import Pipeline

import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import ConfigurationSpace, Configuration
from ConfigSpace.conditions import EqualsCondition, OrConjunction

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType

from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    autoPyTorchComponent,
    find_components,
)
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.setup.network_backbone import NetworkBackboneChoice
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder import (
    BaseForecastingEncoder,
)
from autoPyTorch.utils.common import FitRequirement, HyperparameterSearchSpace
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.flat_encoder \
    import FlatForecastingEncoderChoice
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.seq_encoder import \
    SeqForecastingEncoderChoice
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder import \
    decoders, decoder_addons, add_decoder
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdate


class ForecastingNetworkChoice(autoPyTorchChoice):
    """
    A network is composed of an encoder and decoder. In most of the case, the choice of decoder is heavily dependent on
    the choice of encoder. Thus here "choice" indicates the choice of encoder, then decoder will be determined by
    the encoder.
    """
    def __init__(self,
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        self.include_components = None
        self.exclude_components = None

    def get_components(self) -> Dict[str, autoPyTorchComponent]:
        """Returns the available backbone components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all basebackbone components available
                as choices for learning rate scheduling
        """
        components = OrderedDict()
        components.update({"flat_encoder": FlatForecastingEncoderChoice,
                           "seq_encoder": SeqForecastingEncoderChoice})
        return components

    def get_available_components(
        self,
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        include: List[str] = None,
        exclude: List[str] = None,
        components: Optional[Dict[str, autoPyTorchComponent]] = None
    ) -> Dict[str, autoPyTorchComponent]:
        """Filters out components based on user provided
        include/exclude directives, as well as the dataset properties

        Args:
         include (Optional[Dict[str, Any]]): what hyper-parameter configurations
            to honor when creating the configuration space
         exclude (Optional[Dict[str, Any]]): what hyper-parameter configurations
             to remove from the configuration space
         dataset_properties (Optional[Dict[str, Union[str, int]]]): Caracteristics
             of the dataset to guide the pipeline choices of components

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
            include_top = set()
            self.include_components = {}
            for incl in include:
                if incl not in available_comp:
                    for comp in available_comp.keys():
                        self.include_components[comp] = []
                        if incl.startswith(comp):
                            incl_sub = ":".join(incl.split(":")[1:])
                            if comp in self.include_components:
                                self.include_components[comp].append(incl_sub)
                            else:
                                self.include_components[comp] = [incl_sub]
                            include_top.add(comp)
                else:
                    include_top.add(incl)
            if not include_top:
                raise ValueError(f"Trying to include unknown component: {include}")
            include = list(include_top)
        elif exclude is not None:
            self.exclude_components = {}
            for excl in exclude:
                for comp in available_comp.keys():
                    if excl.startswith(comp):
                        excl_sub = ":".join(excl.split(":")[1:])
                        if comp in self.exclude_components:
                            self.exclude_components[comp].append(excl_sub)
                        else:
                            self.exclude_components[comp] = [excl_sub]

        components_dict = OrderedDict()
        for name in available_comp:
            if include is not None and name not in include:
                continue
            elif exclude is not None and name in exclude:
                continue

            entry = available_comp[name]

            # Exclude itself to avoid infinite loop
            if entry == ForecastingNetworkChoice:
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
            dataset_properties (Optional[Dict[str, str]]): Describes the dataset to work on
            default (Optional[str]): Default backbone to use
            include: Optional[Dict[str, Any]]: what components to include. It is an exhaustive
                list, and will exclusively use this components.
            exclude: Optional[Dict[str, Any]]: which components to skip
            network_type: type of the network, it determines how to handle the sequential data: flat networks
            (FFNN and NBEATS) simply flat the input to a 2D input, whereas seq network receives sequential 3D inputs:
            thus, seq networks could be stacked to form a larger network that is composed of different parts.

        Returns:
            ConfigurationSpace: the configuration space of the hyper-parameters of the
                 chosen component
        """
        if dataset_properties is None:
            dataset_properties = {}

        cs = ConfigurationSpace()

        # Compile a list of legal preprocessors for this problem
        available_encoders = self.get_available_components(
            dataset_properties=dataset_properties,
            include=include, exclude=exclude)

        if len(available_encoders) == 0:
            raise ValueError("No Encoder found")

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

        for name in hp_encoder.choices:
            updates = self._get_search_space_updates(prefix=name)
            include_encoder = None
            exclude_encoder = None
            if include is not None:
                if name in self.include_components:
                    include_encoder = self.include_components[name]
            if exclude is not None:
                if name in self.exclude_components:
                    exclude_encoder = self.exclude_components[name]
            import pdb
            pdb.set_trace()

            config_space = available_encoders[name].get_hyperparameter_search_space(
                dataset_properties=dataset_properties,  # type: ignore
                include=include_encoder,
                exclude=exclude_encoder,
                **updates)
            parent_hyperparameter = {'parent': hp_encoder, 'value': name}
            cs.add_configuration_space(
                name,
                config_space,
                parent_hyperparameter=parent_hyperparameter
            )

        self.configuration_space_ = cs
        self.dataset_properties_ = dataset_properties
        return cs

    def _apply_search_space_update(self, hyperparameter_search_space_update: HyperparameterSearchSpaceUpdate) -> None:
        self._cs_updates[hyperparameter_search_space_update.hyperparameter] = hyperparameter_search_space_update


    @property
    def _defaults_network(self):
        return ['flat_network',
                'seq_network']

    def fit(self, X: Dict[str, Any], y: Any) -> autoPyTorchComponent:
        """Handy method to check if a component is fitted

        Args:
            X (X: Dict[str, Any]):
                Dependencies needed by current component to perform fit
            y (Any):
                not used. To comply with sklearn API
        """
        # Allows to use check_is_fitted on the choice object
        self.fitted_ = True
        assert self.choice is not None, "Cannot call fit without initializing the component"
        return self.choice.fit(X, y)
        #self.choice.fit(X, y)
        #self.choice.transform(X)
        #return self.choice

    def transform(self, X: Dict) -> Dict:
        assert self.choice is not None, "Cannot call transform before the object is initialized"
        return self.choice.transform(X)

    @property
    def _defaults_network(self):
        return ['MLPEncoder']
