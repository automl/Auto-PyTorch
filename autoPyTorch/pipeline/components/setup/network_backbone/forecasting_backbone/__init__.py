from collections import OrderedDict
from typing import Any, Dict, List, Optional

import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

import numpy as np


from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder import \
    AbstractForecastingEncoderChoice
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.\
    flat_encoder import FlatForecastingEncoderChoice
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.\
    seq_encoder import SeqForecastingEncoderChoice
from autoPyTorch.utils.hyperparameter_search_space_update import \
    HyperparameterSearchSpaceUpdate


class ForecastingNetworkChoice(autoPyTorchChoice):
    """
    A network is composed of an encoder and decoder. In most of the case, the choice of decoder is heavily dependent on
    the choice of encoder. Therefore, here "choice" indicates the choice of encoder, then decoder will be determined by
    the encoder.
    """

    def __init__(self,
                 dataset_properties: Dict[str, BaseDatasetPropertiesType],
                 random_state: Optional[np.random.RandomState] = None
                 ):
        super().__init__(dataset_properties, random_state)
        self.include_components: Dict[str, List[str]] = {}
        self.exclude_components: Dict[str, List[str]] = {}

        self.default_components = OrderedDict(
            {"flat_encoder": FlatForecastingEncoderChoice(dataset_properties=self.dataset_properties,
                                                          random_state=self.random_state),
             "seq_encoder": SeqForecastingEncoderChoice(dataset_properties=self.dataset_properties,
                                                        random_state=self.random_state)})

    def get_components(self) -> Dict[str, AbstractForecastingEncoderChoice]:  # type: ignore[override]
        """Returns the available backbone components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all basebackbone components available
                as choices for learning rate scheduling
        """
        return self.default_components

    def get_available_components(  # type: ignore[override]
            self,
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
            include: List[str] = None,
            exclude: List[str] = None,
            components: Optional[Dict[str, AbstractForecastingEncoderChoice]] = None
    ) -> Dict[str, AbstractForecastingEncoderChoice]:
        """Filters out components based on user provided
        include/exclude directives, as well as the dataset properties

        Args:
            include (Optional[Dict[str, Any]]):
                what hyper-parameter configurations to honor when creating the configuration space. It can also include
                nested components, for instance, flat_encoder:MLPEncoder
            exclude (Optional[Dict[str, Any]]):
                what hyper-parameter configurations to remove from the configuration space. It can also include
                nested components, for instance, flat_encoder:MLPEncoder
            dataset_properties (Optional[Dict[str, BaseDatasetPropertiesType]]):
                Characteristics of the dataset to guide the pipeline choices of components

        Returns:
            Dict[str, autoPyTorchComponent]:
                A filtered dict of learning rate backbones

        """
        if dataset_properties is None:
            dataset_properties = {}

        if include is not None and exclude is not None:
            raise ValueError(
                "The argument include and exclude cannot be used together.")

        if components is None:
            available_comp = self.get_components()
        else:
            available_comp = components  # type: ignore[assignment]

        if include is not None:
            include_top = set()
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
            # TODO: Think if there is any case where a backbone is not recommended for a certain dataset

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
                Default backbone to use
            include: Optional[Dict[str, Any]]:
                what components to include. It is an exhaustive list, and will exclusively use this components.
                It can also include nested components, for instance, flat_encoder:MLPEncoder
            exclude: Optional[Dict[str, Any]]:
                which components to skip. It can also include nested components, for instance, flat_encoder:MLPEncoder

        Returns:
            ConfigurationSpace:
                the configuration space of the hyper-parameters of the chosen component
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

            config_space = available_encoders[name].get_hyperparameter_search_space(
                dataset_properties=dataset_properties,
                include=include_encoder,
                exclude=exclude_encoder,
                **updates  # type: ignore[call-arg, arg-type]
            )
            parent_hyperparameter = {'parent': hp_encoder, 'value': name}
            cs.add_configuration_space(
                name,
                config_space,
                parent_hyperparameter=parent_hyperparameter
            )

        self.configuration_space_ = cs
        self.dataset_properties_ = dataset_properties

        return cs

    def set_hyperparameters(self,
                            configuration: Configuration,
                            init_params: Optional[Dict[str, Any]] = None
                            ) -> 'autoPyTorchChoice':
        new_params = {}

        params = configuration.get_dictionary()
        choice = params['__choice__']
        del params['__choice__']

        for param, value in params.items():
            param = param.replace(choice + ':', '')
            new_params[param] = value

        if init_params is not None:
            for param, value in init_params.items():
                param = param.replace(choice + ':', '')
                new_params[param] = value

        choice_component = self.get_components()[choice]

        updates = self._get_search_space_updates(prefix=choice)

        self.new_params = new_params
        sub_configuration_space = choice_component.get_hyperparameter_search_space(
            self.dataset_properties,
            **updates  # type: ignore[call-arg, arg-type]
        )

        sub_configuration = Configuration(sub_configuration_space,
                                          values=new_params)
        self.choice = choice_component.set_hyperparameters(sub_configuration)  # type: ignore[assignment]

        return self

    def _apply_search_space_update(self, hyperparameter_search_space_update: HyperparameterSearchSpaceUpdate) -> None:
        sub_module_name_component = hyperparameter_search_space_update.hyperparameter.split(':')
        if len(sub_module_name_component) <= 2:
            super()._apply_search_space_update(hyperparameter_search_space_update)
        else:
            sub_module_name = sub_module_name_component[0]
            # TODO create a new update and consider special HPs for seq encoder!!!
            update_sub_module = HyperparameterSearchSpaceUpdate(
                hyperparameter_search_space_update.node_name,
                hyperparameter=hyperparameter_search_space_update.hyperparameter.replace(f'{sub_module_name}:', ''),
                value_range=hyperparameter_search_space_update.value_range,
                default_value=hyperparameter_search_space_update.default_value,
                log=hyperparameter_search_space_update.log
            )
            self.get_components()[sub_module_name]._apply_search_space_update(update_sub_module)

    @property
    def _defaults_network(self) -> List[str]:
        return ['flat_network',
                'seq_network']

    def fit(self, X: Dict[str, Any], y: Any = None) -> autoPyTorchComponent:
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

    def transform(self, X: Dict) -> Dict:
        assert self.choice is not None, "Cannot call transform before the object is initialized"
        return self.choice.transform(X)  # type: ignore[no-any-return]
