import os
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import ConfigurationSpace

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    autoPyTorchComponent,
    find_components,
)
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.coalescer.base_coalescer import BaseCoalescer
from autoPyTorch.utils.common import HyperparameterSearchSpace, HyperparameterValueType


coalescer_directory = os.path.split(__file__)[0]
_coalescer = find_components(__package__,
                             coalescer_directory,
                             BaseCoalescer)
_addons = ThirdPartyComponents(BaseCoalescer)


def add_coalescer(coalescer: BaseCoalescer) -> None:
    _addons.add_component(coalescer)


class CoalescerChoice(autoPyTorchChoice):
    """
    Allows for dynamically choosing coalescer component at runtime
    """
    proc_name = "coalescer"

    def get_components(self) -> Dict[str, autoPyTorchComponent]:
        """Returns the available coalescer components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all BaseCoalescer components available
                as choices for coalescer the categorical columns
        """
        # TODO: Create `@property def components(): ...`.
        components = OrderedDict()
        components.update(_coalescer)
        components.update(_addons.components)
        return components

    @staticmethod
    def _get_default_choice(
        avail_components: Dict[str, autoPyTorchComponent],
        include: List[str],
        exclude: List[str],
        defaults: List[str] = ['NoCoalescer', 'MinorityCoalescer'],
    ) -> str:
        # TODO: Make it a base method
        for choice in defaults:
            if choice in avail_components and choice in include and choice not in exclude:
                return choice
        else:
            raise RuntimeError(
                f"Available components is either not included in `include` {include} or "
                f"included in `exclude` {exclude}"
            )

    def _update_config_space(
        self,
        component: CSH.Hyperparameter,
        avail_components: Dict[str, autoPyTorchComponent],
        dataset_properties: Dict[str, BaseDatasetPropertiesType]
    ) -> None:
        # TODO: Make it a base method
        cs = ConfigurationSpace()
        cs.add_hyperparameter(component)

        # add only child hyperparameters of early_preprocessor choices
        for name in component.choices:
            updates = self._get_search_space_updates(prefix=name)
            func4cs = avail_components[name].get_hyperparameter_search_space

            # search space provides different args, so ignore it
            component_config_space = func4cs(dataset_properties, **updates)  # type:ignore[call-arg]
            parent_hyperparameter = {'parent': component, 'value': name}
            cs.add_configuration_space(
                name,
                component_config_space,
                parent_hyperparameter=parent_hyperparameter
            )

        self.configuration_space = cs

    def _check_choices_in_update(
        self,
        choices_in_update: Sequence[HyperparameterValueType],
        avail_components: Dict[str, autoPyTorchComponent]
    ) -> None:
        # TODO: Make it a base method
        if not set(choices_in_update).issubset(avail_components):
            raise ValueError(
                f"The update for {self.__class__.__name__} is expected to be "
                f"a subset of {avail_components}, but got {choices_in_update}"
            )

    def get_hyperparameter_search_space(self,
                                        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
                                        default: Optional[str] = None,
                                        include: Optional[List[str]] = None,
                                        exclude: Optional[List[str]] = None) -> ConfigurationSpace:
        # TODO: Make it a base method

        if dataset_properties is None:
            dataset_properties = dict()

        dataset_properties = {**self.dataset_properties, **dataset_properties}

        avail_cmps = self.get_available_components(
            dataset_properties=dataset_properties,
            include=include,
            exclude=exclude
        )

        if len(avail_cmps) == 0:
            raise ValueError(f"No {self.proc_name} found, please add {self.proc_name} to `include` argument")

        include = include if include is not None else list(avail_cmps.keys())
        exclude = exclude if exclude is not None else []
        if default is None:
            default = self._get_default_choice(avail_cmps, include, exclude)

        updates = self._get_search_space_updates()
        if "__choice__" in updates:
            component = self._get_component_with_updates(
                updates=updates,
                avail_components=avail_cmps,
                dataset_properties=dataset_properties
            )
        else:
            component = self._get_component_without_updates(
                default=default,
                include=include,
                avail_components=avail_cmps,
                dataset_properties=dataset_properties
            )

        self.dataset_properties = dataset_properties
        self._update_config_space(
            component=component,
            avail_components=avail_cmps,
            dataset_properties=dataset_properties
        )
        return self.configuration_space

    def _check_dataset_properties(self, dataset_properties: Dict[str, BaseDatasetPropertiesType]) -> None:
        """
        A mechanism in code to ensure the correctness of the dataset_properties
        It recursively makes sure that the children and parent level requirements
        are honored.

        Args:
            dataset_properties:
        """
        # TODO: Make it a base method
        super()._check_dataset_properties(dataset_properties)
        if any(key not in dataset_properties for key in ['categorical_columns', 'numerical_columns']):
            raise ValueError("Dataset properties must contain information about the type of columns")

    def _get_component_with_updates(
        self,
        updates: Dict[str, HyperparameterSearchSpace],
        avail_components: Dict[str, autoPyTorchComponent],
        dataset_properties: Dict[str, BaseDatasetPropertiesType],
    ) -> CSH.Hyperparameter:
        # TODO: Make it a base method
        choice_key = '__choice__'
        choices_in_update = updates[choice_key].value_range
        default_in_update = updates[choice_key].default_value
        self._check_choices_in_update(
            choices_in_update=choices_in_update,
            avail_components=avail_components
        )
        self._check_update_compatiblity(choices_in_update, dataset_properties)
        return CSH.CategoricalHyperparameter(choice_key, choices_in_update, default_in_update)

    def _get_component_without_updates(
        self,
        avail_components: Dict[str, autoPyTorchComponent],
        dataset_properties: Dict[str, BaseDatasetPropertiesType],
        default: str,
        include: List[str]
    ) -> CSH.Hyperparameter:
        """
        A method to get a hyperparameter information for the component.
        This method is run when we do not get updates from _get_search_space_updates.

        Args:
            avail_components (Dict[str, autoPyTorchComponent]):
                Available components for this processing.
            dataset_properties (Dict[str, BaseDatasetPropertiesType]):
                The properties of the dataset.
            default (str):
                The default component for this processing.
            include (List[str]):
                The components to include for the auto-pytorch searching.

        Returns:
            (CSH.Hyperparameter):
                The hyperparameter information for this processing.
        """
        # TODO: Make an abstract method with NotImplementedError
        choice_key = '__choice__'
        no_proc_key = 'NoCoalescer'
        choices = list(avail_components.keys())

        assert isinstance(dataset_properties['categorical_columns'], list)  # mypy check
        if len(dataset_properties['categorical_columns']) == 0:
            # only no coalescer is compatible if the dataset has only numericals
            default, choices = no_proc_key, [no_proc_key]
            if no_proc_key not in include:
                raise ValueError("Only no coalescer is compatible for a dataset with no categorical column")

        return CSH.CategoricalHyperparameter(choice_key, choices, default_value=default)

    def _check_update_compatiblity(
        self,
        choices_in_update: Sequence[HyperparameterValueType],
        dataset_properties: Dict[str, BaseDatasetPropertiesType]
    ) -> None:
        """
        Check the compatibility of the updates for the components
        in this processing given dataset properties.
        For example, some processing is not compatible with datasets
        with no numerical columns.
        We would like to check such compatibility in this method.

        Args:
            choices_in_update (Sequence[HyperparameterValueType]):
                The choices of components in updates
            dataset_properties (Dict[str, BaseDatasetPropertiesType]):
                The properties of the dataset.
        """
        # TODO: Make an abstract method with NotImplementedError
        assert isinstance(dataset_properties['categorical_columns'], list)  # mypy check
        if len(dataset_properties['categorical_columns']) > 0:
            # no restriction for update if dataset has categorical columns
            return

        if 'NoCoalescer' not in choices_in_update or len(choices_in_update) != 1:
            raise ValueError(
                "Only no coalescer is compatible for a dataset with no categorical column, "
                f"but got {choices_in_update}"
            )
