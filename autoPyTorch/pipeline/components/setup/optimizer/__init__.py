import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import ConfigurationSpace

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    autoPyTorchComponent,
    find_components,
)
from autoPyTorch.pipeline.components.setup.optimizer.base_optimizer import BaseOptimizerComponent

directory = os.path.split(__file__)[0]
_optimizers = find_components(__package__,
                              directory,
                              BaseOptimizerComponent)
_addons = ThirdPartyComponents(BaseOptimizerComponent)


def add_optimizer(optimizer: BaseOptimizerComponent) -> None:
    _addons.add_component(optimizer)


class OptimizerChoice(autoPyTorchChoice):

    def get_components(self) -> Dict[str, autoPyTorchComponent]:
        """Returns the available optimizer components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all BaseOptimizerComponents  available
                as choices
        """
        components = OrderedDict()
        components.update(_optimizers)
        components.update(_addons.components)
        return components

    def get_available_components(
        self,
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        include: List[str] = None,
        exclude: List[str] = None,
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
            Dict[str, autoPyTorchComponent]: A filtered dict of Optimizer
                components
        """
        if dataset_properties is None:
            dataset_properties = {}

        if include is not None and exclude is not None:
            raise ValueError(
                "The argument include and exclude cannot be used together.")

        available_comp = self.get_components()

        if include is not None:
            for incl in include:
                if incl not in available_comp:
                    raise ValueError("Trying to include unknown component: "
                                     "%s" % incl)

        components_dict = OrderedDict()
        for name in available_comp:
            if include is not None and name not in include:
                continue
            elif exclude is not None and name in exclude:
                continue

            entry = available_comp[name]

            # Exclude itself to avoid infinite loop
            if entry == OptimizerChoice or hasattr(entry, 'get_components'):
                continue

            # target_type = dataset_properties['target_type']
            # Apply some automatic filtering here based on dataset

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
            default (Optional[str]): Default component to use
            include: Optional[Dict[str, Any]]: what components to include. It is an exhaustive
                list, and will exclusively use this components.
            exclude: Optional[Dict[str, Any]]: which components to skip

        Returns:
            ConfigurationSpace: the configuration space of the hyper-parameters of the
                 chosen component
        """
        cs = ConfigurationSpace()

        if dataset_properties is None:
            dataset_properties = {}

        # Compile a list of legal preprocessors for this problem
        available_optimizer = self.get_available_components(
            dataset_properties=dataset_properties,
            include=include, exclude=exclude)

        if len(available_optimizer) == 0:
            raise ValueError("No Optimizer found")

        if default is None:
            defaults = [
                'AdamOptimizer',
                'AdamWOptimizer',
                'SGDOptimizer',
                'RMSpropOptimizer'
            ]
            for default_ in defaults:
                if default_ in available_optimizer:
                    default = default_
                    break
        updates = self._get_search_space_updates()
        if '__choice__' in updates.keys():
            choice_hyperparameter = updates['__choice__']
            if not set(choice_hyperparameter.value_range).issubset(available_optimizer):
                raise ValueError("Expected given update for {} to have "
                                 "choices in {} got {}".format(self.__class__.__name__,
                                                               available_optimizer,
                                                               choice_hyperparameter.value_range))
            optimizer = CSH.CategoricalHyperparameter('__choice__',
                                                      choice_hyperparameter.value_range,
                                                      default_value=choice_hyperparameter.default_value)
        else:
            optimizer = CSH.CategoricalHyperparameter(
                '__choice__',
                list(available_optimizer.keys()),
                default_value=default
            )
        cs.add_hyperparameter(optimizer)
        for name in optimizer.choices:
            updates = self._get_search_space_updates(prefix=name)
            config_space = available_optimizer[name].get_hyperparameter_search_space(dataset_properties,  # type: ignore
                                                                                     **updates)
            parent_hyperparameter = {'parent': optimizer, 'value': name}
            cs.add_configuration_space(
                name,
                config_space,
                parent_hyperparameter=parent_hyperparameter
            )

        self.configuration_space_ = cs
        self.dataset_properties_ = dataset_properties
        return cs

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        assert self.choice is not None, "Cannot call transform before the object is initialized"
        return self.choice.transform(X)  # type: ignore[no-any-return]
