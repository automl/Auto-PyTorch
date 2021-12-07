import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import ConfigurationSpace

from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.scaling.base_scaler_choice import \
    ScalerChoice
from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    autoPyTorchComponent,
    find_components,
)
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.forecasting_target_scaling.\
    base_target_scaler import BaseTargetScaler

scaling_directory = os.path.split(__file__)[0]
_scalers = find_components(__package__,
                           scaling_directory,
                           BaseTargetScaler)

_addons = ThirdPartyComponents(BaseTargetScaler)


def add_scaler(scaler: BaseTargetScaler) -> None:
    _addons.add_component(scaler)


class TargetScalerChoice(ScalerChoice):
    """
    Allows for dynamically choosing scaling component at runtime, not
    """

    def get_components(self) -> Dict[str, autoPyTorchComponent]:
        """Returns the available scaler components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all BaseScalers components available
                as choices for scaling
        """
        components = OrderedDict()
        components.update(_scalers)
        components.update(_addons.components)
        return components

    def get_hyperparameter_search_space(self,
                                        dataset_properties: Optional[Dict[str, Any]] = None,
                                        default: Optional[str] = None,
                                        include: Optional[List[str]] = None,
                                        exclude: Optional[List[str]] = None) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        if dataset_properties is None:
            dataset_properties = dict()

        dataset_properties = {**self.dataset_properties, **dataset_properties}

        available_scalers = self.get_available_components(dataset_properties=dataset_properties,
                                                          include=include,
                                                          exclude=exclude)

        if len(available_scalers) == 0:
            raise ValueError("no scalers found, please add a scaler")

        if default is None:
            defaults = ['StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'NoScaler']
            for default_ in defaults:
                if default_ in available_scalers:
                    default = default_
                    break

        # add only no scaler to choice hyperparameters in case the dataset is only categorical
        if len(dataset_properties['numerical_features']) == 0:
            default = 'NoScaler'
            if include is not None and default not in include:
                raise ValueError("Provided {} in include, however, "
                                 "the dataset is incompatible with it".format(include))
            preprocessor = CSH.CategoricalHyperparameter('__choice__',
                                                         ['NoScaler'],
                                                         default_value=default)
        else:
            preprocessor = CSH.CategoricalHyperparameter('__choice__',
                                                         list(available_scalers.keys()),
                                                         default_value=default)
        cs.add_hyperparameter(preprocessor)

        # add only child hyperparameters of early_preprocessor choices
        for name in preprocessor.choices:
            updates = self._get_search_space_updates(prefix=name)
            config_space = available_scalers[name].get_hyperparameter_search_space(dataset_properties,  # type:ignore
                                                                                   **updates)
            parent_hyperparameter = {'parent': preprocessor, 'value': name}
            cs.add_configuration_space(name, config_space,
                                       parent_hyperparameter=parent_hyperparameter)

        self.configuration_space = cs
        self.dataset_properties = dataset_properties
        return cs

    def _check_dataset_properties(self, dataset_properties: Dict[str, Any]) -> None:
        """
        A mechanism in code to ensure the correctness of the fit dictionary
        It recursively makes sure that the children and parent level requirements
        are honored before fit.
        Args:
            dataset_properties (Dict[str, Any]): dictionary holding the dataset properties

        """
        super()._check_dataset_properties(dataset_properties)
        assert "target_columns" in dataset_properties, \
            "Dataset properties must contain information about the target_columns"
