import os
from collections import OrderedDict
from typing import Dict, List, Optional

import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import ConfigurationSpace

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    autoPyTorchComponent,
    find_components,
)
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent

preprocessing_directory = os.path.split(__file__)[0]
_preprocessors = find_components(__package__,
                                 preprocessing_directory,
                                 autoPyTorchFeaturePreprocessingComponent)
_addons = ThirdPartyComponents(autoPyTorchFeaturePreprocessingComponent)


def add_feature_preprocessor(feature_preprocessor: autoPyTorchFeaturePreprocessingComponent) -> None:
    _addons.add_component(feature_preprocessor)


class FeatureProprocessorChoice(autoPyTorchChoice):
    """
    Allows for dynamically choosing feature_preprocessor component at runtime
    """

    def get_components(self) -> Dict[str, autoPyTorchComponent]:
        """Returns the available feature_preprocessor components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all feature preprocessor components available
                as choices for encoding the categorical columns
        """
        components: Dict = OrderedDict()
        components.update(_preprocessors)
        components.update(_addons.components)
        return components

    def get_hyperparameter_search_space(self,
                                        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
                                        default: Optional[str] = None,
                                        include: Optional[List[str]] = None,
                                        exclude: Optional[List[str]] = None) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        if dataset_properties is None:
            dataset_properties = dict()

        dataset_properties = {**self.dataset_properties, **dataset_properties}

        available_ = self.get_available_components(dataset_properties=dataset_properties,
                                                   include=include,
                                                   exclude=exclude)

        if len(available_) == 0:
            raise ValueError("no feature preprocessors found, please add a feature preprocessor")

        if default is None:
            defaults = ['NoFeaturePreprocessor',
                        'FastICA',
                        'KernelPCA',
                        'RandomKitchenSinks',
                        'Nystroem',
                        'PolynomialFeatures',
                        'PowerTransformer',
                        'TruncatedSVD',
                        ]
            for default_ in defaults:
                if default_ in available_:
                    if include is not None and default_ not in include:
                        continue
                    if exclude is not None and default_ in exclude:
                        continue
                    default = default_
                    break

        numerical_columns = dataset_properties['numerical_columns'] \
            if isinstance(dataset_properties['numerical_columns'], List) else []
        updates = self._get_search_space_updates()
        if '__choice__' in updates.keys():
            choice_hyperparameter = updates['__choice__']
            if not set(choice_hyperparameter.value_range).issubset(available_):
                raise ValueError("Expected given update for {} to have "
                                 "choices in {} got {}".format(self.__class__.__name__,
                                                               available_,
                                                               choice_hyperparameter.value_range))
            if len(numerical_columns) == 0:
                assert len(choice_hyperparameter.value_range) == 1
                assert 'NoFeaturePreprocessor' in choice_hyperparameter.value_range, \
                    "Provided {} in choices, however, the dataset " \
                    "is incompatible with it".format(choice_hyperparameter.value_range)
            preprocessor = CSH.CategoricalHyperparameter('__choice__',
                                                         choice_hyperparameter.value_range,
                                                         default_value=choice_hyperparameter.default_value)
        else:
            # add only no feature preprocessor to choice hyperparameters in case the dataset is only categorical
            if len(numerical_columns) == 0:
                default = 'NoFeaturePreprocessor'
                if include is not None and default not in include:
                    raise ValueError("Provided {} in include, however, "
                                     "the dataset is incompatible with it".format(include))
                preprocessor = CSH.CategoricalHyperparameter('__choice__',
                                                             ['NoFeaturePreprocessor'],
                                                             default_value=default)
            else:
                # Truncated SVD requires n_features > n_components
                if len(numerical_columns) == 1:
                    del available_['TruncatedSVD']
                preprocessor = CSH.CategoricalHyperparameter('__choice__',
                                                             list(available_.keys()),
                                                             default_value=default)

        cs.add_hyperparameter(preprocessor)

        # add only child hyperparameters of preprocessor choices
        for name in preprocessor.choices:
            updates = self._get_search_space_updates(prefix=name)
            config_space = available_[name].get_hyperparameter_search_space(dataset_properties,  # type:ignore
                                                                            **updates)
            parent_hyperparameter = {'parent': preprocessor, 'value': name}
            cs.add_configuration_space(name, config_space,
                                       parent_hyperparameter=parent_hyperparameter)

        self.configuration_space = cs
        self.dataset_properties = dataset_properties
        return cs
