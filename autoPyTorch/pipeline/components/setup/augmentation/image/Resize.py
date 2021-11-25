from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
)

import imgaug.augmenters as iaa
from imgaug.augmenters.meta import Augmenter

import numpy as np

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.augmentation.image.base_image_augmenter import BaseImageAugmenter
from autoPyTorch.utils.common import FitRequirement, HyperparameterSearchSpace, add_hyperparameter


class Resize(BaseImageAugmenter):

    def __init__(self, use_augmenter: bool = True,
                 random_state: Optional[Union[int, np.random.RandomState]] = None):
        super().__init__(use_augmenter=use_augmenter)
        self.random_state = random_state
        self.add_fit_requirements([
            FitRequirement('image_height', (int,), user_defined=True, dataset_property=True),
            FitRequirement('image_width', (int,), user_defined=True, dataset_property=True)])

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseImageAugmenter:
        self.check_requirements(X, y)
        if self.use_augmenter:
            self.augmenter: Augmenter = iaa.Resize(size=(X['dataset_properties']['image_height'],
                                                         X['dataset_properties']['image_width']),
                                                   interpolation='linear', name=self.get_properties()['name'])

        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        use_augmenter: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="use_augmenter",
                                                                             value_range=(True, False),
                                                                             default_value=True,
                                                                             ),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        add_hyperparameter(cs, use_augmenter, CategoricalHyperparameter)

        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Any]:
        return {'name': 'Resize'}
