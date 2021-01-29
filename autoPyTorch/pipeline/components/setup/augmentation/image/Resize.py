from typing import Any, Dict, Optional, Tuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
)

import imgaug.augmenters as iaa
from imgaug.augmenters.meta import Augmenter

import numpy as np

from autoPyTorch.pipeline.components.setup.augmentation.image.base_image_augmenter import BaseImageAugmenter
from autoPyTorch.utils.common import FitRequirement


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
            dataset_properties: Optional[Dict[str, str]] = None,
            use_augmenter: Tuple[Tuple, bool] = ((True, False), True),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        use_augmenter = CategoricalHyperparameter('use_augmenter', choices=use_augmenter[0],
                                                  default_value=use_augmenter[1])
        cs.add_hyperparameters([use_augmenter])

        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None
                       ) -> Dict[str, Any]:
        return {'name': 'Resize'}
