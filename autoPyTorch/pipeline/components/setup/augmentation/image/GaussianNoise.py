from typing import Any, Dict, Optional, Tuple, Union

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

import imgaug.augmenters as iaa
from imgaug.augmenters.meta import Augmenter

import numpy as np

from autoPyTorch.pipeline.components.setup.augmentation.image.base_image_augmenter import BaseImageAugmenter


class GaussianNoise(BaseImageAugmenter):
    def __init__(self, use_augmenter: bool = True, sigma_offset: float = 0.3,
                 random_state: Optional[Union[int, np.random.RandomState]] = None):
        super().__init__(use_augmenter=use_augmenter)
        self.random_state = random_state
        self.sigma = (0, sigma_offset)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseImageAugmenter:
        if self.use_augmenter:
            self.augmenter: Augmenter = iaa.AdditiveGaussianNoise(scale=self.sigma, name=self.get_properties()['name'])
        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None,
        use_augmenter: Tuple[Tuple[bool, bool], bool] = ((True, False), True),
        sigma_offset: Tuple[Tuple[float, float], float] = ((0.0, 3.0), 0.3)
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        sigma_offset = UniformFloatHyperparameter('sigma_offset', lower=sigma_offset[0][0], upper=sigma_offset[0][1],
                                                  default_value=sigma_offset[1])
        use_augmenter = CategoricalHyperparameter('use_augmenter', choices=use_augmenter[0],
                                                  default_value=use_augmenter[1])
        cs.add_hyperparameters([use_augmenter, sigma_offset])
        # only add hyperparameters to configuration space if we are using the augmenter
        cs.add_condition(CS.EqualsCondition(sigma_offset, use_augmenter, True))
        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None
                       ) -> Dict[str, Any]:
        return {'name': 'GaussianNoise'}
