from typing import Any, Dict, Optional, Union

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


class RandomCutout(BaseImageAugmenter):
    def __init__(self, use_augmenter: bool = True, p: float = 0.5,
                 random_state: Optional[Union[int, np.random.RandomState]] = None):
        super().__init__(use_augmenter=use_augmenter)
        self.p = p
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseImageAugmenter:
        if self.use_augmenter:
            self.augmenter: Augmenter = iaa.Sometimes(self.p, iaa.Cutout(nb_iterations=(1, 10), size=(0.1, 0.5),
                                                                         random_state=self.random_state),
                                                      name=self.get_properties()['name'])
        return self

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, str]] = None,
            use_augmenter=([True, False], True),
            p=([0.2, 1], 0.5)
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        p = UniformFloatHyperparameter('p', lower=p[0][0], upper=p[0][1], default_value=p[1])
        use_augmenter = CategoricalHyperparameter('use_augmenter', choices=use_augmenter[0],
                                                  default_value=use_augmenter[1])
        cs.add_hyperparameters([p, use_augmenter])

        # only add hyperparameters to configuration space if we are using the augmenter
        cs.add_condition(CS.EqualsCondition(p, use_augmenter, True))
        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None
                       ) -> Dict[str, Any]:
        return {'name': 'RandomCutout'}
