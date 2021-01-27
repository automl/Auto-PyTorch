from typing import Any, Dict, Optional, Union

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

import imgaug.augmenters as iaa
from imgaug.augmenters.meta import Augmenter

import numpy as np

from autoPyTorch.pipeline.components.setup.augmentation.image.base_image_augmenter import BaseImageAugmenter


class RandomAffine(BaseImageAugmenter):
    def __init__(self, use_augmenter: bool = True, scale_offset: float = 0.2,
                 translate_percent_offset: float = 0.3, shear: int = 30,
                 rotate: int = 45, random_state: Optional[Union[int, np.random.RandomState]] = None):
        super().__init__(use_augmenter=use_augmenter)
        self.random_state = random_state
        self.scale = (1, 1 - scale_offset)
        self.translate_percent = (0, translate_percent_offset)
        self.shear = (-shear, shear)
        self.rotate = (-rotate, rotate)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseImageAugmenter:
        if self.use_augmenter:
            self.augmenter: Augmenter = iaa.Affine(scale=self.scale, translate_percent=self.translate_percent,
                                                   rotate=self.rotate, shear=self.shear, mode='symmetric',
                                                   name=self.get_properties()['name'])

        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None,
        use_augmenter=([True, False], True),
        scale_offset=([0, 0.4], 0.2),
        translate_percent_offset=([0, 0.4], 0.2),
        shear=([0, 45], 30),
        rotate=([0, 360], 45)
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        scale_offset = UniformFloatHyperparameter('scale_offset', lower=scale_offset[0][0], upper=scale_offset[0][1],
                                                  default_value=scale_offset[1])

        translate_percent_offset = UniformFloatHyperparameter('translate_percent_offset',
                                                              lower=translate_percent_offset[0][0],
                                                              upper=translate_percent_offset[0][1],
                                                              default_value=translate_percent_offset[1])
        shear = UniformIntegerHyperparameter('shear', lower=shear[0][0], upper=shear[0][1], default_value=shear[1])
        rotate = UniformIntegerHyperparameter('rotate', lower=0, upper=360, default_value=45)

        use_augmenter = CategoricalHyperparameter('use_augmenter', choices=use_augmenter[0],
                                                  default_value=use_augmenter[1])

        cs.add_hyperparameters([scale_offset, translate_percent_offset])
        cs.add_hyperparameters([shear, rotate, use_augmenter])

        # only add hyperparameters to configuration space if we are using the augmenter
        cs.add_condition(CS.EqualsCondition(scale_offset, use_augmenter, True))
        cs.add_condition(CS.EqualsCondition(translate_percent_offset, use_augmenter, True))
        cs.add_condition(CS.EqualsCondition(shear, use_augmenter, True))
        cs.add_condition(CS.EqualsCondition(rotate, use_augmenter, True))

        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None
                       ) -> Dict[str, Any]:
        return {'name': 'RandomAffine'}
