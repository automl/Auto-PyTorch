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

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.augmentation.image.base_image_augmenter import BaseImageAugmenter
from autoPyTorch.utils.common import HyperparameterSearchSpace, get_hyperparameter


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
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        use_augmenter: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="use_augmenter",
                                                                             value_range=(True, False),
                                                                             default_value=True,
                                                                             ),
        scale_offset: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="scale_offset",
                                                                            value_range=(0, 0.4),
                                                                            default_value=0.2,
                                                                            ),
        translate_percent_offset: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="translate_percent_offset",
            value_range=(0, 0.4),
            default_value=0.2),
        shear: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="shear",
                                                                     value_range=(0, 45),
                                                                     default_value=30,
                                                                     ),
        rotate: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="rotate",
                                                                      value_range=(0, 360),
                                                                      default_value=45,
                                                                      ),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        use_augmenter = get_hyperparameter(use_augmenter, CategoricalHyperparameter)
        scale_offset = get_hyperparameter(scale_offset, UniformFloatHyperparameter)
        translate_percent_offset = get_hyperparameter(translate_percent_offset, UniformFloatHyperparameter)
        shear = get_hyperparameter(shear, UniformIntegerHyperparameter)
        rotate = get_hyperparameter(rotate, UniformIntegerHyperparameter)
        cs.add_hyperparameters([use_augmenter, scale_offset, translate_percent_offset])
        cs.add_hyperparameters([shear, rotate])

        # only add hyperparameters to configuration space if we are using the augmenter
        cs.add_condition(CS.EqualsCondition(scale_offset, use_augmenter, True))
        cs.add_condition(CS.EqualsCondition(translate_percent_offset, use_augmenter, True))
        cs.add_condition(CS.EqualsCondition(shear, use_augmenter, True))
        cs.add_condition(CS.EqualsCondition(rotate, use_augmenter, True))

        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Any]:
        return {'name': 'RandomAffine'}
