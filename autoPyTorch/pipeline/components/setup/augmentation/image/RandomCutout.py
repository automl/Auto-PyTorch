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

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.augmentation.image.base_image_augmenter import BaseImageAugmenter
from autoPyTorch.utils.common import HyperparameterSearchSpace, get_hyperparameter


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
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        use_augmenter: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="use_augmenter",
                                                                             value_range=(True, False),
                                                                             default_value=True,
                                                                             ),
        p: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="p",
                                                                 value_range=(0.2, 1.0),
                                                                 default_value=0.5,
                                                                 ),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        use_augmenter = get_hyperparameter(use_augmenter, CategoricalHyperparameter)
        p = get_hyperparameter(p, UniformFloatHyperparameter)
        cs.add_hyperparameters([use_augmenter, p])
        # only add hyperparameters to configuration space if we are using the augmenter
        cs.add_condition(CS.EqualsCondition(p, use_augmenter, True))
        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Any]:
        return {'name': 'RandomCutout'}
