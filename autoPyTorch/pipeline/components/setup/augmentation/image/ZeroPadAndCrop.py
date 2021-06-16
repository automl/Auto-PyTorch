from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter
)

import imgaug.augmenters as iaa
from imgaug.augmenters.meta import Augmenter

import numpy as np

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.augmentation.image.base_image_augmenter import BaseImageAugmenter
from autoPyTorch.utils.common import FitRequirement, HyperparameterSearchSpace, add_hyperparameter


class ZeroPadAndCrop(BaseImageAugmenter):

    def __init__(self, percent: float = 0.1,
                 random_state: Optional[Union[int, np.random.RandomState]] = None):
        super().__init__()
        self.random_state = random_state
        self.percent = percent
        self.pad_augmenter: Optional[Augmenter] = None
        self.crop_augmenter: Optional[Augmenter] = None
        self.add_fit_requirements([
            FitRequirement('image_height', (int,), user_defined=True, dataset_property=True),
            FitRequirement('image_width', (int,), user_defined=True, dataset_property=True)])

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseImageAugmenter:
        self.check_requirements(X, y)
        self.pad_augmenter = iaa.Pad(percent=self.percent, keep_size=False)
        self.crop_augmenter = iaa.CropToFixedSize(height=X['dataset_properties']['image_height'],
                                                  width=X['dataset_properties']['image_width'])
        self.augmenter: Augmenter = iaa.Sequential([
            self.pad_augmenter,
            self.crop_augmenter
        ], name=self.get_properties()['name'])

        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        percent: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='percent',
                                                                       value_range=(0, 0.5),
                                                                       default_value=0.1,
                                                                       )
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        add_hyperparameter(cs, percent, UniformFloatHyperparameter)
        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Any]:
        return {'name': 'ZeroPadAndCrop'}
