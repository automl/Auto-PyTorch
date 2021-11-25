from typing import Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace

from imgaug.augmenters.meta import Augmenter

import numpy as np

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent


class BaseImageAugmenter(autoPyTorchSetupComponent):
    def __init__(self, use_augmenter: bool = True) -> None:
        super().__init__()
        self.use_augmenter = use_augmenter
        self.augmenter: Optional[Augmenter] = None

    def get_image_augmenter(self) -> Optional[Augmenter]:
        """
        Get fitted augmenter. Can only be called if fit()
        has been called on the object.
        Returns:
            BaseEstimator: Fitted augmentor
        """
        if self.augmenter is None and self.use_augmenter:
            raise AttributeError("Can't return augmenter for {}, as augmenter is  "
                                 "set to be used but it has not been fitted"
                                 "  yet".format(self.__class__.__name__))
        return self.augmenter

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if self.augmenter is None:
            raise ValueError("cant call {} without fitting first."
                             .format(self.__class__.__name__))
        # explicitly converting to np array as currently zeropadandcrop gives a list
        return np.array(self.augmenter(images=X))

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        return cs
