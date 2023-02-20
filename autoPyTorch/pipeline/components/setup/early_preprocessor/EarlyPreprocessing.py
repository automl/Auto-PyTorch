from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

import pandas as pd

from scipy.sparse import csr_matrix

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent
from autoPyTorch.pipeline.components.setup.early_preprocessor.utils import get_preprocess_transforms, preprocess, get_preprocessed_dtype
from autoPyTorch.utils.common import FitRequirement


class EarlyPreprocessing(autoPyTorchSetupComponent):

    def __init__(self, random_state: Optional[np.random.RandomState] = None) -> None:
        super().__init__()
        self.random_state = random_state
        self.add_fit_requirements([
            FitRequirement('is_small_preprocess', (bool,), user_defined=True, dataset_property=True),
            FitRequirement('X_train', (np.ndarray, pd.DataFrame, csr_matrix), user_defined=True,
                           dataset_property=False)])

    def fit(self, X: Dict[str, Any], y: Any = None) -> "EarlyPreprocessing":
        self.check_requirements(X, y)

        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:

        transforms = get_preprocess_transforms(X)
        preprocessed_dtype = None
        if X['dataset_properties']['is_small_preprocess']:
            if 'X_train' in X:
                X_train = X['X_train']
            else:
                # Incorporate the transform to the dataset
                X_train = X['backend'].load_datamanager().train_tensors[0]

            X['X_train'] = preprocess(dataset=X_train, transforms=transforms)

            preprocessed_dtype = get_preprocessed_dtype(X['X_train'])

        # We need to also save the preprocess transforms for inference
        X.update({
            'preprocess_transforms': transforms,
            'preprocessed_dtype': preprocessed_dtype
        })
        return X

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> ConfigurationSpace:
        return ConfigurationSpace()

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'EarlyPreprocessing',
            'name': 'Early Preprocessing Node',
        }

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.__class__.__name__
        return string
