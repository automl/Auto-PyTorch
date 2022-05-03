from typing import Any, Dict, Optional, Union
import logging.handlers
import time
import psutil

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

import pandas as pd

from scipy.sparse import spmatrix

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent
from autoPyTorch.pipeline.components.setup.early_preprocessor.utils import get_preprocess_transforms, preprocess
from autoPyTorch.utils.common import FitRequirement
from autoPyTorch.utils.logging_ import get_named_client_logger


class EarlyPreprocessing(autoPyTorchSetupComponent):

    def __init__(self, random_state: Optional[np.random.RandomState] = None) -> None:
        super().__init__()
        self.random_state = random_state
        self.add_fit_requirements([
            FitRequirement('X_train', (np.ndarray, pd.DataFrame, spmatrix), user_defined=True,
                           dataset_property=False)])

    def fit(self, X: Dict[str, Any], y: Any = None) -> "EarlyPreprocessing":
        self.check_requirements(X, y)
        self.logger = get_named_client_logger(
            name=f"{X['num_run']}_{self.__class__.__name__}_{time.time()}",
            # Log to a user provided port else to the default logging port
            port=X['logger_port'
                   ] if 'logger_port' in X else logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        )
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:

        transforms = get_preprocess_transforms(X)
        if 'X_train' in X:
            X_train = X['X_train']
        else:
            # Incorporate the transform to the dataset
            X_train = X['backend'].load_datamanager().train_tensors[0]

        self.logger.debug(f"Available virtual memory: {psutil.virtual_memory().available/1024/1024}, total virtual memroy: {psutil.virtual_memory().total/1024/1024}")
        X['X_train'] = preprocess(dataset=X_train, transforms=transforms)
        self.logger.debug(f"After preprocessing Available virtual memory: {psutil.virtual_memory().available/1024/1024}, total virtual memroy: {psutil.virtual_memory().total/1024/1024}")

        # We need to also save the preprocess transforms for inference
        X.update({
                 'preprocess_transforms': transforms,
                 'shape_after_preprocessing': X['X_train'].shape[1:]
                 })
        return X

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        **kwargs: Any
    ) -> ConfigurationSpace:
        return ConfigurationSpace()

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'EarlyPreprocessing',
            'name': 'Early Preprocessing Node',
        }

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.__class__.__name__
        return string
