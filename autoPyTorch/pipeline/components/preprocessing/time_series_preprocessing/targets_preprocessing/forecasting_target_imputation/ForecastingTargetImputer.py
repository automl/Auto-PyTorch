from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

import pandas as pd

from scipy.sparse import spmatrix

import torch

from sktime.transformations.series.impute import Imputer
from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent
from autoPyTorch.utils.common import FitRequirement


class ForecastingTargetImputer(autoPyTorchComponent):
    """
    Forecasting target imputor

    Attributes:
        random_state (Optional[np.random.RandomState]):
            The random state to use for the imputer.
        numerical_strategy (str: default='mean'):
            The strategy to use for imputing numerical columns.
            Can be one of ['most_frequent', 'constant_!missing!']
    """

    def __init__(
            self,
            random_state: Optional[np.random.RandomState] = None,
            impution_strategy: str = 'mean',
    ):
        super().__init__()
        self.random_state = random_state
        self.inputer = Imputer(method=impution_strategy, random_state=self.random_state, value=0., )

        self.add_fit_requirements([
            FitRequirement('is_small_preprocess', (bool,), user_defined=True, dataset_property=True),
            FitRequirement('y_train', (pd.DataFrame, ), user_defined=True,
                           dataset_property=False)])

    def fit(self, X: Dict[str, Any], y: Optional[Any] = None) -> "ForecastingTargetImputer":
        """
        fits the target inputor  based on the given fit dictionary 'X'.

        Args:
            X (Dict[str, Any]):
                The fit dictionary
            y (Optional[Any]):
                Not Used -- to comply with API

        Returns:
            self:
                returns an instance of self.
        """
        self.check_requirements(X, y)
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds self into the 'X' dictionary and returns it.
        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        if X['dataset_properties']['is_small_preprocess']:
            if 'X_train' in X:
                X_train = X['X_train']
            else:
                # Incorporate the transform to the dataset
                X_train = X['backend'].load_datamanager().train_tensors[0]

            X['X_train'] = preprocess(dataset=X_train, transforms=transforms)
        X.update({'y_train': self.inputer.transform(X['y_train'])})
        return X


