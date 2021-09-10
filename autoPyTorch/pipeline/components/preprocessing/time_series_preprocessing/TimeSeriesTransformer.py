from typing import Any, Dict, List, Optional, Union

import numpy as np

from sklearn.pipeline import Pipeline, make_pipeline

import torch

from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.base_time_series_preprocessing import \
    autoPyTorchTimeSeriesPreprocessingComponent
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.utils import get_time_series_preprocessers
from autoPyTorch.utils.common import FitRequirement, subsampler


class TimeSeriesTransformer(autoPyTorchTimeSeriesPreprocessingComponent):

    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super().__init__()
        self.random_state = random_state
        self.preprocessor: Optional[Pipeline] = None
        self.add_fit_requirements([
            FitRequirement('numerical_features', (List,), user_defined=True, dataset_property=True),
            FitRequirement('categorical_features', (List,), user_defined=True, dataset_property=True)])

    def fit(self, X: Dict[str, Any], y: Any = None) -> "TimeSeriesTransformer":
        """
        Creates a column transformer for the chosen tabular
        preprocessors
        Args:
            X (Dict[str, Any]): fit dictionary

        Returns:
            "TabularColumnTransformer": an instance of self
        """
        self.check_requirements(X, y)

        preprocessors = get_time_series_preprocessers(X)

        if len(X['dataset_properties']['categorical_features']):
            raise ValueError("Categorical features are not yet supported for time series")

        numerical_pipeline = make_pipeline(*preprocessors['numerical'])

        self.preprocessor = numerical_pipeline

        # Where to get the data -- Prioritize X_train if any else
        # get from backend
        # TODO consider how to handle the inconsistency between Transformer and Datasets
        X_train = X['backend'].load_datamanager().train_tensors[0]
        """
        if 'X_train' in X:
            X_train = subsampler(X['X_train'], X['train_indices'])
        else:
            X_train = X['backend'].load_datamanager().train_tensors[0]
        """
        self.preprocessor.fit(X_train)
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the time series transformer to fit dictionary
        Args:
            X (Dict[str, Any]): fit dictionary

        Returns:
            X (Dict[str, Any]): updated fit dictionary
        """
        X.update({'time_series_transformer': self})
        return X

    def __call__(self, X: Union[np.ndarray, torch.tensor]) -> Union[np.ndarray, torch.tensor]:

        if self.preprocessor is None:
            raise ValueError("cant call {} without fitting the column transformer first."
                             .format(self.__class__.__name__))

        #if len(X.shape) == 2:
        #    # expand batch dimension when called on a single record
        #    X = X[np.newaxis, ...]

        return self.preprocessor.transform(X)
