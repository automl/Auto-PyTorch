from typing import Any, Dict, List, Optional, Union

import numpy as np

from sklearn.pipeline import Pipeline, make_pipeline
#from sktime.transformations.panel.compose import ColumnTransformer
from sklearn.compose import ColumnTransformer

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
        self.loc = 0.
        self.scale = 1.

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
        numerical_pipeline = 'drop'
        categorical_pipeline = 'drop'

        preprocessors = get_time_series_preprocessers(X)

        if len(X['dataset_properties']['numerical_columns']):
            numerical_pipeline = make_pipeline(*preprocessors['numerical'])
        if len(X['dataset_properties']['categorical_columns']):
            categorical_pipeline = make_pipeline(*preprocessors['categorical'])

        # as X_train is a 2d array here, we simply use ColumnTransformer from sklearn
        self.preprocessor = ColumnTransformer([
            ('numerical_pipeline', numerical_pipeline, X['dataset_properties']['numerical_columns']),
            ('categorical_pipeline', categorical_pipeline, X['dataset_properties']['categorical_columns'])],
            remainder='passthrough'
        )

        """
        # Where to get the data -- Prioritize X_train if any else
        # get from backend
        if 'X_train' in X:
            X_train = subsampler(X['X_train'], X['train_indices'])
        else:
            X_train = X['backend'].load_datamanager().train_tensors[0]
        """
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
        self.preprocessor.fit(X)

        #if len(X.shape) == 2:
        #    # expand batch dimension when called on a single record
        #    X = X[np.newaxis, ...]

        scaler = self.preprocessor.named_transformers_['numerical_pipeline']['timeseriesscaler']
        loc = scaler.loc
        scale = scaler.scale

        return self.preprocessor.transform(X), loc, scale
