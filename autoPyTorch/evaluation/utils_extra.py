# The functions and classes implemented in this module all require extra requirements.
# We put them here to make it easier to be wrapped by try-except process
from typing import Any, Dict, List, Optional, Union

from ConfigSpace import Configuration

import numpy as np

import pandas as pd

from sklearn.dummy import DummyClassifier

from autoPyTorch.datasets.time_series_dataset import TimeSeriesSequence
from autoPyTorch.utils.common import subsampler


class DummyTimeSeriesForecastingPipeline(DummyClassifier):
    """
    A wrapper class that holds a pipeline for dummy forecasting. For each series, it simply repeats the last element
    in the training series


    Attributes:
        random_state (Optional[Union[int, np.random.RandomState]]):
            Object that contains a seed and allows for reproducible results
        init_params  (Optional[Dict]):
            An optional dictionary that is passed to the pipeline's steps. It complies
            a similar function as the kwargs
        n_prediction_steps (int):
            forecasting horizon
    """
    def __init__(self, config: Configuration,
                 random_state: Optional[Union[int, np.random.RandomState]] = None,
                 init_params: Optional[Dict] = None,
                 n_prediction_steps: int = 1,
                 ) -> None:
        self.config = config
        self.init_params = init_params
        self.random_state = random_state
        super(DummyTimeSeriesForecastingPipeline, self).__init__(strategy="uniform")
        self.n_prediction_steps = n_prediction_steps

    def fit(self, X: Dict[str, Any], y: Any,
            sample_weight: Optional[np.ndarray] = None) -> object:
        self.n_prediction_steps = X['dataset_properties']['n_prediction_steps']
        y_train = subsampler(X['y_train'], X['train_indices'])
        return DummyClassifier.fit(self, np.ones((y_train.shape[0], 1)), y_train, sample_weight)

    def _generate_dummy_forecasting(self, X: List[Union[TimeSeriesSequence, np.ndarray]]) -> List:
        if isinstance(X[0], TimeSeriesSequence):
            X_tail = [x.get_target_values(-1) for x in X]
        else:
            X_tail = [x[-1] for x in X]
        return X_tail

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame],
                      batch_size: int = 1000) -> np.ndarray:
        X_tail = self._generate_dummy_forecasting(X)
        return np.tile(X_tail, (1, self.n_prediction_steps)).astype(np.float32).flatten()

    def predict(self, X: Union[np.ndarray, pd.DataFrame],
                batch_size: int = 1000) -> np.ndarray:
        X_tail = np.asarray(self._generate_dummy_forecasting(X))
        if X_tail.ndim == 1:
            X_tail = np.expand_dims(X_tail, -1)
        return np.tile(X_tail, (1, self.n_prediction_steps)).astype(np.float32).flatten()

    @staticmethod
    def get_default_pipeline_options() -> Dict[str, Any]:
        return {'budget_type': 'epochs',
                'epochs': 1,
                'runtime': 1}
