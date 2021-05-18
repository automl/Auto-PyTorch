import json
import logging.handlers
import os as os
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from catboost import CatBoost

import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from autoPyTorch.constants import STRING_TO_TASK_TYPES, REGRESSION_TASKS
from autoPyTorch.pipeline.components.training.metrics.utils import get_metrics
from autoPyTorch.utils.logging_ import get_named_client_logger


class BaseTraditionalLearner:
    """
    Base class for classifiers.
    """

    def __init__(self,
                 task_type: str,
                 output_type: str,
                 logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 random_state: Optional[np.random.RandomState] = None,
                 name: str = ''):

        self.model: Optional[Union[CatBoost, BaseEstimator]] = None

        self.name = name
        self.logger_port = logger_port
        self.logger = get_named_client_logger(
            name=name,
            host='localhost',
            port=logger_port,
        )

        if random_state is None:
            self.random_state = check_random_state(1)
        else:
            self.random_state = check_random_state(random_state)
        self.config = self.get_config()

        self.all_nan: np.ndarray = np.array(())
        self.num_classes: Optional[int] = None

        self.is_classification = STRING_TO_TASK_TYPES[task_type] not in REGRESSION_TASKS

        self.metric = get_metrics(dataset_properties={'task_type': task_type,
                                                      'output_type': output_type})[0]

    def get_config(self) -> Dict[str, Any]:
        """
        Load the parameters for the classifier model from ../estimator_configs/modelname.json.
        """
        dirname = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(dirname, "../estimator_configs", self.name + ".json")
        with open(config_path, "r") as f:
            config = json.load(f)
        for k, v in config.items():
            if v == "True":
                config[k] = True
            if v == "False":
                config[k] = False
        return config

    def _preprocess(self,
                    X_train: np.ndarray,
                    X_val: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
        self.all_nan = np.all(pd.isnull(X_train), axis=0)
        X_train = X_train[:, ~self.all_nan]
        X_val = X_val[:, ~self.all_nan]

        X_train = np.nan_to_num(X_train)
        X_val = np.nan_to_num(X_val)
        return X_train, X_val

    @abstractmethod
    def _prepare_model(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray
                       ):
        raise NotImplementedError

    @abstractmethod
    def _fit(self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray):

        raise NotImplementedError

    def fit(self, X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray) -> Dict[str, Any]:
        """
             Fit the model (possible using the validation set for early stopping) and
             return the results on the training and validation set.
        """
        X_train, X_val = self._preprocess(X_train, X_val)

        self._prepare_model(X_train, y_train)

        self._fit(X_train, y_train, X_val, y_val)

        results = self._get_results(X_train, y_train, X_val, y_val)

        return results

    def score(self, X_test: np.ndarray, y_test: Union[np.ndarray, List]) -> float:
        """
        Score the model performance on a test set.
        """
        y_pred = self.predict(X_test, predict_proba=self.is_classification)
        return self.metric(y_test, y_pred)

    def predict(self, X_test: np.ndarray,
                predict_proba: bool = False,
                preprocess: bool = True) -> np.ndarray:
        """
        predict the model performance on a test set.
        """
        if preprocess:
            X_test = X_test[:, ~self.all_nan]
            X_test = np.nan_to_num(X_test)
        if predict_proba:
            if not self.is_classification:
                raise ValueError("Can't predict probabilities for a regressor")
            return self.model.predict_proba(X_test)
        y_pred = self.model.predict(X_test)
        return y_pred

    def _get_results(self,
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_val: np.ndarray,
                     y_val: np.ndarray) -> Dict[str, Any]:

        pred_train = self.predict(X_train, predict_proba=self.is_classification, preprocess=False)
        pred_val = self.predict(X_val, predict_proba=self.is_classification, preprocess=False)

        results = dict()

        results["val_preds"] = pred_val.tolist()
        results["labels"] = y_val.tolist()

        results["train_score"] = self.metric(y_train, pred_train)
        results["val_score"] = self.metric(y_val, pred_val)

        return results
