import json
import logging.handlers
import os as os
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from catboost import CatBoost

import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from autoPyTorch.constants import REGRESSION_TASKS, STRING_TO_TASK_TYPES
from autoPyTorch.pipeline.components.training.metrics.utils import get_metrics
from autoPyTorch.utils.logging_ import get_named_client_logger


class BaseTraditionalLearner:
    """
    Base wrapper class for Traditional Learners.

    Args:
        task_type (str):
            Type of the current task. Currently only tabular
            tasks are supported. For more info on the tasks
            available in AutoPyTorch, see
            `autoPyTorch/constants.py`
        output_type (str):
            Type of output. The string depends on the output of
            sklearn's type_of_target. `see
            <https://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.type_of_target.html>`
        logger_port (int) (default=logging.handlers.DEFAULT_TCP_LOGGING_PORT)
        random_state (Optional[np.random.RandomState]):
        name (str, default=''):
            Name of the learner, when not specified,
            uses the name of the class
    """

    def __init__(self,
                 task_type: str,
                 output_type: str,
                 optimize_metric: Optional[str] = None,
                 logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 random_state: Optional[np.random.RandomState] = None,
                 name: Optional[str] = None):

        self.model: Optional[Union[CatBoost, BaseEstimator]] = None

        self.name = name if name is not None else self.__class__.__name__
        self.logger_port = logger_port
        self.logger = get_named_client_logger(
            name=self.name,
            host='localhost',
            port=logger_port,
        )

        if random_state is None:
            self.random_state = check_random_state(1)
        else:
            self.random_state = check_random_state(random_state)
        self.config = self.get_config()

        self.all_nan: Optional[np.ndarray] = None
        self.num_classes: Optional[int] = None

        self.is_classification = STRING_TO_TASK_TYPES[task_type] not in REGRESSION_TASKS

        self.has_val_set = False

        self.metric = get_metrics(dataset_properties={'task_type': task_type,
                                                      'output_type': output_type},
                                  names=[optimize_metric] if optimize_metric is not None else None)[0]

    def get_config(self) -> Dict[str, Union[int, str, float, bool]]:
        """
        Load the parameters for the classifier model from ../estimator_configs/modelname.json.
        """
        dirname = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(dirname, "../estimator_configs", self.name + ".json")
        with open(config_path, "r") as f:
            config: Dict[str, Union[int, str, float, bool]] = json.load(f)
        for k, v in config.items():
            if v == "True":
                config[k] = True
            if v == "False":
                config[k] = False
        return config

    def _preprocess(self,
                    X: np.ndarray
                    ) -> np.ndarray:
        """
        Preprocess the input set, currently imputes the nan columns.
        Can be used to add more preprocessing functionality
        Args:
            X (np.ndarray):
                input data
        Returns:
            (np.ndarray):
                Output data
        """
        if self.all_nan is None:
            self.all_nan = np.all(pd.isnull(X), axis=0)

        X = X[:, ~self.all_nan]
        X = np.nan_to_num(X, copy=False)

        return X

    @abstractmethod
    def _prepare_model(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray
                       ) -> None:
        """
        Abstract method to prepare model. Depending on the
        learner, this function will initialise the underlying
        estimator and the objects needed to do that

        Args:
            X_train (np.ndarray):
                Input training data
            y_train (np.ndarray):
                Target training data
        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def _fit(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None) -> None:
        """
        Method that fits the underlying estimator
        Args:
            X_train (np.ndarray):
                Input training data
            y_train (np.ndarray):
                Target training data
            X_val (np.ndarray):
                Input validation data
            y_val (np.ndarray):
                Output validation data
        Returns:
            None
        """
        raise NotImplementedError

    def fit(self, X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Fit the model (possible using the validation set for early stopping) and
        return the results on the training and validation set.

        Args:
            X_train (np.ndarray):
                Input training data
            y_train (np.ndarray):
                Target training data
            X_val (np.ndarray):
                Input validation data
            y_val (np.ndarray):
                Output validation data
        Returns:
            Dict[str, Any]:
                Dictionary containing the results. see _get_results()
        """
        X_train = self._preprocess(X_train)

        if X_val is not None:
            self.has_val_set = True
            X_val = self._preprocess(X_val)

        self._prepare_model(X_train, y_train)

        self._fit(X_train, y_train, X_val, y_val)

        results = self._get_results(X_train, y_train, X_val, y_val)

        return results

    def score(self, X_test: np.ndarray, y_test: Union[np.ndarray, List]) -> float:
        """
        Score the model performance on a test set.

        Args:
            X_test (np.ndarray):
                Input data
            y_test (Union[np.ndarray, List]):
                Target data
        Returns:
            float: score on the selected metric
        """
        y_pred = self.predict(X_test, predict_proba=self.is_classification)
        return self.metric(y_test, y_pred)

    def predict(self, X_test: np.ndarray,
                predict_proba: bool = False,
                preprocess: bool = True) -> np.ndarray:
        """
        predict the model performance on a test set.

        Args:
            X_test (np.ndarray):
                Input data
            predict_proba (bool, default=False):
                if task is a classification task,
                predict the class probabilities
            preprocess (bool, default=True):
                Whether to preprocess data or not
        Returns:

        """
        assert self.model is not None, "No model found. Can't " \
                                       "predict before fitting. " \
                                       "Call fit before predicting"
        if preprocess:
            X_test = self._preprocess(X_test)
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
        """
        Gather results of the training.
        The following results are calculated:
            1. val_preds: validation predictions
            2. train_preds: training predictions
            3. val_score: score on validation set
            4. train_score: score on the training set

        Args:
            X_train (np.ndarray):
                Input training data
            y_train (np.ndarray):
                Target training data
            X_val (np.ndarray):
                Input validation data
            y_val (np.ndarray):
                Output validation data
        Returns:
            Dict[str, Any]:
                Dictionary containing the results
        """
        pred_train = self.predict(X_train, predict_proba=self.is_classification, preprocess=False)

        results = dict()
        results["train_score"] = self.metric(y_train, pred_train)

        if self.has_val_set:
            pred_val = self.predict(X_val, predict_proba=self.is_classification, preprocess=False)
            results["labels"] = y_val.tolist()
            results["val_preds"] = pred_val.tolist()
            results["val_score"] = self.metric(y_val, pred_val)

        return results
