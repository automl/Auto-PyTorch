import json
import logging.handlers
import os as os
from abc import abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from sklearn.utils import check_random_state

from autoPyTorch.metrics import accuracy
from autoPyTorch.utils.logging_ import get_named_client_logger


class BaseClassifier:
    """
    Base class for classifiers.
    """

    def __init__(self, logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 random_state: Optional[np.random.RandomState] = None, name: str = ''):

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
        self.random_state = random_state
        self.config = self.get_config()

        self.categoricals: np.ndarray = np.array(())
        self.all_nan: np.ndarray = np.array(())
        self.encode_dicts: List = []
        self.num_classes: Optional[int] = None

        self.metric = accuracy

    def get_config(self) -> Dict[str, Any]:
        """
        Load the parameters for the classifier model from ../classifier_configs/modelname.json.
        """
        dirname = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(dirname, "../classifier_configs", self.name + ".json")
        with open(config_path, "r") as f:
            config = json.load(f)
        for k, v in config.items():
            if v == "True":
                config[k] = True
            if v == "False":
                config[k] = False
        return config

    @abstractmethod
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray) -> Dict[str, Any]:
        """
        Fit the model (possible using the validation set for early stopping) and
        return the results on the training and validation set.
        """
        raise NotImplementedError

    @abstractmethod
    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Score the model performance on a test set.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test: np.ndarray, predict_proba: bool = False) -> np.ndarray:
        """
        predict the model performance on a test set.
        """
        raise NotImplementedError
