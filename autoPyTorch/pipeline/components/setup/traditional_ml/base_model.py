import logging.handlers
import os
import sys
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import pandas as pd

from sklearn.utils import check_random_state

import torch

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner.base_traditional_learner import \
    BaseTraditionalLearner
from autoPyTorch.utils.common import FitRequirement


# Disable
def blockPrint() -> None:
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint() -> None:
    sys.stdout = sys.__stdout__


class BaseModelComponent(autoPyTorchSetupComponent):
    """
    Provide an abstract interface for traditional learner methods
    in Auto-Pytorch
    """

    def __init__(
            self,
            random_state: Optional[np.random.RandomState] = None,
            model: Optional[BaseTraditionalLearner] = None,
            device: Optional[torch.device] = None
    ) -> None:
        super(BaseModelComponent, self).__init__()
        if random_state is None:
            self.random_state = check_random_state(1)
        else:
            self.random_state = check_random_state(random_state)
        self.fit_output: Dict[str, Any] = dict()

        self.model: Optional[BaseTraditionalLearner] = model

        self.add_fit_requirements([
            FitRequirement('X_train', (np.ndarray, list, pd.DataFrame), user_defined=False, dataset_property=False),
            FitRequirement('y_train', (np.ndarray, list, pd.Series,), user_defined=False, dataset_property=False),
            FitRequirement('train_indices', (np.ndarray, list), user_defined=False, dataset_property=False)])

    def fit(self, X: Dict[str, Any], y: Any = None) -> autoPyTorchSetupComponent:
        """
        Fits a component by using an input dictionary with pre-requisites

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API

        Returns:
            An instance of self
        """
        # Make sure that input dictionary X has the required
        # information to fit this stage
        self.check_requirements(X, y)

        if isinstance(X['X_train'], pd.DataFrame):
            X['X_train'] = X['X_train'].to_numpy()

        if isinstance(X['y_train'], pd.core.series.Series):
            X['y_train'] = X['y_train'].to_numpy()

        input_shape = X['X_train'].shape[1:]
        output_shape = X['y_train'].shape

        # instantiate model
        self.model = self.build_model(input_shape=input_shape,
                                      logger_port=X['logger_port'] if 'logger_port' in X else
                                      logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                                      output_shape=output_shape,
                                      task_type=X['dataset_properties']['task_type'],
                                      output_type=X['dataset_properties']['output_type'],
                                      optimize_metric=X['optimize_metric'] if 'optimize_metric' in X else None)

        # train model
        blockPrint()
        val_indices = X.get('val_indices', None)
        X_val = None
        y_val = None
        if val_indices is not None:
            X_val = X['X_train'][val_indices]
            y_val = X['y_train'][val_indices]
        self.fit_output = self.model.fit(X['X_train'][X['train_indices']], X['y_train'][X['train_indices']],
                                         X_val, y_val)
        enablePrint()

        # infer
        if 'X_test' in X.keys() and X['X_test'] is not None:
            if isinstance(X['X_test'], pd.DataFrame):
                X['X_test'] = X['X_test'].to_numpy()
            test_preds = self.model.predict(X_test=X['X_test'], predict_proba=self.model.is_classification)
            self.fit_output["test_preds"] = test_preds
        return self

    @abstractmethod
    def build_model(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        logger_port: int,
        task_type: str,
        output_type: str,
        optimize_metric: Optional[str] = None
    ) -> BaseTraditionalLearner:
        """
        This method returns a traditional learner, that is dynamically
        built based on the provided configuration.
        """
        raise NotImplementedError()

    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        assert self.model is not None, "Can't predict without fitting first"
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
        return self.model.predict(X_test=X_test).reshape((-1, 1))

    def predict_proba(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        assert self.model is not None, "Can't predict without fitting first"
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
        return self.model.predict(X_test, predict_proba=True)

    def score(self, X_test: Union[pd.DataFrame, np.ndarray], y_test: Union[pd.Series, np.ndarray, List]) -> float:
        assert self.model is not None, "Can't score without fitting first"
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
        return self.model.score(X_test, y_test)

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        The transform function updates the model and the results in the fit dictionary.
        """
        X.update({'model': self.model})
        X.update({'results': self.fit_output})
        return X

    def get_model(self) -> BaseTraditionalLearner:
        """
        Return the underlying model object.
        Returns:
            model : the underlying model object
        """
        assert self.model is not None, "No model was initialized"
        return self.model

    def check_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
        """
        This common utility makes sure that the input fit dictionary,
        used to fit a given component class, contains the minimum information
        to fit the given component, and it's parents
        """

        # Honor the parent requirements
        super().check_requirements(X, y)

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.model.__class__.__name__
        return string
