import os
import sys
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

import pandas as pd

import torch

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent
from autoPyTorch.pipeline.components.setup.traditional_ml.classifier_models.base_classifier import BaseClassifier
from autoPyTorch.utils.common import FitRequirement


# Disable
def blockPrint() -> None:
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint() -> None:
    sys.stdout = sys.__stdout__


class BaseModelComponent(autoPyTorchSetupComponent):
    """
    Provide an abstract interface for traditional classification methods
    in Auto-Pytorch
    """

    def __init__(
            self,
            random_state: Optional[np.random.RandomState] = None,
            model: Optional[BaseClassifier] = None,
            device: Optional[torch.device] = None
    ) -> None:
        super(BaseModelComponent, self).__init__()
        self.random_state = random_state
        self.fit_output: Dict[str, Any] = dict()

        self.model: Optional[BaseClassifier] = model

        self.add_fit_requirements([
            FitRequirement('X_train', (np.ndarray, list, pd.DataFrame), user_defined=False, dataset_property=False),
            FitRequirement('y_train', (np.ndarray, list, pd.Series,), user_defined=False, dataset_property=False),
            FitRequirement('train_indices', (np.ndarray, list), user_defined=False, dataset_property=False),
            FitRequirement('val_indices', (np.ndarray, list), user_defined=False, dataset_property=False)])

    def fit(self, X: Dict[str, Any], y: Any = None) -> autoPyTorchSetupComponent:
        """
        Fits a component by using an input dictionary with pre-requisites

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API

        Returns:
            A instance of self
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
                                      logger_port=X['logger_port'],
                                      output_shape=output_shape)

        # train model
        blockPrint()
        self.fit_output = self.model.fit(X['X_train'][X['train_indices']], X['y_train'][X['train_indices']],
                                         X['X_train'][X['val_indices']], X['y_train'][X['val_indices']])
        enablePrint()

        # infer
        if 'X_test' in X.keys() and X['X_test'] is not None:
            if isinstance(X['X_test'], pd.DataFrame):
                X['X_test'] = X['X_test'].to_numpy()
            test_preds = self.model.predict(X_test=X['X_test'], predict_proba=True)
            self.fit_output["test_preds"] = test_preds
        return self

    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...],
                    logger_port: int) -> BaseClassifier:
        """
        This method returns a pytorch model, that is dynamically built using
        a self.config that is model specific, and contains the additional
        configuration hyperparameters to build a domain specific model
        """
        raise NotImplementedError()

    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        assert self.model is not None, "Cant predict without fitting first"
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
        return self.model.predict(X_test=X_test).reshape((-1, 1))

    def predict_proba(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        assert self.model is not None, "Cant predict without fitting first"
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
        return self.model.predict(X_test, predict_proba=True)

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        The transform function updates the model in the X dictionary.
        """
        X.update({'model': self.model})
        X.update({'results': self.fit_output})
        return X

    def get_model(self) -> BaseClassifier:
        """
        Return the underlying model object.
        Returns:
            model : the underlying model object
        """
        assert self.model is not None, "No model was initialized"
        return self.model

    def check_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
        """
        This common utility makes sure that the input dictionary X,
        used to fit a given component class, contains the minimum information
        to fit the given component, and it's parents
        """

        # Honor the parent requirements
        super().check_requirements(X, y)

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.model.__class__.__name__
        return string
