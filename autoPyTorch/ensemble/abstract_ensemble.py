from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from autoPyTorch.pipeline.base_pipeline import BasePipeline


class AbstractEnsemble(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.identifiers_: List[Tuple[int, int, float]] = []

    @abstractmethod
    def fit(
        self,
        base_models_predictions: np.ndarray,
        true_targets: np.ndarray,
        model_identifiers: List[Tuple[int, int, float]],
    ) -> 'AbstractEnsemble':
        """Fit an ensemble given predictions of base models and targets.
        Ensemble building maximizes performance (in contrast to
        hyperparameter optimization)!

        Args:
            base_models_predictions (np.ndarray):
                array of shape = [n_base_models, n_data_points, n_targets]
                This are the predictions of the individual models found by SMAC
            true_targets (np.ndarray) : array of shape [n_targets]
                This is the ground truth of the above predictions
            model_identifiers (List[Tuple[int, int, float]]): identifier for each base model.
                Can be used for practical text output of the ensemble.

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, base_models_predictions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Create ensemble predictions from the base model predictions.

        Args:
            base_models_predictions (Union[np.ndarray, List[np.ndarray]]) : array of
                shape = [n_base_models, n_data_points, n_targets]
                Same as in the fit method.

        Returns:
            predicted array
        """
        self

    @abstractmethod
    def get_models_with_weights(self, models: Dict[Any, BasePipeline]) -> List[Tuple[float, BasePipeline]]:
        """Return a list of (weight, model) pairs

        Args:
            models : dict {identifier : model object}
                The identifiers are the same as the one presented to the fit()
                method. Models can be used for nice printing.
        Returns:
            array of weights : [(weight_1, model_1), ..., (weight_n, model_n)]
        """

    @abstractmethod
    def get_selected_model_identifiers(self) -> List[Tuple[int, int, float]]:
        """Return identifiers of models in the ensemble.
        This includes models which have a weight of zero!

        Returns:
            The selected models (seed, idx, budget) from smac
        """

    @abstractmethod
    def get_validation_performance(self) -> float:
        """Return validation performance of ensemble.

        Returns:
            Score
        """

    def update_identifiers(
        self,
        replace_identifiers_mapping: Dict[Tuple[int, int, float], Tuple[int, int, float]]
    ) -> None:
        identifiers = self.identifiers_.copy()
        for i, identifier in enumerate(self.identifiers_):
            identifiers[i] = replace_identifiers_mapping.get(identifier, identifier)
        self.identifiers_ = identifiers
