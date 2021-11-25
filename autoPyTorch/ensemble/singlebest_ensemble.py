import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from smac.runhistory.runhistory import RunHistory

from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.ensemble.abstract_ensemble import AbstractEnsemble
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric


class SingleBest(AbstractEnsemble):
    """
    In the case of a crash, this class searches
    for the best individual model.
    Such model is returned as an ensemble of a single
    object, to comply with the expected interface of an
    AbstractEnsemble.
    """
    def __init__(
        self,
        metric: autoPyTorchMetric,
        run_history: RunHistory,
        seed: int,
        backend: Backend,
        metric_kwargs: Dict = {},

    ):
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.seed = seed
        self.backend = backend

        # Add some default values -- at least 1 model in ensemble is assumed
        self.indices_ = [0]
        self.weights_ = [1.0]
        self.run_history = run_history
        self.identifiers_ = self.get_identifiers_from_run_history()
        self.best_performance = self.metric._worst_possible_result

    def get_identifiers_from_run_history(self) -> List[Tuple[int, int, float]]:
        """
        This method parses the run history, to identify
        the best performing model
        It populates the identifiers attribute, which is used
        by the backend to access the actual model
        """
        best_model_identifier = []
        best_model_score = self.metric._worst_possible_result

        for run_key in self.run_history.data.keys():
            run_value = self.run_history.data[run_key]
            score = self.metric._optimum - (self.metric._sign * run_value.cost)

            if (score > best_model_score and self.metric._sign > 0) \
                    or (score < best_model_score and self.metric._sign < 0):

                # Make sure that the individual best model actually exists
                model_dir = self.backend.get_numrun_directory(
                    self.seed,
                    run_value.additional_info['num_run'],
                    run_key.budget,
                )
                model_file_name = self.backend.get_model_filename(
                    self.seed,
                    run_value.additional_info['num_run'],
                    run_key.budget,
                )
                file_path = os.path.join(model_dir, model_file_name)
                if not os.path.exists(file_path):
                    continue

                best_model_identifier = [(
                    self.seed,
                    run_value.additional_info['num_run'],
                    run_key.budget,
                )]
                best_model_score = score

        if not best_model_identifier:
            raise ValueError(
                "No valid model found in run history. This means smac was not able to fit"
                " a valid model. Please check the log file for errors."
            )

        self.best_performance = best_model_score

        return best_model_identifier

    def predict(self, predictions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        return predictions[0]

    def __str__(self) -> str:
        return 'Single Model Selection:\n\tMembers: %s' \
               '\n\tWeights: %s\n\tIdentifiers: %s' % \
               (self.indices_, self.weights_,
                ' '.join([str(identifier) for idx, identifier in
                          enumerate(self.identifiers_)
                          if self.weights_[idx] > 0]))

    def get_models_with_weights(self, models: Dict[Any, BasePipeline]
                                ) -> List[Tuple[float, BasePipeline]]:
        output = []
        for i, weight in enumerate(self.weights_):
            if weight > 0.0:
                identifier = self.identifiers_[i]
                model = models[identifier]
                output.append((weight, model))

        output.sort(reverse=True, key=lambda t: t[0])

        return output

    def get_selected_model_identifiers(self) -> List[Tuple[int, int, float]]:
        output = []

        for i, weight in enumerate(self.weights_):
            identifier = self.identifiers_[i]
            if weight > 0.0:
                output.append(identifier)

        return output

    def fit(
        self,
        base_models_predictions: np.ndarray,
        true_targets: np.ndarray,
        model_identifiers: List[Tuple[int, int, float]],
    ) -> 'SingleBest':
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
        return self

    def get_validation_performance(self) -> float:
        """Return validation performance of ensemble.

        Returns:
            Score
        """
        return self.best_performance
