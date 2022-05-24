from collections import Counter
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from autoPyTorch.ensemble.abstract_ensemble import AbstractEnsemble
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_loss


class EnsembleSelection(AbstractEnsemble):
    def __init__(
        self,
        ensemble_size: int,
        metric: autoPyTorchMetric,
        task_type: int,
        random_state: np.random.RandomState,
        metric_kwargs: Dict = {},
    ) -> None:
        self.ensemble_size = ensemble_size
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.random_state = random_state
        self.task_type = task_type

    def __getstate__(self) -> Dict[str, Any]:
        # Cannot serialize a metric if
        # it is user defined.
        # That is, if doing pickle dump
        # the metric won't be the same as the
        # one in __main__. we don't use the metric
        # in the EnsembleSelection so this should
        # be fine
        self.metric = None  # type: ignore
        return self.__dict__

    def fit(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
        identifiers: List[Tuple[int, int, float]],
    ) -> AbstractEnsemble:
        """
        Builds a ensemble given the individual models out of fold predictions.
        Fundamentally, defines a set of weights on how to perform a soft-voting
        aggregation of the models in the given identifiers.

        Args:
            predictions (List[np.ndarray]):
                A list of individual model predictions of shape (n_datapoints, n_targets)
                corresponding to the OutOfFold estimate of the ground truth
            labels (np.ndarray):
                The ground truth targets of shape (n_datapoints, n_targets)
            identifiers: List[Tuple[int, int, float]]
                A list of model identifiers, each with the form
                (seed, number of run, budget)

        Returns:
            A copy of self
        """
        self.ensemble_size = int(self.ensemble_size)
        if self.ensemble_size < 1:
            raise ValueError('Ensemble size cannot be less than one!')

        self._fit(predictions, labels)
        self._calculate_weights()
        self.identifiers_ = identifiers
        return self

    def _fit(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
    ) -> None:
        """
        Fast version of Rich Caruana's ensemble selection method.

        For more details, please check the paper
        "Ensemble Selection from Library of Models" by R Caruana  (2004)

        Args:
            predictions (List[np.ndarray]):
                A list of individual model predictions of shape (n_datapoints, n_targets)
                corresponding to the OutOfFold estimate of the ground truth
            identifiers (List[Tuple[int, int, float]]):
                A list of model identifiers, each with the form
                (seed, number of run, budget)
        """
        self.num_input_models_ = len(predictions)

        ensemble: List[np.ndarray] = []
        trajectory = []
        order = []

        ensemble_size = self.ensemble_size

        weighted_ensemble_prediction = np.zeros(
            predictions[0].shape,
            dtype=np.float64,
        )
        fant_ensemble_prediction = np.zeros(
            weighted_ensemble_prediction.shape,
            dtype=np.float64,
        )
        for i in range(ensemble_size):
            losses = np.zeros(
                (len(predictions)),
                dtype=np.float64,
            )
            s = len(ensemble)
            if s > 0:
                np.add(
                    weighted_ensemble_prediction,
                    ensemble[-1],
                    out=weighted_ensemble_prediction,
                )

            # Memory-efficient averaging!
            for j, pred in enumerate(predictions):
                # fant_ensemble_prediction is the prediction of the current ensemble
                # and should be ([predictions[selected_prev_iterations] + predictions[j])/(s+1)
                # We overwrite the contents of fant_ensemble_prediction
                # directly with weighted_ensemble_prediction + new_prediction and then scale for avg
                np.add(
                    weighted_ensemble_prediction,
                    pred,
                    out=fant_ensemble_prediction
                )
                np.multiply(
                    fant_ensemble_prediction,
                    (1. / float(s + 1)),
                    out=fant_ensemble_prediction
                )

                # Calculate loss is versatile and can return a dict of slosses
                losses[j] = calculate_loss(
                    metrics=[self.metric],
                    target=labels,
                    prediction=fant_ensemble_prediction,
                    task_type=self.task_type,
                    **self.metric_kwargs
                )[self.metric.name]

            all_best = np.argwhere(losses == np.nanmin(losses)).flatten()
            best = self.random_state.choice(all_best)
            ensemble.append(predictions[best])
            trajectory.append(losses[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

        self.indices_: List[int] = order
        self.trajectory_: List[float] = trajectory
        self.train_loss_: float = trajectory[-1]

    def _calculate_weights(self) -> None:
        """
        Calculates the contribution each of the individual models
        should have, in the final ensemble soft voting. It does so by
        a frequency counting scheme. In particular, how many times a model
        was used during hill climbing optimization.
        """
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros(
            (self.num_input_models_,),
            dtype=np.float64,
        )
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.ensemble_size
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights

    def predict(self, predictions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Given a list of predictions from the individual model, this method
        aggregates the predictions using a soft voting scheme with the weights
        found during training.

        Args:
            predictions (List[np.ndarray]):
                A list of predictions from the individual base models.

        Returns:
            average (np.ndarray): Soft voting predictions of ensemble models, using
                                the weights found during ensemble selection (self._weights)
        """

        average = np.zeros_like(predictions[0], dtype=np.float64)
        tmp_predictions = np.empty_like(predictions[0], dtype=np.float64)

        # if predictions.shape[0] == len(self.weights_),
        # predictions include those of zero-weight models.
        if len(predictions) == len(self.weights_):
            for pred, weight in zip(predictions, self.weights_):
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)

        # if prediction model.shape[0] == len(non_null_weights),
        # predictions do not include those of zero-weight models.
        elif len(predictions) == np.count_nonzero(self.weights_):
            non_null_weights = [w for w in self.weights_ if w > 0]
            for pred, weight in zip(predictions, non_null_weights):
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)

        # If none of the above applies, then something must have gone wrong.
        else:
            raise ValueError("The dimensions of ensemble predictions"
                             " and ensemble weights do not match!")
        del tmp_predictions
        return average

    def __str__(self) -> str:
        return 'Ensemble Selection:\n\tTrajectory: %s\n\tMembers: %s' \
               '\n\tWeights: %s\n\tIdentifiers: %s' % \
               (' '.join(['%d: %5f' % (idx, performance)
                         for idx, performance in enumerate(self.trajectory_)]),
                self.indices_, self.weights_,
                ' '.join([str(identifier) for idx, identifier in
                          enumerate(self.identifiers_)
                          if self.weights_[idx] > 0]))

    def get_models_with_weights(
        self,
        models: Dict[Any, BasePipeline]
    ) -> List[Tuple[float, BasePipeline]]:
        """
        Handy function to tag the provided input models with a given weight.

        Args:
            models (List[Tuple[float, BasePipeline]]):
                A dictionary that maps a model's name to it's actual python object.

        Returns:
            output (List[Tuple[float, BasePipeline]]):
                each model with the related weight, sorted by ascending
                performance. Notice that ensemble selection solves a minimization
                problem.
        """
        output = []
        for i, weight in enumerate(self.weights_):
            if weight > 0.0:
                identifier = self.identifiers_[i]
                model = models[identifier]
                output.append((weight, model))

        output.sort(reverse=True, key=lambda t: t[0])

        return output

    def get_selected_model_identifiers(self) -> List[Tuple[int, int, float]]:
        """
        After training of ensemble selection, not all models will be used.
        Some of them will have zero weight. This procedure filters this models
        out.

        Returns:
            output (List[Tuple[int, int, float]]):
                The models actually used by ensemble selection
        """
        output = []

        for i, weight in enumerate(self.weights_):
            identifier = self.identifiers_[i]
            if weight > 0.0:
                output.append(identifier)

        return output

    def get_validation_performance(self) -> float:
        """
        Returns the best optimization performance seen during hill climbing

        Returns:
            (float):
                best ensemble training performance
        """
        return self.trajectory_[-1]
