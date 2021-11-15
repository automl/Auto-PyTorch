from abc import ABCMeta
from typing import Any, Callable, List, Optional

import numpy as np

import sklearn.metrics
from sklearn.utils.multiclass import type_of_target


class autoPyTorchMetric(object, metaclass=ABCMeta):

    def __init__(self,
                 name: str,
                 score_func: Callable[..., float],
                 optimum: float,
                 worst_possible_result: float,
                 sign: float,
                 kwargs: Any) -> None:
        self.name = name
        self._kwargs = kwargs
        self._metric_func = score_func
        self._optimum = optimum
        self._worst_possible_result = worst_possible_result
        self._sign = sign

    def __call__(self,
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 sample_weight: Optional[List[float]] = None
                 ) -> float:
        raise NotImplementedError()

    def get_metric_func(self) -> Callable:
        return self._metric_func

    def __repr__(self) -> str:
        return self.name


class _PredictMetric(autoPyTorchMetric):
    def __call__(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            sample_weight: Optional[List[float]] = None
    ) -> float:
        """
        Evaluate predicted target values for X relative to y_true.

        Args:
            y_true (np.ndarray):
                Gold standard target values for X.
            y_pred (np.ndarray):
                [n_samples x n_classes], Model predictions
            sample_weight (Optional[np.ndarray]):
                Sample weights.

        Returns:
            score (float):
                Score function applied to prediction of estimator on X.
        """
        type_true = type_of_target(y_true)
        if type_true == 'binary' and type_of_target(y_pred) == 'continuous' and \
                len(y_pred.shape) == 1:
            # For a pred autoPyTorchMetric, no threshold, nor probability is required
            # If y_true is binary, and y_pred is continuous
            # it means that a rounding is necessary to obtain the binary class
            y_pred = np.around(y_pred, decimals=0)
        elif len(y_pred.shape) == 1 or y_pred.shape[1] == 1 or \
                type_true == 'continuous':
            # must be regression, all other task types would return at least
            # two probabilities
            pass
        elif type_true in ['binary', 'multiclass']:
            y_pred = np.argmax(y_pred, axis=1)
        elif type_true == 'multilabel-indicator':
            y_pred[y_pred > 0.5] = 1.0
            y_pred[y_pred <= 0.5] = 0.0
        elif type_true == 'continuous-multioutput':
            pass
        else:
            raise ValueError(type_true)

        if sample_weight is not None:
            return self._sign * self._metric_func(y_true, y_pred,
                                                  sample_weight=sample_weight,
                                                  **self._kwargs)
        else:
            return self._sign * self._metric_func(y_true, y_pred,
                                                  **self._kwargs)


class _ProbaMetric(autoPyTorchMetric):
    def __call__(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            sample_weight: Optional[List[float]] = None
    ) -> float:

        """
        Evaluate predicted probabilities for X relative to y_true.
        Args:
            y_true (np.ndarray):
                Gold standard target values for X. These must be class labels,
                not probabilities.
            y_pred (np.ndarray):
                [n_samples x n_classes], Model predictions
            sample_weight (Optional[np.ndarray]):
                Sample weights.

        Returns:
            score (float):
                Score function applied to prediction of estimator on X.
        """

        if self._metric_func is sklearn.metrics.log_loss:
            n_labels_pred = np.array(y_pred).reshape((len(y_pred), -1)).shape[1]
            n_labels_test = len(np.unique(y_true))
            if n_labels_pred != n_labels_test:
                labels = list(range(n_labels_pred))
                if sample_weight is not None:
                    return self._sign * self._metric_func(y_true, y_pred,
                                                          sample_weight=sample_weight,
                                                          labels=labels,
                                                          **self._kwargs)
                else:
                    return self._sign * self._metric_func(y_true, y_pred,
                                                          labels=labels, **self._kwargs)

        if sample_weight is not None:
            return self._sign * self._metric_func(y_true, y_pred,
                                                  sample_weight=sample_weight,
                                                  **self._kwargs)
        else:
            return self._sign * self._metric_func(y_true, y_pred,
                                                  **self._kwargs)


class _ThresholdMetric(autoPyTorchMetric):
    def __call__(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            sample_weight: Optional[List[float]] = None
    ) -> float:
        """Evaluate decision function output for X relative to y_true.
        Args:
            y_true (np.ndarray):
                Gold standard target values for X. These must be class labels,
                not probabilities.
            y_pred (np.ndarray):
                [n_samples x n_classes], Model predictions
            sample_weight (Optional[np.ndarray]):
                Sample weights.

        Returns:
            score (float):
                Score function applied to prediction of estimator on X.
        """
        y_type = type_of_target(y_true)
        if y_type not in ("binary", "multilabel-indicator"):
            raise ValueError("{0} format is not supported".format(y_type))

        if y_type == "binary":
            if y_pred.ndim > 1:
                y_pred = y_pred[:, 1]
        elif isinstance(y_pred, list):
            y_pred = np.vstack([p[:, -1] for p in y_pred]).T

        if sample_weight is not None:
            return self._sign * self._metric_func(y_true, y_pred,
                                                  sample_weight=sample_weight,
                                                  **self._kwargs)
        else:
            return self._sign * self._metric_func(y_true, y_pred, **self._kwargs)


def make_metric(
    name: str,
    score_func: Callable,
    optimum: float = 1.0,
    worst_possible_result: float = 0.0,
    greater_is_better: bool = True,
    needs_proba: bool = False,
    needs_threshold: bool = False,
    **kwargs: Any
) -> autoPyTorchMetric:
    """
    Make a autoPyTorchMetric from a performance metric or loss function.
    Factory inspired by scikit-learn which wraps scikit-learn scoring functions
    to be used in autoPyTorch.

    Args:
        name (str):
            Name of the metric
        score_func (Callable):
            Score function (or loss function) with signature
            ``score_func(y, y_pred, **kwargs)``.
        optimum (Union[int, float]: default=1):
            The best score achievable by the score function, i.e. maximum in case of
            metric function and minimum in case of loss function.
        greater_is_better (bool: default=True):
            Whether score_func is a score function (default), meaning high is good,
            or a loss function, meaning low is good. In the latter case, the
            autoPyTorchMetric object will sign-flip the outcome of the score_func.
        needs_proba (bool: default=False):
            Whether score_func requires predict_proba to get probability estimates
            out of a classifier.
        needs_threshold (bool: default=True):
            Whether score_func takes a continuous decision certainty.
            This only works for binary classification.
        **kwargs : additional arguments
            Additional parameters to be passed to score_func.

    Returns
        autoPyTorchMetric (Callable):
            Callable object that returns a scalar score; greater is better.

    """
    sign = 1 if greater_is_better else -1
    if needs_proba:
        return _ProbaMetric(name, score_func, optimum, worst_possible_result, sign, kwargs)
    elif needs_threshold:
        return _ThresholdMetric(name, score_func, optimum, worst_possible_result, sign, kwargs)
    else:
        return _PredictMetric(name, score_func, optimum, worst_possible_result, sign, kwargs)
