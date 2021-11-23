import warnings
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from autoPyTorch.constants import (
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
    FORECASTING_TASKS,
    STRING_TO_TASK_TYPES,
    TASK_TYPES,
)
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.metrics import CLASSIFICATION_METRICS, \
    REGRESSION_METRICS, FORECASTING_METRICS


def sanitize_array(array: np.ndarray) -> np.ndarray:
    """
    Replace NaN and Inf (there should not be any!)
    :param array:z
    :return:
    """
    a = np.ravel(array)
    finite = np.isfinite(a)
    if np.any(finite):
        maxi = np.nanmax(a[finite])
        mini = np.nanmin(a[finite])
    else:
        maxi = mini = 0
    array[array == float('inf')] = maxi
    array[array == float('-inf')] = mini
    mid = (maxi + mini) / 2
    array[np.isnan(array)] = mid
    return array


def get_supported_metrics(dataset_properties: Dict[str, Any]) -> Dict[str, autoPyTorchMetric]:
    task_type = dataset_properties['task_type']

    if STRING_TO_TASK_TYPES[task_type] in REGRESSION_TASKS:
        return REGRESSION_METRICS
    elif STRING_TO_TASK_TYPES[task_type] in CLASSIFICATION_TASKS:
        return CLASSIFICATION_METRICS
    elif STRING_TO_TASK_TYPES[task_type] in FORECASTING_TASKS:
        return FORECASTING_METRICS
    else:
        raise NotImplementedError(task_type)


def get_metrics(dataset_properties: Dict[str, Any],
                names: Optional[Iterable[str]] = None,
                all_supported_metrics: bool = False,
                ) -> List[autoPyTorchMetric]:
    """
    Returns metrics for current task_type, if names is None and
    all_supported_metrics is False, returns preset default for
    given task

    Args:
        dataset_properties: Dict[str, Any]
        contains information about the dataset and task type
        names: Optional[Iterable[str]]
        names of metrics to return
        all_supported_metrics: bool
        if true, returns all metrics that are relevant to task_type

    Returns:

    """
    assert 'task_type' in dataset_properties, \
        "Expected dataset_properties to have task_type got {}".format(dataset_properties.keys())
    assert 'output_type' in dataset_properties, \
        "Expected dataset_properties to have output_type got {}".format(dataset_properties.keys())
    if all_supported_metrics:
        assert names is None, "Can't pass names when all_supported_metrics are true"

    if STRING_TO_TASK_TYPES[dataset_properties['task_type']] not in TASK_TYPES:
        raise NotImplementedError(dataset_properties['task_type'])

    default_metrics = dict(classification=dict({'multiclass': 'accuracy',
                                                'binary': 'accuracy',
                                                'multiclass-multioutput': 'f1'}),
                           regression=dict({'continuous': 'r2',
                                            'continuous-multioutput': 'r2'}))

    supported_metrics = get_supported_metrics(dataset_properties)
    metrics = list()  # type: List[autoPyTorchMetric]
    if names is not None:
        for name in names:
            if name not in supported_metrics.keys():
                raise ValueError("Invalid name entered for task {}, currently "
                                 "supported metrics for task include {}".format(dataset_properties['task_type'],
                                                                                list(supported_metrics.keys())))
            else:
                metric = supported_metrics[name]
                metrics.append(metric)
    else:
        if all_supported_metrics:
            metrics.extend(list(supported_metrics.values()))
        else:
            if STRING_TO_TASK_TYPES[dataset_properties['task_type']] in CLASSIFICATION_TASKS:
                metrics.append(supported_metrics[default_metrics['classification'][dataset_properties['output_type']]])
            if STRING_TO_TASK_TYPES[dataset_properties['task_type']] in REGRESSION_TASKS:
                metrics.append(supported_metrics[default_metrics['regression'][dataset_properties['output_type']]])

    return metrics


def calculate_score(
        target: np.ndarray,
        prediction: np.ndarray,
        task_type: int,
        metrics: Iterable[autoPyTorchMetric],
        **score_kwargs: Dict
) -> Dict[str, float]:
    score_dict = dict()
    if task_type in FORECASTING_TASKS:
        cprediction = sanitize_array(prediction)
        for metric_ in metrics:
            score_dict[metric_.name] = metric_(target, cprediction, **score_kwargs)
    elif task_type in REGRESSION_TASKS:
        cprediction = sanitize_array(prediction)
        for metric_ in metrics:
            try:
                score_dict[metric_.name] = metric_(target, cprediction)
            except ValueError as e:
                warnings.warn(f"{e} {e.args[0]}")
                if e.args[0] == "Mean Squared Logarithmic Error cannot be used when " \
                                "targets contain negative values.":
                    continue
                else:
                    raise e
    else:
        for metric_ in metrics:
            try:
                score_dict[metric_.name] = metric_(target, prediction)
            except ValueError as e:
                if e.args[0] == 'multiclass format is not supported':
                    continue
                elif e.args[0] == "Samplewise metrics are not available " \
                                  "outside of multilabel classification.":
                    continue
                elif e.args[0] == "Target is multiclass but " \
                                  "average='binary'. Please choose another average " \
                                  "setting, one of [None, 'micro', 'macro', 'weighted'].":
                    continue
                elif e.args[0] == "The labels array needs to contain at " \
                                  "least two labels for log_loss, got [0].":
                    continue
                else:
                    raise e
    return score_dict
