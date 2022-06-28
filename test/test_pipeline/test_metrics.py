import numpy as np

import pytest

import sklearn.metrics

import sktime.performance_metrics.forecasting as forecasting_metrics

from autoPyTorch.constants import (
    BINARY,
    CONTINUOUS,
    OUTPUT_TYPES_TO_STRING,
    STRING_TO_TASK_TYPES,
    TABULAR_CLASSIFICATION,
    TABULAR_REGRESSION,
    TASK_TYPES_TO_STRING,
    TIMESERIES_FORECASTING
)
from autoPyTorch.metrics import (
    accuracy,
    balanced_accuracy,
    compute_mase_coefficient,
    mean_squared_error
)
from autoPyTorch.pipeline.components.training.metrics.base import (
    ForecastingMetricMixin,
    _ForecastingMetric,
    _PredictMetric,
    _ThresholdMetric,
    autoPyTorchMetric,
    make_metric
)
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_loss, calculate_score, get_metrics


@pytest.mark.parametrize('output_type', ['multiclass',
                                         'multiclass-multioutput',
                                         'binary'])
def test_get_no_name_classification(output_type):
    dataset_properties = {'task_type': 'tabular_classification',
                          'output_type': output_type}
    metrics = get_metrics(dataset_properties)
    for metric in metrics:
        assert isinstance(metric, autoPyTorchMetric)


@pytest.mark.parametrize('output_type', ['continuous', 'continuous-multioutput'])
def test_get_no_name_regression(output_type):
    dataset_properties = {'task_type': 'tabular_regression',
                          'output_type': output_type}
    metrics = get_metrics(dataset_properties)
    for metric in metrics:
        assert isinstance(metric, autoPyTorchMetric)


@pytest.mark.parametrize('output_type', ['continuous', 'continuous-multioutput'])
def test_get_no_name_forecasting(output_type):
    dataset_properties = {'task_type': 'time_series_forecasting',
                          'output_type': output_type}
    metrics = get_metrics(dataset_properties)
    for metric in metrics:
        assert isinstance(metric, ForecastingMetricMixin)


@pytest.mark.parametrize('metric', ['accuracy', 'average_precision',
                                    'balanced_accuracy', 'f1'])
def test_get_name(metric):
    dataset_properties = {'task_type': TASK_TYPES_TO_STRING[TABULAR_CLASSIFICATION],
                          'output_type': OUTPUT_TYPES_TO_STRING[BINARY]}
    metrics = get_metrics(dataset_properties, [metric])
    for i in range(len(metrics)):
        assert isinstance(metrics[i], autoPyTorchMetric)
        assert metrics[i].name.lower() == metric.lower()


def test_get_name_error():
    dataset_properties = {'task_type': TASK_TYPES_TO_STRING[TABULAR_CLASSIFICATION],
                          'output_type': OUTPUT_TYPES_TO_STRING[BINARY]}
    names = ['root_mean_sqaured_error', 'average_precision']
    with pytest.raises(ValueError, match=r"Invalid name entered for task [a-z]+_[a-z]+, "):
        get_metrics(dataset_properties, names)


def test_classification_metrics():
    # test of all classification metrics
    dataset_properties = {'task_type': TASK_TYPES_TO_STRING[TABULAR_CLASSIFICATION],
                          'output_type': OUTPUT_TYPES_TO_STRING[BINARY]}
    y_target = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 1])
    metrics = get_metrics(dataset_properties=dataset_properties, all_supported_metrics=True)
    score_dict = calculate_score(y_pred, y_target, STRING_TO_TASK_TYPES[dataset_properties['task_type']], metrics)
    assert isinstance(score_dict, dict)
    for name, score in score_dict.items():
        assert isinstance(name, str)
        assert isinstance(score, float)


def test_regression_metrics():
    # test of all regression metrics
    dataset_properties = {'task_type': TASK_TYPES_TO_STRING[TABULAR_REGRESSION],
                          'output_type': OUTPUT_TYPES_TO_STRING[CONTINUOUS]}
    y_target = np.array([0.1, 0.6, 0.7, 0.4])
    y_pred = np.array([0.6, 0.7, 0.4, 1])
    metrics = get_metrics(dataset_properties=dataset_properties, all_supported_metrics=True)
    score_dict = calculate_score(y_pred, y_target, STRING_TO_TASK_TYPES[dataset_properties['task_type']], metrics)

    assert isinstance(score_dict, dict)
    for name, score in score_dict.items():
        assert isinstance(name, str)
        assert isinstance(score, float)


def test_forecasting_metric():
    # test of all regression metrics
    dataset_properties = {'task_type': TASK_TYPES_TO_STRING[TIMESERIES_FORECASTING],
                          'output_type': OUTPUT_TYPES_TO_STRING[CONTINUOUS]}
    n_prediction_steps = 5
    n_seq = 2
    n_targets = 2

    y_target = np.zeros([n_seq, n_prediction_steps, n_targets])
    y_pred = np.ones([n_seq, n_prediction_steps, n_targets])
    mase_coefficient = np.ones([n_seq, n_prediction_steps, n_targets]) * 2
    metrics = get_metrics(dataset_properties=dataset_properties, all_supported_metrics=True)
    forecasting_kwargs = {'sp': 4,
                          'n_prediction_steps': n_prediction_steps,
                          'mase_coefficient': mase_coefficient,
                          }
    score_dict = calculate_score(y_pred, y_target, STRING_TO_TASK_TYPES[dataset_properties['task_type']], metrics,
                                 **forecasting_kwargs)
    assert isinstance(score_dict, dict)
    for name, score in score_dict.items():
        assert isinstance(name, str)
        assert isinstance(score, float)
    forecasting_kwargs = {'sp': 4,
                          'n_prediction_steps': n_prediction_steps,
                          'mase_coefficient': np.ones([1, n_prediction_steps, n_targets]),
                          }
    with pytest.raises(ValueError, match="the shape of MASE coefficient and target_shape must be consistent"):
        score_dict = calculate_score(y_pred, y_target, STRING_TO_TASK_TYPES[dataset_properties['task_type']], metrics,
                                     **forecasting_kwargs)


def test_predictmetric_binary():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

    scorer = _PredictMetric(
        'accuracy', sklearn.metrics.accuracy_score, 1, 0, 1, {})

    score = scorer(y_true, y_pred)
    assert score == pytest.approx(1.0)

    y_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    score = scorer(y_true, y_pred)
    assert score == pytest.approx(0.5)

    y_pred = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    score = scorer(y_true, y_pred)
    assert score == pytest.approx(0.5)

    scorer = _PredictMetric(
        'bac', sklearn.metrics.balanced_accuracy_score,
        1, 0, 1, {})

    score = scorer(y_true, y_pred)
    assert score, pytest.approx(0.5)

    scorer = _PredictMetric(
        'accuracy', sklearn.metrics.accuracy_score, 1, 0, -1, {})

    y_pred = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    score = scorer(y_true, y_pred)
    assert score == pytest.approx(-1.0)


def test_threshold_scorer_binary():
    y_true = [0, 0, 1, 1]
    y_pred = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

    scorer = _ThresholdMetric(
        'roc_auc', sklearn.metrics.roc_auc_score, 1, 0, 1, {})

    score = scorer(y_true, y_pred)
    assert score == pytest.approx(1.0)

    y_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    score = scorer(y_true, y_pred)
    assert score == pytest.approx(0.5)

    y_pred = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    score = scorer(y_true, y_pred)
    assert score == pytest.approx(0.5)

    scorer = _ThresholdMetric(
        'roc_auc', sklearn.metrics.roc_auc_score, 1, 0, -1, {})

    y_pred = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    score = scorer(y_true, y_pred)
    assert score == pytest.approx(-1.0)


def test_forecastingcomputation():
    scorer_mean = _ForecastingMetric(
        'mean_mape', forecasting_metrics.mean_absolute_percentage_error, 0.0, np.finfo(np.float64).max, 1,
        kwargs=dict(aggregation='mean'),
    )
    scorer_median = _ForecastingMetric(
        'median_mape', forecasting_metrics.mean_absolute_percentage_error, 0.0, np.finfo(np.float64).max, 1,
        kwargs=dict(aggregation='median'),
    )

    n_seq = 3
    n_prediction_steps = 5
    n_targets = 2

    y_true = np.expand_dims(
        [np.arange(n_prediction_steps) + i * 10 for i in range(n_seq)], -1
    ).repeat(n_targets, axis=-1)
    y_pred = y_true + 1
    score_mean = scorer_mean(y_true=y_true, y_pred=y_pred, sp=1, n_prediction_steps=n_prediction_steps)
    score_median = scorer_median(y_true=y_true, y_pred=y_pred, sp=1, n_prediction_steps=n_prediction_steps)

    score_all = []
    for true_seq, pred_seq in zip(y_true, y_pred):
        score_all.append(forecasting_metrics.mean_absolute_percentage_error(y_true=true_seq, y_pred=pred_seq))
    assert score_mean == np.mean(score_all)
    assert score_median == np.median(score_all)

    # Additional parameters
    horizon_weight = [0.1, 0.2, 0.3, 0.4, 0.5]
    score_mean = scorer_mean(y_true=y_true, y_pred=y_pred, sp=1,
                             n_prediction_steps=n_prediction_steps, horizon_weight=horizon_weight)
    score_all = []
    for true_seq, pred_seq in zip(y_true, y_pred):
        score_all.append(forecasting_metrics.mean_absolute_percentage_error(y_true=true_seq, y_pred=pred_seq,
                                                                            horizon_weight=horizon_weight))
    assert score_mean == np.mean(score_all)


def test_sign_flip():
    y_true = np.arange(0, 1.01, 0.1)
    y_pred = y_true.copy()

    scorer = make_metric(
        'r2', sklearn.metrics.r2_score, greater_is_better=True)

    score = scorer(y_true, y_pred + 1.0)
    assert score == pytest.approx(-9.0)

    score = scorer(y_true, y_pred + 0.5)
    assert score == pytest.approx(-1.5)

    score = scorer(y_true, y_pred)
    assert score == pytest.approx(1.0)

    scorer = make_metric(
        'r2', sklearn.metrics.r2_score, greater_is_better=False)

    score = scorer(y_true, y_pred + 1.0)
    assert score == pytest.approx(9.0)

    score = scorer(y_true, y_pred + 0.5)
    assert score == pytest.approx(1.5)

    score = scorer(y_true, y_pred)
    assert score == pytest.approx(-1.0)


def test_classification_only_metric():
    y_true = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    y_pred = \
        np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    scorer = accuracy

    score = calculate_score(y_true, y_pred, TABULAR_CLASSIFICATION, [scorer])

    previous_score = scorer._optimum
    assert score['accuracy'] == pytest.approx(previous_score)


def test_calculate_loss():
    # In a 0-1 ranged scorer, make sure that the loss
    # has a expected positive value
    y_pred = np.array([0, 1, 0, 1, 1, 1, 0, 0, 0, 0])
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 0, 0, 0])
    score = sklearn.metrics.accuracy_score(y_true, y_pred)
    assert pytest.approx(score) == calculate_score(
        target=y_true,
        prediction=y_pred,
        task_type=TABULAR_CLASSIFICATION,
        metrics=[accuracy],
    )['accuracy']
    loss = 1.0 - score
    assert pytest.approx(loss) == calculate_loss(
        target=y_true,
        prediction=y_pred,
        task_type=TABULAR_CLASSIFICATION,
        metrics=[accuracy],
    )['accuracy']

    # Test the dictionary case
    score_dict = calculate_score(
        target=y_true,
        prediction=y_pred,
        task_type=TABULAR_CLASSIFICATION,
        metrics=[accuracy, balanced_accuracy],
    )
    expected_score_dict = {
        'accuracy': 0.9,
        'balanced_accuracy': 0.9285714285714286,
    }
    loss_dict = calculate_loss(
        target=y_true,
        prediction=y_pred,
        task_type=TABULAR_CLASSIFICATION,
        metrics=[accuracy, balanced_accuracy],
    )
    for expected_metric, expected_score in expected_score_dict.items():
        assert pytest.approx(expected_score) == score_dict[expected_metric]
        assert pytest.approx(1 - expected_score) == loss_dict[expected_metric]

    # Lastly make sure that metrics whose optimum is zero
    # are also properly working
    y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    y_pred = np.array([0.11, 0.22, 0.33, 0.44, 0.55, 0.66])
    score = sklearn.metrics.mean_squared_error(y_true, y_pred)
    assert pytest.approx(score) == calculate_score(
        target=y_true,
        prediction=y_pred,
        task_type=TABULAR_REGRESSION,
        metrics=[mean_squared_error],
    )['mean_squared_error']
    loss = score
    assert pytest.approx(loss) == calculate_loss(
        target=y_true,
        prediction=y_pred,
        task_type=TABULAR_REGRESSION,
        metrics=[mean_squared_error],
    )['mean_squared_error']


def test_compute_mase_coefficient():
    past_target = np.arange(12)
    mase_value_1 = compute_mase_coefficient(past_target, 15)
    assert mase_value_1 == 1 / np.mean(past_target)
    mase_value_2 = compute_mase_coefficient(past_target, 5)
    assert mase_value_2 == 0.2

    past_target = np.ones(12) * 2
    assert compute_mase_coefficient(past_target, 15) == 0.5
    assert compute_mase_coefficient(past_target, 5) == 0.5

    past_target = np.zeros(12)
    assert compute_mase_coefficient(past_target, 15) == 1.
    assert compute_mase_coefficient(past_target, 5) == 1.
