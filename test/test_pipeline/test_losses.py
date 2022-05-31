import numpy as np

import pytest

import torch
from torch import nn
from torch.nn.modules.loss import _Loss as Loss

from autoPyTorch.pipeline.components.training.losses import (
    LogProbLoss,
    MAPELoss,
    MASELoss,
    QuantileLoss,
    get_loss,
    losses
)
from autoPyTorch.utils.implementations import (
    LossWeightStrategyWeighted,
    LossWeightStrategyWeightedBinary,
    get_loss_weight_strategy
)


@pytest.mark.parametrize('output_type', ['multiclass',
                                         'binary',
                                         'continuous-multioutput',
                                         'continuous'])
def test_get_no_name(output_type):
    dataset_properties = {'task_type': 'tabular_classification', 'output_type': output_type}
    loss = get_loss(dataset_properties)
    assert isinstance(loss(), nn.Module)


@pytest.mark.parametrize('output_type_name', [('multiclass', 'CrossEntropyLoss'),
                                              ('binary', 'BCEWithLogitsLoss')])
def test_get_name(output_type_name):
    output_type, name = output_type_name
    dataset_properties = {'task_type': 'tabular_classification', 'output_type': output_type}
    loss = get_loss(dataset_properties, name)()
    assert isinstance(loss, nn.Module)
    assert str(loss) == f"{name}()"


def test_get_name_error():
    dataset_properties = {'task_type': 'tabular_classification', 'output_type': 'multiclass'}
    name = 'BCELoss'
    with pytest.raises(ValueError, match=r"Invalid name entered for task [a-z]+_[a-z]+, "):
        get_loss(dataset_properties, name)


@pytest.mark.parametrize('weighted', [True, False])
@pytest.mark.parametrize('loss_details', ['loss_cross_entropy_multiclass',
                                          'loss_cross_entropy_binary',
                                          'loss_bce',
                                          'loss_mse',
                                          'loss_mape'], indirect=True)
def test_losses(weighted, loss_details):
    dataset_properties, predictions, name, targets, labels = loss_details
    loss = get_loss(dataset_properties=dataset_properties, name=name)
    weights = None
    if bool(weighted) and 'classification' in dataset_properties['task_type']:
        strategy = get_loss_weight_strategy(loss)
        weights = strategy(y=labels)
        weights = torch.from_numpy(weights)
        weights = weights.type(torch.FloatTensor)
        kwargs = {'pos_weight': weights} if loss.__name__ == 'BCEWithLogitsLoss' else {'weight': weights}
    loss = loss() if weights is None else loss(**kwargs)
    score = loss(predictions, targets)
    assert isinstance(score, torch.Tensor)
    # Ensure it is a one element tensor
    assert len(score.size()) == 0


def test_loss_dict():
    assert 'classification' in losses.keys()
    assert 'regression' in losses.keys()
    assert 'forecasting' in losses.keys()
    for task in losses.values():
        for loss in task.values():
            assert 'module' in loss.keys()
            assert isinstance(loss['module'](), Loss)
            assert 'supported_output_types' in loss.keys()
            assert isinstance(loss['supported_output_types'], list)


@pytest.mark.parametrize('target,expected_weights', [
    (
        # Expected 4 classes where first one is majority one
        np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        # We reduce the contribution of the first class which has double elements
        np.array([0.5, 1., 1., 1.]),
    ),
    (
        # Expected 2 classes -- multilable format
        np.array([[1, 0], [1, 0], [1, 0], [0, 1]]),
        # We reduce the contribution of the first class which 3 to 1 ratio
        np.array([2 / 3, 2]),
    ),
    (
        # Expected 2 classes -- (-1, 1) format
        np.array([[1], [1], [1], [0]]),
        # We reduce the contribution of the second class, which has a 3 to 1 ratio
        np.array([2, 2 / 3]),
    ),
    (
        # Expected 2 classes -- single column
        # We have to reduce the contribution of the second class with 5 to 1 ratio
        np.array([1, 1, 1, 1, 1, 0]),
        # We reduce the contribution of the first class which has double elements
        np.array([3, 6 / 10]),
    ),
])
def test_lossweightstrategyweighted(target, expected_weights):
    weights = LossWeightStrategyWeighted()(target)
    np.testing.assert_array_equal(weights, expected_weights)
    assert nn.CrossEntropyLoss(weight=torch.Tensor(weights))(
        torch.zeros(target.shape[0], len(weights)).float(),
        torch.from_numpy(target.argmax(1)).long() if len(target.shape) > 1
        else torch.from_numpy(target).long()
    ) > 0


@pytest.mark.parametrize('target,expected_weights', [
    (
        # Expected 2 classes -- multilable format
        np.array([[1, 0], [1, 0], [1, 0], [0, 1]]),
        # We reduce the contribution of the first class which 3 to 1 ratio
        np.array([1 / 3, 3]),
    ),
    (
        # Expected 2 classes -- (-1, 1) format
        np.array([[1], [1], [1], [0]]),
        # We reduce the contribution of the second class, which has a 3 to 1 ratio
        np.array([1 / 3]),
    ),
    (
        # Expected 2 classes -- single column
        # We have to reduce the contribution of the second class with 5 to 1 ratio
        np.array([1, 1, 1, 1, 1, 0]),
        # We reduce the contribution of the first class which has double elements
        np.array([0.2]),
    ),
])
def test_lossweightstrategyweightedbinary(target, expected_weights):
    weights = LossWeightStrategyWeightedBinary()(target)
    np.testing.assert_array_equal(weights, expected_weights)
    assert nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(weights))(
        torch.from_numpy(target).float(),
        torch.from_numpy(target).float(),
    ) > 0


def test_forecasting_losses():
    target_dims = [2, 3, 1]
    targets = torch.Tensor([[0.0, 1.0, 2.0],
                            [0.0, 0.0, 0.0]]).reshape(target_dims)
    prediction_prob = torch.distributions.normal.Normal(
        torch.zeros(2, 3, 1),
        torch.ones(2, 3, 1)
    )
    prediction_value = torch.Tensor([[[0.0, 0.0, 0.0],
                                      [0.5, 0.5, 0.5]]]
                                    ).reshape(target_dims)

    log_prob_loss_raw = LogProbLoss(reduction="raw")
    loss_prob_raw = log_prob_loss_raw(prediction_prob, targets)
    assert torch.allclose(loss_prob_raw, - prediction_prob.log_prob(targets))

    log_prob_loss_mean = LogProbLoss(reduction="mean")
    loss_prob_mean = log_prob_loss_mean(prediction_prob, targets)
    assert loss_prob_mean == torch.mean(loss_prob_raw)

    log_prob_loss_sum = LogProbLoss(reduction="sum")
    loss_prob_sum = log_prob_loss_sum(prediction_prob, targets)
    assert loss_prob_sum == torch.sum(loss_prob_raw)

    mape_loss = MAPELoss(reduction="raw")
    loss_mape = mape_loss(prediction_value, targets)
    assert torch.allclose(loss_mape, torch.Tensor([[0., 1., 1.], [0., 0., 0.]]).reshape(target_dims))

    mase_loss = MASELoss(reduction="raw")
    loss_mase_1 = mase_loss(prediction_value, targets)
    assert torch.allclose(loss_mase_1, torch.Tensor([[0., 1., 2.], [0.5, 0.5, 0.5]]).reshape(target_dims))

    mase_loss.set_mase_coefficient(torch.Tensor([[2.0], [1.0]]))
    loss_mase_2 = mase_loss(prediction_value, targets)
    assert torch.allclose(loss_mase_2, torch.Tensor([[0., 2., 4.], [0.5, 0.5, 0.5]]).reshape(target_dims))

    mase_loss.set_mase_coefficient(torch.Tensor([[2.0, 2.0]]))
    with pytest.raises(ValueError, match="If self._mase_coefficient is a Tensor"):
        _ = mase_loss(prediction_value, targets)

    quantile_loss = QuantileLoss(reduction="raw")
    diff = 0.5
    quantile_prediction = [
        targets + diff
    ]
    loss_quantile_1 = quantile_loss(quantile_prediction, targets)
    assert torch.all(loss_quantile_1 == diff / 2)

    quantiles = [0.1, 0.5, 0.8]
    quantile_loss.set_quantiles([0.1, 0.5, 0.8])
    quantile_prediction = [
        targets - diff, targets - diff, targets - diff
    ]
    loss_quantile_2 = quantile_loss(quantile_prediction, targets)
    assert torch.allclose(loss_quantile_2, torch.ones_like(loss_quantile_2) * diff * np.mean(quantiles))
