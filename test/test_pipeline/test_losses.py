import numpy as np

import pytest

import torch
from torch import nn
from torch.nn.modules.loss import _Loss as Loss

from autoPyTorch.pipeline.components.training.losses import get_loss, losses
from autoPyTorch.utils.implementations import (
    LossWeightStrategyWeighted,
    LossWeightStrategyWeightedBinary,
    get_loss_weight_strategy,
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
                                          'loss_mse'], indirect=True)
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
