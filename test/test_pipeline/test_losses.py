import pytest

import torch
from torch import nn

from autoPyTorch.pipeline.components.training.losses import get_loss
from autoPyTorch.utils.implementations import get_loss_weight_strategy


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
