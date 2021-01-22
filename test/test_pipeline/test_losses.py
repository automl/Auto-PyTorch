import pytest

import torch
from torch import nn

from autoPyTorch.pipeline.components.training.losses import get_loss_instance


@pytest.mark.parametrize('output_type', ['multiclass',
                                         'binary',
                                         'continuous-multioutput',
                                         'continuous'])
def test_get_no_name(output_type):
    dataset_properties = {'task_type': 'tabular_classification', 'output_type': output_type}
    loss = get_loss_instance(dataset_properties)
    assert isinstance(loss, nn.Module)


@pytest.mark.parametrize('output_type_name', [('multiclass', 'CrossEntropyLoss'),
                                              ('binary', 'BCEWithLogitsLoss')])
def test_get_name(output_type_name):
    output_type, name = output_type_name
    dataset_properties = {'task_type': 'tabular_classification', 'output_type': output_type}
    loss = get_loss_instance(dataset_properties, name)
    assert isinstance(loss, nn.Module)
    assert str(loss) == f"{name}()"


def test_get_name_error():
    dataset_properties = {'task_type': 'tabular_classification', 'output_type': 'multiclass'}
    name = 'BCELoss'
    with pytest.raises(ValueError, match=r"Invalid name entered for task [a-z]+_[a-z]+, "):
        get_loss_instance(dataset_properties, name)


def test_losses():
    list_properties = [{'task_type': 'tabular_classification', 'output_type': 'multiclass'},
                       {'task_type': 'tabular_classification', 'output_type': 'binary'},
                       {'task_type': 'tabular_regression', 'output_type': 'continuous'}]
    pred_cross_entropy = torch.randn(4, 4, requires_grad=True)
    list_predictions = [pred_cross_entropy, torch.empty(4).random_(2), torch.randn(4)]
    list_names = [None, 'BCEWithLogitsLoss', None]
    list_targets = [torch.empty(4, dtype=torch.long).random_(4), torch.empty(4).random_(2), torch.randn(4)]
    for dataset_properties, pred, target, name in zip(list_properties, list_predictions, list_targets, list_names):
        loss = get_loss_instance(dataset_properties=dataset_properties, name=name)
        score = loss(pred, target)
        assert isinstance(score, torch.Tensor)
