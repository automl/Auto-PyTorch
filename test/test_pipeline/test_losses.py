import pytest

import torch
from torch import nn

from autoPyTorch.constants import STRING_TO_OUTPUT_TYPES
from autoPyTorch.pipeline.components.training.losses import get_loss_instance
from autoPyTorch.utils.implementations import get_loss_weight_strategy


@pytest.mark.parametrize('output_type', ['multiclass',
                                         'binary',
                                         'continuous-multioutput',
                                         'continuous'])
def test_get_no_name(output_type):
    dataset_properties = {'task_type': 'tabular_classification', 'output_type': output_type}
    loss = get_loss_instance(dataset_properties)
    assert isinstance(loss(), nn.Module)


@pytest.mark.parametrize('output_type_name', [('multiclass', 'CrossEntropyLoss'),
                                              ('binary', 'BCEWithLogitsLoss')])
def test_get_name(output_type_name):
    output_type, name = output_type_name
    dataset_properties = {'task_type': 'tabular_classification', 'output_type': output_type}
    loss = get_loss_instance(dataset_properties, name)()
    assert isinstance(loss, nn.Module)
    assert str(loss) == f"{name}()"


def test_get_name_error():
    dataset_properties = {'task_type': 'tabular_classification', 'output_type': 'multiclass'}
    name = 'BCELoss'
    with pytest.raises(ValueError, match=r"Invalid name entered for task [a-z]+_[a-z]+, "):
        get_loss_instance(dataset_properties, name)


@pytest.mark.parametrize('weighted', [True, False])
def test_losses(weighted):
    list_properties = [{'task_type': 'tabular_classification', 'output_type': 'multiclass'},
                       {'task_type': 'tabular_classification', 'output_type': 'binary'},
                       {'task_type': 'tabular_regression', 'output_type': 'continuous'}]
    pred_cross_entropy = torch.randn(4, 4, requires_grad=True)
    list_predictions = [pred_cross_entropy, torch.empty(4).random_(2), torch.randn(4)]
    list_names = [None, 'BCEWithLogitsLoss', None]
    list_targets = [torch.empty(4, dtype=torch.long).random_(4), torch.empty(4).random_(2), torch.randn(4)]
    labels = [torch.empty(100, dtype=torch.long).random_(4), torch.empty(100, dtype=torch.long).random_(2), None]
    for dataset_properties, pred, target, name, label in zip(list_properties, list_predictions,
                                                             list_targets, list_names, labels):
        loss = get_loss_instance(dataset_properties=dataset_properties, name=name)
        weights = None
        if bool(weighted) and 'classification' in dataset_properties['task_type']:
            strategy = get_loss_weight_strategy(output_type=STRING_TO_OUTPUT_TYPES[dataset_properties['output_type']])
            weights = strategy(y=label)
            weights = torch.from_numpy(weights)
            weights = weights.type(torch.FloatTensor)
            kwargs = {'pos_weight': weights} if 'binary' in dataset_properties['output_type'] else {'weight': weights}
        loss = loss() if weights is None else loss(**kwargs)
        score = loss(pred, target)
        assert isinstance(score, torch.Tensor)
