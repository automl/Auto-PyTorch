"""
Loss functions available in autoPyTorch

Classification:
            CrossEntropyLoss: supports multiclass, binary output types
            BCEWithLogitsLoss: supports binary output types
        Default: CrossEntropyLoss
Regression:
            MSELoss: supports continuous output types
            L1Loss: supports continuous output types
        Default: MSELoss
"""
from typing import Any, Dict, Optional, Type

from torch.nn.modules.loss import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    L1Loss,
    MSELoss
)
from torch.nn.modules.loss import _Loss as Loss

from autoPyTorch.constants import BINARY, CLASSIFICATION_TASKS, CONTINUOUS, MULTICLASS, REGRESSION_TASKS, \
    STRING_TO_OUTPUT_TYPES, STRING_TO_TASK_TYPES, TASK_TYPES_TO_STRING


losses = dict(classification=dict(
    CrossEntropyLoss=dict(
        module=CrossEntropyLoss, supported_output_types=[MULTICLASS, BINARY]),
    BCEWithLogitsLoss=dict(
        module=BCEWithLogitsLoss, supported_output_types=[BINARY])),
    regression=dict(
        MSELoss=dict(
            module=MSELoss, supported_output_types=[CONTINUOUS]),
        L1Loss=dict(
            module=L1Loss, supported_output_types=[CONTINUOUS])))

default_losses: Dict[str, Type[Loss]] = dict(classification=CrossEntropyLoss, regression=MSELoss)


def get_default(task: int) -> Type[Loss]:
    """
    Utility function to get default loss for the task
    Args:
        task (int):

    Returns:
        Type[torch.nn.modules.loss._Loss]
    """
    if task in CLASSIFICATION_TASKS:
        return default_losses['classification']
    elif task in REGRESSION_TASKS:
        return default_losses['regression']
    else:
        raise ValueError("Invalid task type {}".format(TASK_TYPES_TO_STRING[task]))


def get_supported_losses(task: int, output_type: int) -> Dict[str, Type[Loss]]:
    """
    Utility function to get supported losses for a given task and output type
    Args:
        task (int): integer identifier for the task
        output_type: integer identifier for the output type of the task

    Returns:
        Returns a dictionary containing the losses supported for the given
        inputs. Key-Name, Value-Module
    """
    supported_losses = dict()
    if task in CLASSIFICATION_TASKS:
        for key, value in losses['classification'].items():
            if output_type in value['supported_output_types']:
                supported_losses[key] = value['module']
    elif task in REGRESSION_TASKS:
        for key, value in losses['regression'].items():
            if output_type in value['supported_output_types']:
                supported_losses[key] = value['module']
    return supported_losses


def get_loss(dataset_properties: Dict[str, Any], name: Optional[str] = None) -> Type[Loss]:
    """
    Utility function to get losses for the given dataset properties.
    If name is mentioned, checks if the loss is compatible with
    the dataset properties and returns the specific loss
    Args:
        dataset_properties (Dict[str, Any]): Dictionary containing
        properties of the dataset. Must contain task_type and
        output_type as strings.
        name (Optional[str]): name of the specific loss

    Returns:
        Type[torch.nn.modules.loss._Loss]
    """
    assert 'task_type' in dataset_properties, \
        "Expected dataset_properties to have task_type got {}".format(dataset_properties.keys())
    assert 'output_type' in dataset_properties, \
        "Expected dataset_properties to have output_type got {}".format(dataset_properties.keys())

    task = STRING_TO_TASK_TYPES[dataset_properties['task_type']]
    output_type = STRING_TO_OUTPUT_TYPES[dataset_properties['output_type']]
    supported_losses = get_supported_losses(task, output_type)

    if name is not None:
        if name not in supported_losses.keys():
            raise ValueError("Invalid name entered for task {}, and output type {} currently supported losses"
                             " for task include {}".format(dataset_properties['task_type'],
                                                           dataset_properties['output_type'],
                                                           list(supported_losses.keys())))
        else:
            loss = supported_losses[name]
    else:
        loss = get_default(task)

    return loss
