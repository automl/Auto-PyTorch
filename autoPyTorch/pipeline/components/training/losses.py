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
Forecasting:
            LogProbLoss: supports distribution output types
            QuantileLoss: supports quantile output type
            MAPELoss: supports continuous output types
            MASELoss: supports continuous output types
            L1Loss: supports continuous output types
"""
from typing import Any, Dict, List, Optional, Type, Union

import torch
from torch.nn.modules.loss import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    L1Loss,
    MSELoss
)
from torch.nn.modules.loss import _Loss as Loss

from autoPyTorch.constants import BINARY, CLASSIFICATION_TASKS, CONTINUOUS, FORECASTING_TASKS, MULTICLASS, \
    REGRESSION_TASKS, STRING_TO_OUTPUT_TYPES, STRING_TO_TASK_TYPES, TASK_TYPES_TO_STRING


class AbstractForecastingLoss(Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super(AbstractForecastingLoss, self).__init__(reduction=reduction)

    def aggregate_loss(self, loss_values: torch.Tensor) -> torch.Tensor:
        if self.reduction == 'mean':
            return loss_values.mean()
        elif self.reduction == 'sum':
            return loss_values.sum()
        else:
            return loss_values


class LogProbLoss(AbstractForecastingLoss):
    def forward(self, input_dist: torch.distributions.Distribution, target_tensor: torch.Tensor) -> torch.Tensor:
        scores = input_dist.log_prob(target_tensor)
        return self.aggregate_loss(-scores)


class MAPELoss(AbstractForecastingLoss):
    def forward(self, predictions: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        # https://github.com/awslabs/gluon-ts/blob/master/src/gluonts/model/n_beats/_network.py
        denominator = torch.abs(target_tensor)
        diff = torch.abs(predictions - target_tensor)

        flag = (denominator == 0).float()

        mape = (diff * (1 - flag)) / (denominator + flag)

        return self.aggregate_loss(mape)


class MASELoss(AbstractForecastingLoss):
    def __init__(self, reduction: str = 'mean') -> None:
        super(MASELoss, self).__init__(reduction=reduction)
        self._mase_coefficient: Union[float, torch.Tensor] = 1.0

    def set_mase_coefficient(self, mase_coefficient: torch.Tensor) -> 'MASELoss':
        """
        set mase coefficient for computing MASE losses
        Args:
            mase_coefficient (torch.Tensor): mase coefficient, its dimensions corresponds to [B, L, N] and can be
                broadcasted

        Returns:

        """
        if len(mase_coefficient.shape) == 2:
            mase_coefficient = mase_coefficient.unsqueeze(1)

        self._mase_coefficient = mase_coefficient
        return self

    def forward(self,
                predictions: torch.Tensor,
                target_tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(self._mase_coefficient, torch.Tensor):
            mase_shape = self._mase_coefficient.shape
            pred_shape = predictions.shape
            if len(mase_shape) == len(pred_shape):
                if mase_shape[0] != pred_shape[0] or mase_shape[-1] != pred_shape[-1]:
                    raise ValueError(f"If self._mase_coefficient is a Tensor, it must have the same batch size and "
                                     f"num_targets as the predictions, However, their shapes are {mase_shape}"
                                     f"(self._mase_coefficient) and {pred_shape}(pred_shape)")
        loss_values = torch.abs(predictions - target_tensor) * self._mase_coefficient
        return self.aggregate_loss(loss_values)


class QuantileLoss(AbstractForecastingLoss):
    def __init__(self, reduction: str = 'mean', quantiles: List[float] = [0.5]) -> None:
        super(QuantileLoss, self).__init__(reduction=reduction)
        self.quantiles = quantiles

    def set_quantiles(self, quantiles: List[float]) -> None:
        self.quantiles = quantiles

    def forward(self,
                predictions: List[torch.Tensor],
                target_tensor: torch.Tensor) -> torch.Tensor:
        assert len(self.quantiles) == len(predictions)
        losses_all = []
        for q, y_pred in zip(self.quantiles, predictions):
            diff = target_tensor - y_pred

            loss_q = torch.max(q * diff, (q - 1) * diff)
            losses_all.append(loss_q.unsqueeze(-1))

        losses_all = torch.mean(torch.concat(losses_all, dim=-1), dim=-1)

        return self.aggregate_loss(losses_all)


losses = dict(
    classification=dict(
        CrossEntropyLoss=dict(
            module=CrossEntropyLoss, supported_output_types=[MULTICLASS, BINARY]),
        BCEWithLogitsLoss=dict(
            module=BCEWithLogitsLoss, supported_output_types=[BINARY])),
    regression=dict(
        MSELoss=dict(
            module=MSELoss, supported_output_types=[CONTINUOUS]),
        L1Loss=dict(
            module=L1Loss, supported_output_types=[CONTINUOUS])),
    forecasting=dict(
        LogProbLoss=dict(
            module=LogProbLoss, supported_output_types=[CONTINUOUS]),
        MSELoss=dict(
            module=MSELoss, supported_output_types=[CONTINUOUS]),
        L1Loss=dict(
            module=L1Loss, supported_output_types=[CONTINUOUS]),
        MAPELoss=dict(
            module=MAPELoss, supported_output_types=[CONTINUOUS]),
        MASELoss=dict(
            module=MASELoss, supported_output_types=[CONTINUOUS]),
    )
)

default_losses: Dict[str, Type[Loss]] = dict(classification=CrossEntropyLoss,
                                             regression=MSELoss,
                                             forecasting=MASELoss)

LOSS_TYPES = ['regression', 'distribution']


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
    elif task in FORECASTING_TASKS:
        return default_losses['forecasting']
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
    elif task in FORECASTING_TASKS:
        for key, value in losses['forecasting'].items():
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
