from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import warnings
import numpy as np

import pandas as pd

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter

from torch.distributions import (
    AffineTransform,
    TransformedDistribution,
)

from autoPyTorch.pipeline.components.setup.lr_scheduler.constants import StepIntervalUnit
from autoPyTorch.pipeline.components.training.metrics.metrics import MASE_LOSSES
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_score

from autoPyTorch.pipeline.components.training.trainer.base_trainer import BaseTrainerComponent, BudgetTracker


class ForecastingBaseTrainerComponent(BaseTrainerComponent, ABC):
    def prepare(
            self,
            metrics: List[Any],
            model: torch.nn.Module,
            criterion: Type[torch.nn.Module],
            budget_tracker: BudgetTracker,
            optimizer: Optimizer,
            device: torch.device,
            metrics_during_training: bool,
            scheduler: _LRScheduler,
            task_type: int,
            labels: Union[np.ndarray, torch.Tensor, pd.DataFrame],
            step_interval: Union[str, StepIntervalUnit] = StepIntervalUnit.batch,
            dataset_properties: Optional[Dict]=None
    ) -> None:
        for metric in metrics:
            if metric in MASE_LOSSES:
                warnings.warn("MASE Losses are not supported for trainer! We remove them here")
                metrics.remove(metric)

        super().prepare(metrics=metrics,
                        model=model,
                        criterion=criterion,
                        budget_tracker=budget_tracker,
                        optimizer=optimizer,
                        device=device,
                        metrics_during_training=metrics_during_training,
                        scheduler=scheduler,
                        task_type=task_type,
                        labels=labels,
                        step_interval=step_interval
                        )
        metric_kwargs = {"sp": dataset_properties.get("sp", 1),
                         "n_prediction_steps": dataset_properties.get("n_prediction_steps", 1)}
        self.metrics_kwargs = metric_kwargs

    def train_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int,
                    writer: Optional[SummaryWriter],
                    ) -> Tuple[float, Dict[str, float]]:
        """
        Train the model for a single epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): generator of features/label
            epoch (int): The current epoch used solely for tracking purposes

        Returns:
            float: training loss
            Dict[str, float]: scores for each desired metric
        """

        loss_sum = 0.0
        N = 0
        self.model.train()
        outputs_data = list()
        targets_data = list()

        for step, (data, targets) in enumerate(train_loader):
            if self.budget_tracker.is_max_time_reached():
                break

            loss, outputs = self.train_step(data, targets)

            if self.metrics_during_training:
                # save for metric evaluation
                outputs_data.append(outputs.detach().cpu())
                targets_data.append(targets.detach().cpu())

            batch_size = data["value"].size(0)
            loss_sum += loss * batch_size
            N += batch_size

            if writer:
                writer.add_scalar(
                    'Train/loss',
                    loss,
                    epoch * len(train_loader) + step,
                )

        self._scheduler_step(step_interval=StepIntervalUnit.epoch, loss=loss_sum / N)

        if self.metrics_during_training:
            return loss_sum / N, self.compute_metrics(outputs_data, targets_data)
        else:
            return loss_sum / N, {}

    def rescale_output_distribution(self,
                                      outputs: torch.distributions.Distribution,
                                      loc: Optional[torch.Tensor],
                                      scale: Optional[torch.Tensor]):
        # https://github.com/awslabs/gluon-ts/blob/master/src/gluonts/torch/modules/distribution_output.py
        if loc is not None or scale is not None:
            transfomr = AffineTransform(loc=0.0 if loc is None else loc,
                                        scale=1.0 if scale is None else scale,
                                        )
            outputs = TransformedDistribution(outputs, [transfomr])
        return outputs

    def train_step(self, data: Dict[str, torch.Tensor], targets:  Dict[str, Union[torch.Tensor, np.ndarray]])\
            -> Tuple[float, torch.Tensor]:
        """
        Allows to train 1 step of gradient descent, given a batch of train/labels

        Args:
            data (torch.Tensor): input features to the network
            targets (torch.Tensor): ground truth to calculate loss

        Returns:
            torch.Tensor: The predictions of the network
            float: the loss incurred in the prediction
        """
        X = data['value']
        loc = data['loc']
        scale = data['scale']

        # prepare
        X = X.float().to(self.device)

        targets = self.cast_targets(targets)

        X, criterion_kwargs = self.data_preparation(X, targets)

        # training
        self.optimizer.zero_grad()
        outputs = self.model(X)

        outputs = self.rescale_output_distribution(outputs, loc=loc, scale=scale)

        loss_func = self.criterion_preparation(**criterion_kwargs)
        loss = loss_func(self.criterion, outputs)
        loss.backward()
        self.optimizer.step()
        self._scheduler_step(step_interval=StepIntervalUnit.batch, loss=loss.item())

        return loss.item(), outputs

    def evaluate(self, test_loader: torch.utils.data.DataLoader, epoch: int,
                 writer: Optional[SummaryWriter],
                 ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the model in both metrics and criterion

        Args:
            test_loader (torch.utils.data.DataLoader): generator of features/label
            epoch (int): the current epoch for tracking purposes

        Returns:
            float: test loss
            Dict[str, float]: scores for each desired metric
        """
        self.model.eval()

        loss_sum = 0.0
        N = 0
        outputs_data = list()
        targets_data = list()

        with torch.no_grad():
            for step, (data, targets) in enumerate(test_loader):
                X = data['value']
                loc = data['loc']
                scale = data['scale']

                batch_size = X.shape[0]

                # prepare
                X = X.float().to(self.device)

                targets = self.cast_targets(targets)

                X, criterion_kwargs = self.data_preparation(X, targets)

                outputs = self.model(X)

                outputs = self.rescale_output_distribution(outputs, loc=loc, scale=scale)

                loss = self.criterion(outputs, targets)

                loss_sum += loss.item() * batch_size
                N += batch_size
                if loc is None and scale is None:
                    outputs_data.append(outputs.mean.detach().cpu())
                else:
                    if loc is None:
                        loc = 0.
                    if scale is None:
                        scale = 1.
                    outputs_data.append(outputs.base_dist.mean * scale + loc)
                targets_data.append(targets.detach().cpu())

                if writer:
                    writer.add_scalar(
                        'Val/loss',
                        loss.item(),
                        epoch * len(test_loader) + step,
                    )

        self._scheduler_step(step_interval=StepIntervalUnit.valid, loss=loss_sum / N)

        self.model.train()
        return loss_sum / N, self.compute_metrics(outputs_data, targets_data)

    def compute_metrics(self, outputs_data: List[torch.Tensor], targets_data: List[torch.Tensor]
                        ) -> Dict[str, float]:
        # TODO: change once Ravin Provides the PR
        outputs_data = torch.cat(outputs_data, dim=0).numpy()
        targets_data = torch.cat(targets_data, dim=0).numpy()
        return calculate_score(targets_data, outputs_data, self.task_type, self.metrics, **self.metrics_kwargs)
