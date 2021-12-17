from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import warnings
import numpy as np

import pandas as pd

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter

from torch.distributions import (
    AffineTransform,
    TransformedDistribution,
)

from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.forecasting_target_scaling. \
    base_target_scaler import BaseTargetScaler
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.forecasting_target_scaling. \
    TargetNoScaler import TargetNoScaler
from autoPyTorch.pipeline.components.setup.lr_scheduler.constants import StepIntervalUnit
from autoPyTorch.pipeline.components.setup.network.forecasting_network import ForecastingNet, ForecastingDeepARNet

from autoPyTorch.pipeline.components.training.metrics.utils import calculate_score

from autoPyTorch.pipeline.components.training.trainer.base_trainer import BaseTrainerComponent, BudgetTracker


class ForecastingBaseTrainerComponent(BaseTrainerComponent, ABC):
    def prepare(
            self,
            metrics: List[Any],
            model: ForecastingNet,
            criterion: Type[torch.nn.Module],
            budget_tracker: BudgetTracker,
            optimizer: Optimizer,
            device: torch.device,
            metrics_during_training: bool,
            scheduler: _LRScheduler,
            task_type: int,
            labels: Union[np.ndarray, torch.Tensor, pd.DataFrame],
            step_interval: Union[str, StepIntervalUnit] = StepIntervalUnit.batch,
            dataset_properties: Optional[Dict] = None,
            target_scaler: BaseTargetScaler = TargetNoScaler(),
    ) -> None:
        # metrics_during_training is not appliable when computing scaled values
        metrics_during_training = False
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
        self.target_scaler = target_scaler  # typing: BaseTargetScaler

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

            batch_size = data["past_target"].size(0)
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

    def rescale_output(self,
                       outputs: Union[torch.distributions.Distribution, torch.Tensor],
                       loc: Optional[torch.Tensor],
                       scale: Optional[torch.Tensor],
                       device: torch.device = torch.device('cpu')):
        # https://github.com/awslabs/gluon-ts/blob/master/src/gluonts/torch/modules/distribution_output.py
        if loc is not None or scale is not None:
            if isinstance(outputs, torch.distributions.Distribution):
                transform = AffineTransform(loc=0.0 if loc is None else loc.to(device),
                                            scale=1.0 if scale is None else scale.to(device),
                                            )
                outputs = TransformedDistribution(outputs, [transform])
            else:
                if loc is None:
                    outputs = outputs * scale.to(device)
                elif scale is None:
                    outputs = outputs + loc.to(device)
                else:
                    outputs = outputs * scale.to(device) + loc.to(device)
        return outputs

    def train_step(self, data: Dict[str, torch.Tensor], future_targets: torch.Tensor) \
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
        past_target = data['past_target']

        # prepare
        past_target = past_target.float()
        if self.model.future_target_required:
            past_target, scaled_future_targets, loc, scale = self.target_scaler(past_target, future_targets)
        else:
            past_target, _, loc, scale = self.target_scaler(past_target)

        past_target = past_target.to(self.device)

        future_targets = self.cast_targets(future_targets)
        if isinstance(self.model, ForecastingDeepARNet) and self.model.encoder_bijective_seq_output:
            future_targets = torch.cat([past_target[:, 1:, ], future_targets], dim=1)
            past_target, criterion_kwargs = self.data_preparation(past_target, future_targets)
        else:
            past_target, criterion_kwargs = self.data_preparation(past_target, future_targets)

        # training
        self.optimizer.zero_grad()
        if self.model.future_target_required:
            outputs = self.model(past_target, scaled_future_targets.float().to(self.device))
        else:
            outputs = self.model(past_target)

        outputs = self.rescale_output(outputs, loc=loc, scale=scale, device=self.device)

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

        mase_coefficients = list()

        with torch.no_grad():
            for step, (data, future_targets) in enumerate(test_loader):
                past_target = data['past_target']

                mase_coefficients.append(data['mase_coefficient'])

                batch_size = past_target.shape[0]

                # prepare
                past_target = past_target.float()

                past_target, _, loc, scale = self.target_scaler(past_target)

                past_target = past_target.to(self.device)

                future_targets = self.cast_targets(future_targets)

                past_target, criterion_kwargs = self.data_preparation(past_target, future_targets)

                outputs = self.model(past_target)

                if isinstance(self.model, ForecastingDeepARNet):
                    # DeepAR only generate sampled points, we replace log_prob loss with MSELoss
                    outputs = self.model.pred_from_net_output(outputs)
                    outputs_rescaled = self.rescale_output(outputs, loc=loc, scale=scale, device=self.device)
                    loss = F.mse_loss(outputs_rescaled, future_targets)
                    outputs = outputs.detach().cpu()
                else:
                    if isinstance(outputs, list):
                        outputs_rescaled = [self.rescale_output(output,
                                                                loc=loc,
                                                                scale=scale,
                                                                device=self.device) for output in outputs]

                        loss = [self.criterion(output_rescaled, future_targets) for output_rescaled in outputs_rescaled]
                        loss = torch.mean(torch.Tensor(loss))
                    else:
                        outputs_rescaled = self.rescale_output(outputs, loc=loc, scale=scale, device=self.device)
                        loss = self.criterion(outputs_rescaled, future_targets)
                    outputs = self.model.pred_from_net_output(outputs).detach().cpu()

                loss_sum += loss.item() * batch_size
                N += batch_size

                if loc is None and scale is None:
                    outputs_data.append(outputs)
                else:
                    if loc is None:
                        loc = 0.
                    if scale is None:
                        scale = 1.
                    outputs_data.append(outputs * scale + loc)
                targets_data.append(future_targets.detach().cpu())

                if writer:
                    writer.add_scalar(
                        'Val/loss',
                        loss.item(),
                        epoch * len(test_loader) + step,
                    )

        # mase_coefficent has the shape [B, 1, 1]
        # to be compatible with outputs_data with shape [B, n_prediction_steps, num_output]
        mase_coefficients = np.expand_dims(torch.cat(mase_coefficients, dim=0).numpy(), axis=[1])
        self.metrics_kwargs.update({'mase_cofficient': mase_coefficients})

        self._scheduler_step(step_interval=StepIntervalUnit.valid, loss=loss_sum / N)

        self.model.train()
        return loss_sum / N, self.compute_metrics(outputs_data, targets_data)

    def compute_metrics(self, outputs_data: List[torch.Tensor], targets_data: List[torch.Tensor]
                        ) -> Dict[str, float]:
        # TODO: change once Ravin Provides the PR
        outputs_data = torch.cat(outputs_data, dim=0).numpy()
        targets_data = torch.cat(targets_data, dim=0).numpy()

        return calculate_score(targets_data, outputs_data, self.task_type, self.metrics, **self.metrics_kwargs)
