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


from autoPyTorch.constants import REGRESSION_TASKS, FORECASTING_TASKS
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.forecasting_target_scaling. \
    base_target_scaler import BaseTargetScaler
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.forecasting_target_scaling. \
    TargetNoScaler import TargetNoScaler
from autoPyTorch.pipeline.components.setup.lr_scheduler.constants import StepIntervalUnit
from autoPyTorch.pipeline.components.setup.network.forecasting_network import ForecastingNet, ForecastingDeepARNet, \
    NBEATSNet, ForecastingSeq2SeqNet
from autoPyTorch.pipeline.components.training.losses import MASELoss


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
            window_size: int = 20,
            dataset_properties: Optional[Dict] = None,
            target_scaler: BaseTargetScaler = TargetNoScaler(),
            backcast_loss_ratio: Optional[float] = None,
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
        self.backcast_loss_ratio = backcast_loss_ratio
        self.window_size = window_size
        self.model.device = self.device

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
        import time
        time_start = time.time()
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

        print(f'time used for trainging epoch {epoch}: {time.time() - time_start}')
        print(f'Loss for epoch {epoch}: {loss_sum/N}')
        if self.metrics_during_training:
            return loss_sum / N, self.compute_metrics(outputs_data, targets_data)
        else:
            return loss_sum / N, {}

    def cast_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if self.task_type in REGRESSION_TASKS or FORECASTING_TASKS:
            targets = targets.float()
            # make sure that targets will have same shape as outputs (really important for mse loss for example)
            if targets.ndim == 1:
                targets = targets.unsqueeze(1)
        else:
            targets = targets.long()
        return targets

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
        past_target = data['past_target'].float()

        future_targets = self.cast_targets(future_targets)

        if isinstance(self.criterion, MASELoss):
            self.criterion.set_mase_coefficient(data['mase_coefficient'].float().to(self.device))

        # training
        self.optimizer.zero_grad()

        if isinstance(self.model, NBEATSNet):
            past_target = past_target[:, -self.window_size:]
            past_target, criterion_kwargs_past = self.data_preparation(past_target,
                                                                       past_target.to(self.device))
            past_target, criterion_kwargs_future = self.data_preparation(past_target, future_targets.to(self.device))
            backcast, forecast = self.model(past_target)

            loss_func_backcast = self.criterion_preparation(**criterion_kwargs_past)
            loss_func_forecast = self.criterion_preparation(**criterion_kwargs_future)

            loss_backcast = loss_func_backcast(self.criterion, backcast)
            loss_forecast = loss_func_forecast(self.criterion, forecast)

            loss = loss_forecast + loss_backcast * self.backcast_loss_ratio

            outputs = forecast
        else:
            if isinstance(self.model, ForecastingDeepARNet) and self.model.encoder_bijective_seq_output:
                if self.window_size > past_target.shape[1]:
                    all_targets = torch.cat([past_target[:, 1:, ], future_targets], dim=1)
                else:
                    if self.window_size == 1:
                        all_targets = future_targets
                    else:
                        all_targets = torch.cat([past_target[:, 1 - self.window_size:, ], future_targets], dim=1)
                past_target, criterion_kwargs = self.data_preparation(past_target, all_targets.to(self.device))
            else:
                past_target, criterion_kwargs = self.data_preparation(past_target, future_targets.to(self.device))

            outputs = self.model(past_target, future_targets)

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
        import time
        time_start = time.time()
        if not isinstance(self.model, (ForecastingDeepARNet, ForecastingSeq2SeqNet)):
            # To save time, we simply make one step prediction for DeepAR and Seq2Seq
            self.model.eval()
        if isinstance(self.model, ForecastingDeepARNet):
            self.model.only_generate_future_dist = True

        loss_sum = 0.0
        N = 0
        outputs_data = list()
        targets_data = list()

        mase_coefficients = list()

        with torch.no_grad():
            for step, (data, future_targets) in enumerate(test_loader):
                past_target = data['past_target'].float()

                mase_coefficients.append(data['mase_coefficient'])
                if isinstance(self.criterion, MASELoss):
                    self.criterion.set_mase_coefficient(data['mase_coefficient'].float().to(self.device))

                batch_size = past_target.shape[0]

                future_targets = self.cast_targets(future_targets)

                past_target, criterion_kwargs = self.data_preparation(past_target, future_targets)

                if isinstance(self.model, (ForecastingDeepARNet, ForecastingSeq2SeqNet)):
                    outputs = self.model(past_target, future_targets)
                else:
                    outputs = self.model(past_target)

                # prepare
                future_targets = future_targets.to(self.device)

                if isinstance(outputs, list):
                    loss = [self.criterion(output, future_targets) for output in outputs]
                    loss = torch.mean(torch.Tensor(loss))
                else:
                    loss = self.criterion(outputs, future_targets)
                outputs = self.model.pred_from_net_output(outputs)
                outputs = outputs.detach().cpu()

                loss_sum += loss.item() * batch_size
                N += batch_size

                outputs_data.append(outputs)
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
        print(f'time for evaluation: {time.time() - time_start}')
        return loss_sum / N, self.compute_metrics(outputs_data, targets_data)

    def compute_metrics(self, outputs_data: List[torch.Tensor], targets_data: List[torch.Tensor]
                        ) -> Dict[str, float]:
        # TODO: change once Ravin Provides the PR
        outputs_data = torch.cat(outputs_data, dim=0).numpy()
        targets_data = torch.cat(targets_data, dim=0).numpy()

        return calculate_score(targets_data, outputs_data, self.task_type, self.metrics, **self.metrics_kwargs)
