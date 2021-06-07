import time
from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Type, Union


import numpy as np

import pandas as pd

from sklearn.utils import check_random_state

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter


from autoPyTorch.constants import REGRESSION_TASKS
from autoPyTorch.pipeline.components.training.base_training import autoPyTorchTrainingComponent
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_score
from autoPyTorch.utils.implementations import get_loss_weight_strategy


class StepIntervalUnit(Enum):
    """
    By which interval we perform the step for learning rate schedulers.

    Attributes:
        batch (str): We update every batch evaluation
        epoch (str): We update every epoch
    """
    batch = 'batch'
    epoch = 'epoch'


class _NewLossParameters(NamedTuple):
    """
    Parameters for a new loss function.

    Attributes:
        y_a (torch.Tensor): The main output
        y_b (torch.Tensor): The output to mix up
                            (for standard loss function, it is None)
        lam (float): a mixup coefficient of y_a and y_b
                     lam accounts for how much percentage input A
                     contributes.
    """
    y_a: torch.Tensor
    lam: float = 1.0
    y_b: Optional[torch.Tensor] = None


class BudgetTracker(object):
    def __init__(self,
                 budget_type: str,
                 max_epochs: Optional[int] = None,
                 max_runtime: Optional[int] = None,
                 ):
        """
        An object for tracking when to stop the network training.
        It handles epoch based criteria as well as training based criteria.

        It also allows to define a 'epoch_or_time' budget type, which means,
        the first of them both which is exhausted, is honored
        """
        self.start_time = time.time()
        self.budget_type = budget_type
        self.max_epochs = max_epochs
        self.max_runtime = max_runtime

    def is_max_epoch_reached(self, epoch: int) -> bool:

        # Make None a method to run without this constrain
        if self.max_epochs is None:
            return False
        if self.budget_type in ['epochs', 'epoch_or_time'] and epoch > self.max_epochs:
            return True
        return False

    def is_max_time_reached(self) -> bool:
        # Make None a method to run without this constrain
        if self.max_runtime is None:
            return False
        elapsed_time = time.time() - self.start_time
        if self.budget_type in ['runtime', 'epoch_or_time'] and elapsed_time > self.max_runtime:
            return True
        return False


class RunSummary(object):
    def __init__(
        self,
        total_parameter_count: float,
        trainable_parameter_count: float,
    ):
        """
        A useful object to track performance per epoch.

        It allows to track train, validation and test information not only for
        debug, but for research purposes (Like understanding overfit).

        It does so by tracking a metric/loss at the end of each epoch.
        """
        self.performance_tracker = {
            'start_time': {},
            'end_time': {},
        }  # type: Dict[str, Dict]

        self.total_parameter_count = total_parameter_count
        self.trainable_parameter_count = trainable_parameter_count

        # Allow to track the training, val, test performance
        for loss_type in ['train_loss', 'val_loss', 'test_loss']:
            self.performance_tracker[loss_type] = {}

        # Allow to track the metrics performance
        for metric_type in ['train_metrics', 'val_metrics', 'test_metrics']:
            self.performance_tracker[metric_type] = {}

    def add_performance(self,
                        epoch: int,
                        start_time: float,
                        end_time: float,
                        train_loss: float,
                        train_metrics: Dict[str, float],
                        val_metrics: Dict[str, float] = {},
                        test_metrics: Dict[str, float] = {},
                        val_loss: Optional[float] = None,
                        test_loss: Optional[float] = None,
                        ) -> None:
        """
        Tracks performance information about the run, useful for
        plotting individual runs
        """
        self.performance_tracker['train_loss'][epoch] = train_loss
        self.performance_tracker['val_loss'][epoch] = val_loss
        self.performance_tracker['test_loss'][epoch] = test_loss
        self.performance_tracker['start_time'][epoch] = start_time
        self.performance_tracker['end_time'][epoch] = end_time
        self.performance_tracker['train_metrics'][epoch] = train_metrics
        self.performance_tracker['val_metrics'][epoch] = val_metrics
        self.performance_tracker['test_metrics'][epoch] = test_metrics

    def get_best_epoch(self, loss_type: str = 'val_loss') -> int:
        return np.argmin(
            [self.performance_tracker[loss_type][e]
             for e in range(1, len(self.performance_tracker[loss_type]) + 1)]
        ) + 1  # Epochs start at 1

    def get_last_epoch(self) -> int:
        if 'train_loss' not in self.performance_tracker:
            return 0
        else:
            return max(self.performance_tracker['train_loss'].keys())

    def repr_last_epoch(self) -> str:
        """
        For debug purposes, returns a nice representation of last epoch
        performance

        Returns:
            str: A nice representation of the last epoch
        """
        last_epoch = len(self.performance_tracker['train_loss'])
        string = "\n"
        string += '=' * 40
        string += f"\n\t\tEpoch {last_epoch}\n"
        string += '=' * 40
        string += "\n"
        for key, value in sorted(self.performance_tracker.items()):
            if isinstance(value[last_epoch], dict):
                # Several metrics can be passed
                string += "\t{}:\n".format(
                    key,
                )
                for sub_key, sub_value in sorted(value[last_epoch].items()):
                    string += "\t\t{}: {}\n".format(
                        sub_key,
                        sub_value,
                    )
            else:
                string += "\t{}: {}\n".format(
                    key,
                    value[last_epoch],
                )
        string += '=' * 40
        return string


class BaseTrainerComponent(autoPyTorchTrainingComponent):

    def __init__(self, random_state: Optional[np.random.RandomState] = None) -> None:
        if random_state is None:
            # A trainer components need a random state for
            # sampling -- for example in MixUp training
            self.random_state = check_random_state(1)
        else:
            self.random_state = random_state
        super().__init__(random_state=self.random_state)

        self.weighted_loss: bool = False

    def set_training_params(
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
        step_unit: Union[str, StepIntervalUnit] = StepIntervalUnit.batch
    ) -> None:

        step_unit_types = (str, StepIntervalUnit)
        if not isinstance(step_unit, step_unit_types):
            raise ValueError('step_unit must be either {}, but got {}.'.format(
                step_unit_types,
                type(step_unit)
            ))

        if isinstance(step_unit, str):
            if step_unit not in StepIntervalUnit.__members__.keys():
                raise ValueError(
                    'step_unit must be {}, but got {}'.format(
                        list(StepIntervalUnit.__members__.keys()),
                        step_unit
                    ))

            step_unit = StepIntervalUnit.__members__[step_unit]

        # Save the device to be used
        self.device = device

        # Setup the metrics
        self.metrics = metrics

        # Weights for the loss function
        weight_dict = {}
        if self.weighted_loss:
            weight_dict = self._get_class_weights(criterion, labels)

        # Setup the loss function
        self.criterion = criterion(**weight_dict)
        # setup the model
        self.model = model.to(device)

        # setup the optimizers
        self.optimizer = optimizer

        # The budget tracker
        self.budget_tracker = budget_tracker

        # For best performance, we allow option to prevent comparing metrics every time
        self.metrics_during_training = metrics_during_training

        # Scheduler
        self.scheduler = scheduler
        self.step_unit = step_unit

        # task type (used for calculating metrics)
        self.task_type = task_type

    def on_epoch_start(self, X: Dict[str, Any], epoch: int) -> None:
        """
        Optional place holder for AutoPytorch Extensions.

        An user can define what happens on every epoch start or every epoch end.
        """
        pass

    def on_epoch_end(self, X: Dict[str, Any], epoch: int) -> bool:
        """
        Optional place holder for AutoPytorch Extensions.
        An user can define what happens on every epoch start or every epoch end.
        If returns True, the training is stopped

        """
        return False

    def _scheduler_step(
        self,
        step_interval: StepIntervalUnit,
        loss: Optional[torch.Tensor] = None
    ) -> None:
        if self.step_unit != step_interval:
            return

        if self.scheduler:
            if 'ReduceLROnPlateau' in self.scheduler.__class__.__name__:
                self.scheduler.step(loss)
            else:
                self.scheduler.step()

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

            # save for metric evaluation
            outputs_data.append(outputs.detach().cpu())
            targets_data.append(targets.detach().cpu())

            batch_size = data.size(0)
            loss_sum += loss * batch_size
            N += batch_size

            if writer:
                writer.add_scalar(
                    'Train/loss',
                    loss,
                    epoch * len(train_loader) + step,
                )

        self._scheduler_step(step_interval=StepIntervalUnit.epoch, loss=loss)

        if self.metrics_during_training:
            return loss_sum / N, self._compute_metrics(outputs_data, targets_data)
        else:
            return loss_sum / N, {}

    def _cast_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if self.task_type in REGRESSION_TASKS:
            targets = targets.float().to(self.device)
            # make sure that targets will have same shape as outputs (really important for mse loss for example)
            if targets.ndim == 1:
                targets = targets.unsqueeze(1)
        else:
            targets = targets.long().to(self.device)
        return targets

    def train_step(self, data: torch.Tensor, targets: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Allows to train 1 step of gradient descent, given a batch of train/labels

        Args:
            data (torch.Tensor): input features to the network
            targets (torch.Tensor): ground truth to calculate loss

        Returns:
            torch.Tensor: The predictions of the network
            float: the loss incurred in the prediction
        """
        # prepare
        data = data.float().to(self.device)
        targets = self._cast_targets(targets)

        data, new_loss_params = self._data_preprocessing(data, targets)

        # training
        self.optimizer.zero_grad()
        outputs = self.model(data)
        loss_fn = self._get_new_loss_fn(new_loss_params)
        loss = loss_fn(self.criterion, outputs)
        loss.backward()
        self.optimizer.step()
        self._scheduler_step(step_interval=StepIntervalUnit.batch, loss=loss)

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
                batch_size = data.shape[0]

                data = data.float().to(self.device)
                targets = self._cast_targets(targets)

                outputs = self.model(data)

                loss = self.criterion(outputs, targets)
                loss_sum += loss.item() * batch_size
                N += batch_size

                outputs_data.append(outputs.detach().cpu())
                targets_data.append(targets.detach().cpu())

                if writer:
                    writer.add_scalar(
                        'Val/loss',
                        loss.item(),
                        epoch * len(test_loader) + step,
                    )

        self.model.train()
        return loss_sum / N, self._compute_metrics(outputs_data, targets_data)

    def _compute_metrics(self, outputs_data: List[torch.Tensor], targets_data: List[torch.Tensor]
                         ) -> Dict[str, float]:
        # TODO: change once Ravin Provides the PR
        outputs_data = torch.cat(outputs_data, dim=0).numpy()
        targets_data = torch.cat(targets_data, dim=0).numpy()
        return calculate_score(targets_data, outputs_data, self.task_type, self.metrics)

    def _get_class_weights(self, criterion: Type[torch.nn.Module], labels: Union[np.ndarray, torch.Tensor, pd.DataFrame]
                           ) -> Dict[str, np.ndarray]:
        strategy = get_loss_weight_strategy(criterion)
        weights = strategy(y=labels)
        weights = torch.from_numpy(weights)
        weights = weights.float().to(self.device)
        if criterion.__name__ == 'BCEWithLogitsLoss':
            return {'pos_weight': weights}
        else:
            return {'weight': weights}

    def _data_preprocessing(self, X: torch.Tensor, y: torch.Tensor,
                            ) -> Tuple[torch.Tensor, _NewLossParameters]:
        """
        Depending on the trainer choice, data fed to the network might be pre-processed
        on a different way. That is, in standard training we provide the data to the
        network as we receive it to the loader. Some regularization techniques, like mixup
        alter the data.

        Args:
            X (torch.Tensor): The batch training features
            y (torch.Tensor): The batch training labels

        Returns:
            torch.Tensor: that processes data
            _NewLossParameters: arguments to the new loss function
        """
        raise NotImplementedError

    def _get_new_loss_fn(
        self,
        new_loss_params: _NewLossParameters
    ) -> Callable:  # type: ignore
        """
        Depending on the trainer choice, the criterion is not directly applied to the
        traditional y_pred/y_ground_truth pairs, but rather it might have a slight transformation.
        For example, in the case of mixup training, we need to account for the lambda mixup

        Args:
            new_loss_params (_NewLossParameters):
                Modifiers to the new loss calculation

        Returns:
            Callable: a lambda function that contains the new criterion calculation recipe
        """
        raise NotImplementedError
