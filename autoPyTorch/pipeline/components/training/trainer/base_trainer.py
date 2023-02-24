import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant
)

import numpy as np

import pandas as pd

from sklearn.utils import check_random_state

import torch
from torch.optim import Optimizer, swa_utils
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter

from autoPyTorch.constants import CLASSIFICATION_TASKS, FORECASTING_TASKS, REGRESSION_TASKS, STRING_TO_TASK_TYPES
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.lr_scheduler.constants import StepIntervalUnit
from autoPyTorch.pipeline.components.training.base_training import autoPyTorchTrainingComponent
from autoPyTorch.pipeline.components.training.metrics.metrics import (
    CLASSIFICATION_METRICS,
    FORECASTING_METRICS,
    REGRESSION_METRICS,
)
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_score
from autoPyTorch.pipeline.components.training.trainer.utils import Lookahead, swa_update
from autoPyTorch.utils.common import FitRequirement, HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter
from autoPyTorch.utils.implementations import get_loss_weight_strategy


class BudgetTracker(object):
    def __init__(self,
                 budget_type: str,
                 max_epochs: Optional[int] = None,
                 max_runtime: Optional[int] = None,
                 ):
        """
        An object for tracking when to stop the network training.
        It handles epoch based criteria as well as training based criteria.

        It also allows to define a 'epoch_or_time' budget type, which means, the first of them both which is
        exhausted, is honored

        Args:
            budget_type (str):
                Type of budget to be used when fitting the pipeline.
                Possible values are 'epochs', 'runtime', or 'epoch_or_time'
            max_epochs (Optional[int], default=None):
                Maximum number of epochs to train the pipeline for
            max_runtime (Optional[int], default=None):
                Maximum number of seconds to train the pipeline for
        """
        self.start_time = time.time()
        self.budget_type = budget_type
        self.max_epochs = max_epochs
        self.max_runtime = max_runtime

    def is_max_epoch_reached(self, epoch: int) -> bool:
        """
        For budget type 'epoch' or 'epoch_or_time' return True if the maximum number of epochs is reached.

        Args:
            epoch (int):
                the current epoch

        Returns:
            bool:
                True if the current epoch is larger than the maximum epochs, False otherwise.
                Additionally, returns False if the run is without this constraint.
        """
        # Make None a method to run without this constraint
        if self.max_epochs is None:
            return False
        if self.budget_type in ['epochs', 'epoch_or_time'] and epoch > self.max_epochs:
            return True
        return False

    def is_max_time_reached(self) -> bool:
        """
        For budget type 'runtime' or 'epoch_or_time' return True if the maximum runtime is reached.

        Returns:
            bool:
                True if the maximum runtime is reached, False otherwise.
                Additionally, returns False if the run is without this constraint.
        """
        # Make None a method to run without this constraint
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
        optimize_metric: Optional[str] = None,
    ) -> None:
        """
        A useful object to track performance per epoch.

        It allows to track train, validation and test information not only for debug, but for research purposes
        (Like understanding overfit).

        It does so by tracking a metric/loss at the end of each epoch.

        Args:
            total_parameter_count (float):
                the total number of parameters of the model
            trainable_parameter_count (float):
                only the parameters being optimized
            optimize_metric (Optional[str], default=None):
                name of the metric that is used to evaluate a pipeline.
        """
        self.performance_tracker: Dict[str, Dict] = {
            'start_time': {},
            'end_time': {},
        }

        self.total_parameter_count = total_parameter_count
        self.trainable_parameter_count = trainable_parameter_count
        self.optimize_metric = optimize_metric

        # Allow to track the training performance
        self.performance_tracker['train_loss'] = {}

        # Allow to track the val performance
        self.performance_tracker['val_loss'] = {}

        # Allow to track the test performance
        self.performance_tracker['test_loss'] = {}

        # Allow to track the metrics performance
        for metric in ['train_metrics', 'val_metrics', 'test_metrics']:
            self.performance_tracker[metric] = {}

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
        Tracks performance information about the run, useful for plotting individual runs.

        Args:
            epoch (int):
                the current epoch
            start_time (float):
                timestamp at the beginning of current epoch
            end_time (float):
                timestamp when gathering the information after the current epoch
            train_loss (float):
                the training loss
            train_metrics (Dict[str, float]):
                training scores for each desired metric
            val_metrics (Dict[str, float]):
                validation scores for each desired metric
            test_metrics (Dict[str, float]):
                test scores for each desired metric
            val_loss (Optional[float], default=None):
                the validation loss
            test_loss (Optional[float], default=None):
                the test loss

        Returns:
            None
        """
        self.performance_tracker['train_loss'][epoch] = train_loss
        self.performance_tracker['val_loss'][epoch] = val_loss
        self.performance_tracker['test_loss'][epoch] = test_loss
        self.performance_tracker['start_time'][epoch] = start_time
        self.performance_tracker['end_time'][epoch] = end_time
        self.performance_tracker['train_metrics'][epoch] = train_metrics
        self.performance_tracker['val_metrics'][epoch] = val_metrics
        self.performance_tracker['test_metrics'][epoch] = test_metrics

    def get_best_epoch(self, split_type: str = 'val') -> int:
        """
        Get the epoch with the best metric.

        Args:
            split_type (str, default=val):
                Which split's metric to consider.
                Possible values are 'train' or 'val

        Returns:
            int:
                the epoch with the best metric
        """
        # If we compute for optimization, prefer the performance
        # metric to the loss
        if self.optimize_metric is not None:
            metrics_type = f"{split_type}_metrics"
            if self.optimize_metric in CLASSIFICATION_METRICS:
                scorer = CLASSIFICATION_METRICS[self.optimize_metric]
            elif self.optimize_metric in REGRESSION_METRICS:
                scorer = REGRESSION_METRICS[self.optimize_metric]
            elif self.optimize_metric in FORECASTING_METRICS:
                scorer = FORECASTING_METRICS[self.optimize_metric]
            else:
                raise NotImplementedError(f"Unsupported optimizer metric: {self.optimize_metric}")

            # Some metrics maximize, other minimize!
            opt_func = np.argmax if scorer._sign > 0 else np.argmin
            return int(opt_func(
                [metrics[self.optimize_metric] for metrics in self.performance_tracker[metrics_type].values()]
            )) + 1  # Epochs start at 1
        else:
            loss_type = f"{split_type}_loss"
            return int(np.argmin(
                list(self.performance_tracker[loss_type].values()),
            )) + 1  # Epochs start at 1

    def get_last_epoch(self) -> int:
        """
        Get the last epoch.

        Returns:
            int:
                the last epoch
        """
        if 'train_loss' not in self.performance_tracker:
            return 0
        else:
            return int(max(self.performance_tracker['train_loss'].keys()))

    def repr_last_epoch(self) -> str:
        """
        For debug purposes, returns a nice representation of last epoch
        performance

        Returns:
            str:
                A nice representation of the last epoch
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

    def is_empty(self) -> bool:
        """
        Checks if the object is empty or not

        Returns:
            bool:
                True if the object is empty, False otherwise
        """
        # if train_loss is empty, we can be sure that RunSummary is empty.
        return not bool(self.performance_tracker['train_loss'])


class BaseTrainerComponent(autoPyTorchTrainingComponent):
    """
    Base class for training.

    Args:
        weighted_loss (int, default=0):
            In case for classification, whether to weight the loss function according to the distribution of classes
            in the target
        use_stochastic_weight_averaging (bool, default=True):
            whether to use stochastic weight averaging. Stochastic weight averaging is a simple average of
            multiple points(model parameters) along the trajectory of SGD. SWA has been proposed in
            [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407)
        use_snapshot_ensemble (bool, default=True):
            whether to use snapshot ensemble
        se_lastk (int, default=3):
            Number of snapshots of the network to maintain
        use_lookahead_optimizer (bool, default=True):
            whether to use lookahead optimizer
        random_state (Optional[np.random.RandomState]):
            Object that contains a seed and allows for reproducible results
        swa_model (Optional[torch.nn.Module], default=None):
            Averaged model used for Stochastic Weight Averaging
        model_snapshots (Optional[List[torch.nn.Module]], default=None):
            List of model snapshots in case snapshot ensemble is used
        **lookahead_config (Any):
            keyword arguments for the lookahead optimizer including:
            la_steps (int):
                number of lookahead steps
            la_alpha (float):
                linear interpolation factor. 1.0 recovers the inner optimizer.
    """
    def __init__(self, weighted_loss: int = 0,
                 use_stochastic_weight_averaging: bool = True,
                 use_snapshot_ensemble: bool = True,
                 se_lastk: int = 3,
                 use_lookahead_optimizer: bool = True,
                 random_state: Optional[np.random.RandomState] = None,
                 swa_model: Optional[torch.nn.Module] = None,
                 model_snapshots: Optional[List[torch.nn.Module]] = None,
                 **lookahead_config: Any) -> None:
        if random_state is None:
            # A trainer components need a random state for
            # sampling -- for example in MixUp training
            self.random_state = check_random_state(1)
        else:
            self.random_state = random_state
        super().__init__(random_state=self.random_state)
        self.weighted_loss = weighted_loss
        self.use_stochastic_weight_averaging = use_stochastic_weight_averaging
        self.use_snapshot_ensemble = use_snapshot_ensemble
        self.se_lastk = se_lastk
        self.use_lookahead_optimizer = use_lookahead_optimizer
        self.swa_model = swa_model
        self.model_snapshots = model_snapshots
        # Add default values for the lookahead optimizer
        if len(lookahead_config) == 0:
            lookahead_config = {f'{Lookahead.__name__}:la_steps': 6,
                                f'{Lookahead.__name__}:la_alpha': 0.6}
        self.lookahead_config = lookahead_config
        self.add_fit_requirements([
            FitRequirement("is_cyclic_scheduler", (bool,), user_defined=False, dataset_property=False),
        ])

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
        model_final_activation: Optional[torch.nn.Module] = None,
        numerical_columns: Optional[List[int]] = None,
        **kwargs: Dict
    ) -> None:

        # Save the device to be used
        self.device = device

        # Setup the metrics
        self.metrics = metrics

        # Weights for the loss function
        kwargs = {}
        if self.weighted_loss:
            kwargs = self.get_class_weights(criterion, labels)

        # Setup the loss function
        self.criterion = criterion(**kwargs)
        # setup the model
        self.model = model.to(device)
        self.model_final_activation = model_final_activation.to(device)

        # in case we are using swa, maintain an averaged model,
        if self.use_stochastic_weight_averaging:
            self.swa_model = swa_utils.AveragedModel(self.model, avg_fn=swa_update)

        # in case we are using se or swa, initialise budget_threshold to know when to start swa or se
        self._budget_threshold = 0
        if self.use_stochastic_weight_averaging or self.use_snapshot_ensemble:
            if budget_tracker.max_epochs is None:
                raise ValueError("Budget for stochastic weight averaging or snapshot ensemble must be `epoch`.")

            self._budget_threshold = int(0.75 * budget_tracker.max_epochs)

        # in case we are using se, initialise list to store model snapshots
        if self.use_snapshot_ensemble:
            self.model_snapshots = list()

        # in case we are using, swa or se with early stopping,
        # we need to make sure network params are only updated
        # from the swa model if the swa model was actually updated
        self.swa_updated: bool = False

        # setup the optimizers
        if self.use_lookahead_optimizer:
            optimizer = Lookahead(optimizer=optimizer, config=self.lookahead_config)
        self.optimizer = optimizer

        # The budget tracker
        self.budget_tracker = budget_tracker

        # For best performance, we allow option to prevent comparing metrics every time
        self.metrics_during_training = metrics_during_training

        # Scheduler
        self.scheduler = scheduler
        self.step_interval = step_interval

        # task type (used for calculating metrics)
        self.task_type = task_type

        # for cutout trainer, we need the list of numerical columns
        self.numerical_columns = numerical_columns

    def on_epoch_start(self, X: Dict[str, Any], epoch: int) -> None:
        """
        Optional placeholder for AutoPytorch Extensions.
        A user can define what happens on every epoch start or every epoch end.

        Args:
            X (Dict[str, Any]):
                Dictionary with fitted parameters. It is a message passing mechanism, in which during a transform,
                a components adds relevant information so that further stages can be properly fitted
            epoch (int):
                the current epoch
        """
        pass

    def _swa_update(self) -> None:
        """
        Perform Stochastic Weight Averaging model update
        """
        if self.swa_model is None:
            raise ValueError("SWA model cannot be none when stochastic weight averaging is enabled")
        self.swa_model.update_parameters(self.model)
        self.swa_updated = True

    def _se_update(self, epoch: int) -> None:
        """
        Add latest model or swa_model to model snapshot ensemble

        Args:
            epoch (int):
                current epoch
        """
        if self.model_snapshots is None:
            raise ValueError("model snapshots cannot be None when snapshot ensembling is enabled")
        is_last_epoch = (epoch == self.budget_tracker.max_epochs)
        if is_last_epoch and self.use_stochastic_weight_averaging:
            model_copy = deepcopy(self.swa_model)
        else:
            model_copy = deepcopy(self.model)

        assert model_copy is not None
        model_copy.cpu()
        self.model_snapshots.append(model_copy)
        self.model_snapshots = self.model_snapshots[-self.se_lastk:]

    def on_epoch_end(self, X: Dict[str, Any], epoch: int) -> bool:
        """
        Optional placeholder for AutoPytorch Extensions.
        A user can define what happens on every epoch start or every epoch end.
        If returns True, the training is stopped.

        Args:
            X (Dict[str, Any]):
                Dictionary with fitted parameters. It is a message passing mechanism, in which during a transform,
                a components adds relevant information so that further stages can be properly fitted
            epoch (int):
                the current epoch

        """
        if X['is_cyclic_scheduler']:
            if hasattr(self.scheduler, 'T_cur') and self.scheduler.T_cur == 0 and epoch != 1:
                if self.use_stochastic_weight_averaging:
                    self._swa_update()
                if self.use_snapshot_ensemble:
                    self._se_update(epoch=epoch)
        else:
            if epoch > self._budget_threshold and self.use_stochastic_weight_averaging:
                self._swa_update()

            if (
                self.use_snapshot_ensemble
                and self.budget_tracker.max_epochs is not None
                and epoch > (self.budget_tracker.max_epochs - self.se_lastk)
            ):
                self._se_update(epoch=epoch)
        return False

    def _scheduler_step(
        self,
        step_interval: StepIntervalUnit,
        loss: Optional[float] = None
    ) -> None:

        if self.step_interval != step_interval:
            return

        if not self.scheduler:  # skip if no scheduler defined
            return

        try:
            """ Some schedulers might use metrics """
            self.scheduler.step(metrics=loss)
        except TypeError:
            self.scheduler.step()

    def train_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int,
                    writer: Optional[SummaryWriter],
                    ) -> Tuple[Optional[float], Dict[str, float]]:
        """
        Train the model for a single epoch.

        Args:
            train_loader (torch.utils.data.DataLoader):
                generator of features/label
            epoch (int):
                The current epoch used solely for tracking purposes
            writer (Optional[SummaryWriter]):
                Object to keep track of the training loss in an event file

        Returns:
            float:
                training loss
            Dict[str, float]:
                scores for each desired metric
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
                # Store probability data
                if self.model_final_activation is not None:
                    outputs = self.model_final_activation(outputs)
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

        if N == 0:
            return None, {}

        self._scheduler_step(step_interval=StepIntervalUnit.epoch, loss=loss_sum / N)

        if self.metrics_during_training:
            return loss_sum / N, self.compute_metrics(outputs_data, targets_data)
        else:
            return loss_sum / N, {}

    def cast_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if self.task_type in (REGRESSION_TASKS + FORECASTING_TASKS):
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
            data (torch.Tensor):
                input features to the network
            targets (torch.Tensor):
                ground truth to calculate loss

        Returns:
            torch.Tensor:
                The predictions of the network
            float:
                the loss incurred in the prediction
        """
        # prepare
        data = data.float().to(self.device)
        targets = self.cast_targets(targets)

        data, criterion_kwargs = self.data_preparation(data, targets)

        # training
        self.optimizer.zero_grad()
        outputs = self.model(data)
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
            test_loader (torch.utils.data.DataLoader):
                generator of features/label
            epoch (int):
                the current epoch for tracking purposes
            writer (Optional[SummaryWriter]):
                Object to keep track of the test loss in an event file

        Returns:
            float:
                test loss
            Dict[str, float]:
                scores for each desired metric
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
                targets = self.cast_targets(targets)

                outputs = self.model(data)

                loss = self.criterion(outputs, targets)
                loss_sum += loss.item() * batch_size
                N += batch_size

                # Store probability data
                if self.model_final_activation is not None:
                    outputs = self.model_final_activation(outputs)
                outputs_data.append(outputs.detach().cpu())
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
        return calculate_score(targets_data, outputs_data, self.task_type, self.metrics)

    def get_class_weights(self, criterion: Type[torch.nn.Module], labels: Union[np.ndarray, torch.Tensor, pd.DataFrame]
                          ) -> Dict[str, torch.Tensor]:
        strategy = get_loss_weight_strategy(criterion)
        weights = strategy(y=labels)
        weights = torch.from_numpy(weights)
        weights = weights.float().to(self.device)
        if criterion.__name__ == 'BCEWithLogitsLoss':
            return {'pos_weight': weights}
        else:
            return {'weight': weights}

    def data_preparation(self, X: torch.Tensor, y: torch.Tensor,
                         ) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
        """
        Depending on the trainer choice, data fed to the network might be pre-processed on a different way. That is,
        in standard training we provide the data to the network as we receive it to the loader. Some regularization
        techniques, like mixup alter the data.

        Args:
            X (torch.Tensor):
                The batch training features
            y (torch.Tensor):
                The batch training labels

        Returns:
            torch.Tensor: that processes data
            Dict[str, np.ndarray]: arguments to the criterion function
                                   TODO: Fix this typing. It is not np.ndarray.
        """
        raise NotImplementedError

    def criterion_preparation(self, y_a: torch.Tensor, y_b: torch.Tensor = None, lam: float = 1.0
                              ) -> Callable:  # type: ignore
        """
        Depending on the trainer choice, the criterion is not directly applied to the traditional
        y_pred/y_ground_truth pairs, but rather it might have a slight transformation.
        For example, in the case of mixup training, we need to account for the lambda mixup

        Args:
            y_a (torch.Tensor):
                the batch label of the first training example used in trainer
            y_b (torch.Tensor, default=None):
                if applicable, the batch label of the second training example used in trainer
            lam (float):
                trainer coefficient

        Returns:
            Callable:
                a lambda function that contains the new criterion calculation recipe
        """
        raise NotImplementedError()

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        weighted_loss: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="weighted_loss",
            value_range=(1, ),
            default_value=1),
        la_steps: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="la_steps",
            value_range=(5, 10),
            default_value=6,
            log=False),
        la_alpha: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="la_alpha",
            value_range=(0.5, 0.8),
            default_value=0.6,
            log=False),
        use_lookahead_optimizer: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="use_lookahead_optimizer",
            value_range=(True, False),
            default_value=True),
        use_stochastic_weight_averaging: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="use_stochastic_weight_averaging",
            value_range=(True, False),
            default_value=True),
        use_snapshot_ensemble: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="use_snapshot_ensemble",
            value_range=(True, False),
            default_value=True),
        se_lastk: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="se_lastk",
            value_range=(3, ),
            default_value=3),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        add_hyperparameter(cs, use_stochastic_weight_averaging, CategoricalHyperparameter)
        snapshot_ensemble_flag = any(use_snapshot_ensemble.value_range)

        use_snapshot_ensemble = get_hyperparameter(use_snapshot_ensemble, CategoricalHyperparameter)
        cs.add_hyperparameter(use_snapshot_ensemble)

        if snapshot_ensemble_flag:
            se_lastk = get_hyperparameter(se_lastk, Constant)
            cs.add_hyperparameter(se_lastk)
            cond = EqualsCondition(se_lastk, use_snapshot_ensemble, True)
            cs.add_condition(cond)

        lookahead_flag = any(use_lookahead_optimizer.value_range)
        use_lookahead_optimizer = get_hyperparameter(use_lookahead_optimizer, CategoricalHyperparameter)
        cs.add_hyperparameter(use_lookahead_optimizer)

        if lookahead_flag:
            la_config_space = Lookahead.get_hyperparameter_search_space(la_steps=la_steps,
                                                                        la_alpha=la_alpha)
            parent_hyperparameter = {'parent': use_lookahead_optimizer, 'value': True}
            cs.add_configuration_space(
                Lookahead.__name__,
                la_config_space,
                parent_hyperparameter=parent_hyperparameter
            )

        """
        # TODO, decouple the weighted loss from the trainer
        if dataset_properties is not None:
            if STRING_TO_TASK_TYPES[dataset_properties['task_type']] in CLASSIFICATION_TASKS:
                add_hyperparameter(cs, weighted_loss, CategoricalHyperparameter)
        """
        # TODO, decouple the weighted loss from the trainer. Uncomment the code above and
        # remove the code below. Also update the method signature, so the weighted loss
        # is not a constant.
        if dataset_properties is not None:
            if STRING_TO_TASK_TYPES[str(dataset_properties['task_type'])] in CLASSIFICATION_TASKS:
                add_hyperparameter(cs, weighted_loss, Constant)

        return cs
