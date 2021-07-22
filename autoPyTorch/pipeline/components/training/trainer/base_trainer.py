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


from autoPyTorch.constants import CLASSIFICATION_TASKS, REGRESSION_TASKS, STRING_TO_TASK_TYPES
from autoPyTorch.pipeline.components.training.base_training import autoPyTorchTrainingComponent
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_score
from autoPyTorch.pipeline.components.training.trainer.utils import Lookahead, swa_average_function
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
    """
    Base class for training
    Args:
        weighted_loss (int, default=0): In case for classification, whether to weight
            the loss function according to the distribution of classes in the target
        use_stochastic_weight_averaging (bool, default=True): whether to use stochastic
            weight averaging. Stochastic weight averaging is a simple average of
            multiple points(model parameters) along the trajectory of SGD. SWA
            has been proposed in
            [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407)
        use_snapshot_ensemble (bool, default=True): whether to use snapshot
            ensemble
        se_lastk (int, default=3): Number of snapshots of the network to maintain
        use_lookahead_optimizer (bool, default=True): whether to use lookahead
            optimizer
        random_state:
        **lookahead_config:
    """
    def __init__(self, weighted_loss: int = 0,
                 use_stochastic_weight_averaging: bool = True,
                 use_snapshot_ensemble: bool = True,
                 se_lastk: int = 3,
                 use_lookahead_optimizer: bool = True,
                 random_state: Optional[Union[np.random.RandomState, int]] = None,
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
        self.batch_fit_times = []
        self.data_loading_times = []

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
        numerical_columns: Optional[List[int]] = None
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

        # in case we are using swa, maintain an averaged model,
        if self.use_stochastic_weight_averaging:
            self.swa_model = swa_utils.AveragedModel(self.model, avg_fn=swa_average_function)

        # in case we are using se or swa, initialise budget_threshold to know when to start swa or se
        self._budget_threshold = 0
        if self.use_stochastic_weight_averaging or self.use_snapshot_ensemble:
            assert budget_tracker.max_epochs is not None, "Can only use stochastic weight averaging or snapshot " \
                                                          "ensemble when budget is epochs"
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

        # task type (used for calculating metrics)
        self.task_type = task_type

        # for cutout trainer, we need the list of numerical columns
        self.numerical_columns = numerical_columns

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
        if X['is_cyclic_scheduler']:
            if hasattr(self.scheduler, 'T_cur') and self.scheduler.T_cur == 0 and epoch != 1:
                if self.use_stochastic_weight_averaging:
                    assert self.swa_model is not None, "SWA model can't be none when" \
                                                       " stochastic weight averaging is enabled"
                    self.swa_model.update_parameters(self.model)
                    self.swa_updated = True
                if self.use_snapshot_ensemble:
                    assert self.model_snapshots is not None, "model snapshots container can't be " \
                                                             "none when snapshot ensembling is enabled"
                    model_copy = deepcopy(self.swa_model) if self.use_stochastic_weight_averaging \
                        else deepcopy(self.model)
                    assert model_copy is not None
                    model_copy.cpu()
                    self.model_snapshots.append(model_copy)
                    self.model_snapshots = self.model_snapshots[-self.se_lastk:]
        else:
            if epoch > self._budget_threshold:
                if self.use_stochastic_weight_averaging:
                    assert self.swa_model is not None, "SWA model can't be none when" \
                                                       " stochastic weight averaging is enabled"
                    self.swa_model.update_parameters(self.model)
                    self.swa_updated = True
                if self.use_snapshot_ensemble:
                    assert self.model_snapshots is not None, "model snapshots container can't be " \
                                                             "none when snapshot ensembling is enabled"
                    model_copy = deepcopy(self.swa_model) if self.use_stochastic_weight_averaging \
                        else deepcopy(self.model)
                    assert model_copy is not None
                    model_copy.cpu()
                    self.model_snapshots.append(model_copy)
                    self.model_snapshots = self.model_snapshots[-self.se_lastk:]
        return False

    def train_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int,
                    writer: Optional[SummaryWriter],
                    ) -> Tuple[float, Dict[str, float]]:
        '''
            Trains the model for a single epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): generator of features/label
            epoch (int): The current epoch used solely for tracking purposes

        Returns:
            float: training loss
            Dict[str, float]: scores for each desired metric
        '''

        loss_sum = 0.0
        N = 0
        self.model.train()
        outputs_data = list()
        targets_data = list()

        batch_load_start_time = time.time()
        for step, (data, targets) in enumerate(train_loader):
            self.data_loading_times.append(time.time() - batch_load_start_time)
            batch_train_start = time.time()
            if self.budget_tracker.is_max_time_reached():
                break

            loss, outputs = self.train_step(data, targets)

            self.batch_fit_times.append(time.time()-batch_train_start)
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
            batch_load_start_time = time.time()

        if self.scheduler:
            if 'ReduceLROnPlateau' in self.scheduler.__class__.__name__:
                self.scheduler.step(loss)
            else:
                self.scheduler.step()

        if self.metrics_during_training:
            return loss_sum / N, self.compute_metrics(outputs_data, targets_data)
        else:
            return loss_sum / N, {}

    def cast_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if self.task_type in REGRESSION_TASKS:
            targets = targets.float().to(self.device)
            # make sure that targets will have same shape as outputs (really important for mse loss for example)
            if targets.ndim == 1:
                targets = targets.unsqueeze(1)
        else:
            targets = targets.long().to(self.device)
        return targets

    def train_step(self, data: np.ndarray, targets: np.ndarray) -> Tuple[float, torch.Tensor]:
        """
        Allows to train 1 step of gradient descent, given a batch of train/labels

        Args:
            data (np.ndarray): input features to the network
            targets (np.ndarray): ground truth to calculate loss

        Returns:
            torch.Tensor: The predictions of the network
            float: the loss incurred in the prediction
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

        return loss.item(), outputs

    def evaluate(self, test_loader: torch.utils.data.DataLoader, epoch: int,
                 writer: Optional[SummaryWriter],
                 ) -> Tuple[float, Dict[str, float]]:
        '''
            Evaluates the model in both metrics and criterion

        Args:
            test_loader (torch.utils.data.DataLoader): generator of features/label
            epoch (int): the current epoch for tracking purposes

        Returns:
            float: test loss
            Dict[str, float]: scores for each desired metric
        '''
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

                outputs_data.append(outputs.detach().cpu())
                targets_data.append(targets.detach().cpu())

                if writer:
                    writer.add_scalar(
                        'Val/loss',
                        loss.item(),
                        epoch * len(test_loader) + step,
                    )

        self.model.train()
        return loss_sum / N, self.compute_metrics(outputs_data, targets_data)

    def compute_metrics(self, outputs_data: np.ndarray, targets_data: np.ndarray
                        ) -> Dict[str, float]:
        # TODO: change once Ravin Provides the PR
        outputs_data = torch.cat(outputs_data, dim=0).numpy()
        targets_data = torch.cat(targets_data, dim=0).numpy()
        return calculate_score(targets_data, outputs_data, self.task_type, self.metrics)

    def get_class_weights(self, criterion: Type[torch.nn.Module], labels: Union[np.ndarray, torch.Tensor, pd.DataFrame]
                          ) -> Dict[str, np.ndarray]:
        strategy = get_loss_weight_strategy(criterion)
        weights = strategy(y=labels)
        weights = torch.from_numpy(weights)
        weights = weights.float().to(self.device)
        if criterion.__name__ == 'BCEWithLogitsLoss':
            return {'pos_weight': weights}
        else:
            return {'weight': weights}

    def data_preparation(self, X: np.ndarray, y: np.ndarray,
                         ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Depending on the trainer choice, data fed to the network might be pre-processed
        on a different way. That is, in standard training we provide the data to the
        network as we receive it to the loader. Some regularization techniques, like mixup
        alter the data.

        Args:
            X (np.ndarray): The batch training features
            y (np.ndarray): The batch training labels

        Returns:
            np.ndarray: that processes data
            Dict[str, np.ndarray]: arguments to the criterion function
        """
        raise NotImplementedError()

    def criterion_preparation(self, y_a: np.ndarray, y_b: np.ndarray = None, lam: float = 1.0
                              ) -> Callable:  # type: ignore
        """
        Depending on the trainer choice, the criterion is not directly applied to the
        traditional y_pred/y_ground_truth pairs, but rather it might have a slight transformation.
        For example, in the case of mixup training, we need to account for the lambda mixup

        Args:
            kwargs (Dict): an expanded dictionary with modifiers to the
                                  criterion calculation

        Returns:
            Callable: a lambda that contains the new criterion calculation recipe
        """
        raise NotImplementedError()

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict] = None,
        weighted_loss: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="weighted_loss",
            value_range=[1],
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
            value_range=(3,),
            default_value=3),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        add_hyperparameter(cs, use_stochastic_weight_averaging, CategoricalHyperparameter)
        snapshot_ensemble_flag = False
        if any(use_snapshot_ensemble.value_range):
            snapshot_ensemble_flag = True

        use_snapshot_ensemble = get_hyperparameter(use_snapshot_ensemble, CategoricalHyperparameter)
        cs.add_hyperparameter(use_snapshot_ensemble)

        if snapshot_ensemble_flag:
            se_lastk = get_hyperparameter(se_lastk, Constant)
            cs.add_hyperparameter(se_lastk)
            cond = EqualsCondition(se_lastk, use_snapshot_ensemble, True)
            cs.add_condition(cond)

        lookahead_flag = False
        if any(use_lookahead_optimizer.value_range):
            lookahead_flag = True

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
            if STRING_TO_TASK_TYPES[dataset_properties['task_type']] in CLASSIFICATION_TASKS:
                add_hyperparameter(cs, weighted_loss, Constant)

        return cs
