import collections
import logging.handlers
import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, cast

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
)

import numpy as np

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter

from autoPyTorch.constants import CLASSIFICATION_TASKS, STRING_TO_TASK_TYPES
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    autoPyTorchComponent,
    find_components,
)
from autoPyTorch.pipeline.components.training.losses import get_loss
from autoPyTorch.pipeline.components.training.metrics.utils import get_metrics
from autoPyTorch.pipeline.components.training.trainer.base_trainer import (
    BaseTrainerComponent,
    BudgetTracker,
    RunSummary,
)
from autoPyTorch.utils.common import FitRequirement, get_device_from_fit_dictionary
from autoPyTorch.utils.logging_ import get_named_client_logger

trainer_directory = os.path.split(__file__)[0]
_trainers = find_components(__package__,
                            trainer_directory,
                            BaseTrainerComponent)
_addons = ThirdPartyComponents(BaseTrainerComponent)


def add_trainer(trainer: BaseTrainerComponent) -> None:
    _addons.add_component(trainer)


class TrainerChoice(autoPyTorchChoice):
    """This class is an interface to the PyTorch trainer.


    To map to pipeline terminology, a choice component will implement the epoch
    loop through fit, whereas the component who is chosen will dictate how a single
    epoch happens, that is, how batches of data are fed and used to train the network.

    """

    def __init__(self,
                 dataset_properties: Dict[str, BaseDatasetPropertiesType],
                 random_state: Optional[np.random.RandomState] = None
                 ):

        super().__init__(dataset_properties=dataset_properties,
                         random_state=random_state)
        self.run_summary: Optional[RunSummary] = None
        self.writer: Optional[SummaryWriter] = None
        self.early_stopping_split_type: Optional[str] = None
        self._fit_requirements: Optional[List[FitRequirement]] = [
            FitRequirement("lr_scheduler", (_LRScheduler,), user_defined=False, dataset_property=False),
            FitRequirement("num_run", (int,), user_defined=False, dataset_property=False),
            FitRequirement(
                "optimizer", (Optimizer,), user_defined=False, dataset_property=False),
            FitRequirement("train_data_loader",
                           (torch.utils.data.DataLoader,),
                           user_defined=False, dataset_property=False),
            FitRequirement("val_data_loader",
                           (torch.utils.data.DataLoader,),
                           user_defined=False, dataset_property=False)]
        self.checkpoint_dir: Optional[str] = None

    def get_fit_requirements(self) -> Optional[List[FitRequirement]]:
        return self._fit_requirements

    def get_components(self) -> Dict[str, autoPyTorchComponent]:
        """Returns the available trainer components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all components available
                as choices for learning rate scheduling
        """
        components: Dict[str, autoPyTorchComponent] = collections.OrderedDict()
        components.update(_trainers)
        components.update(_addons.components)
        return components

    def get_hyperparameter_search_space(
        self,
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        default: Optional[str] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> ConfigurationSpace:
        """Returns the configuration space of the current chosen components

        Args:
            dataset_properties (Optional[Dict[str, BaseDatasetPropertiesType]]): Describes the dataset to work on
            default (Optional[str]): Default scheduler to use
            include: Optional[Dict[str, Any]]: what components to include. It is an exhaustive
                list, and will exclusively use this components.
            exclude: Optional[Dict[str, Any]]: which components to skip

        Returns:
            ConfigurationSpace: the configuration space of the hyper-parameters of the
                 chosen component
        """
        cs = ConfigurationSpace()

        if dataset_properties is None:
            dataset_properties = {}

        dataset_properties = {**self.dataset_properties, **dataset_properties}

        # Compile a list of legal trainers for this problem
        available_trainers = self.get_available_components(
            dataset_properties=dataset_properties,
            include=include, exclude=exclude)

        if len(available_trainers) == 0:
            raise ValueError("No trainer found")

        if default is None:
            defaults = ['StandardTrainer',
                        ]
            for default_ in defaults:
                if default_ in available_trainers:
                    default = default_
                    break
        updates = self._get_search_space_updates()
        if '__choice__' in updates.keys():
            choice_hyperparameter = updates['__choice__']
            if not set(choice_hyperparameter.value_range).issubset(available_trainers):
                raise ValueError("Expected given update for {} to have "
                                 "choices in {} got {}".format(self.__class__.__name__,
                                                               available_trainers,
                                                               choice_hyperparameter.value_range))
            trainer = CategoricalHyperparameter('__choice__',
                                                choice_hyperparameter.value_range,
                                                default_value=choice_hyperparameter.default_value)
        else:
            trainer = CategoricalHyperparameter(
                '__choice__',
                list(available_trainers.keys()),
                default_value=default
            )
        cs.add_hyperparameter(trainer)
        for name in trainer.choices:
            updates = self._get_search_space_updates(prefix=name)
            config_space = available_trainers[name].get_hyperparameter_search_space(dataset_properties,  # type:ignore
                                                                                    **updates)
            parent_hyperparameter = {'parent': trainer, 'value': name}
            cs.add_configuration_space(
                name,
                config_space,
                parent_hyperparameter=parent_hyperparameter
            )

        self.configuration_space_ = cs
        self.dataset_properties_ = dataset_properties
        return cs

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """The transform function calls the transform function of the
        underlying model and returns the transformed array.

        Args:
            X (np.ndarray): input features

        Returns:
            np.ndarray: Transformed features
        """
        X.update({'run_summary': self.run_summary})
        return X

    def fit(self, X: Dict[str, Any], y: Any = None, **kwargs: Any) -> autoPyTorchComponent:
        """
        Fits a component by using an input dictionary with pre-requisites

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API

        Returns:
            A instance of self
        """
        # Make sure that the prerequisites are there
        self.check_requirements(X, y)

        # Setup the logger
        self.logger = get_named_client_logger(
            name=f"{X['num_run']}_{time.time()}",
            # Log to a user provided port else to the default logging port
            port=X['logger_port'] if 'logger_port' in X else logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        )

        # Call the actual fit function.
        self._fit(
            X=X,
            y=y,
            **kwargs
        )

        return cast(autoPyTorchComponent, self.choice)

    def prepare_trainer(self, X: Dict) -> None:
        """
        prepare trainer, forecasting tasks require more parameters
        """
        assert self.choice is not None

        # Support additional user metrics
        metrics = get_metrics(dataset_properties=X['dataset_properties'])
        if 'additional_metrics' in X:
            metrics.extend(get_metrics(dataset_properties=X['dataset_properties'], names=X['additional_metrics']))
        if 'optimize_metric' in X and X['optimize_metric'] not in [m.name for m in metrics]:
            metrics.extend(get_metrics(dataset_properties=X['dataset_properties'], names=[X['optimize_metric']]))
        additional_losses = X['additional_losses'] if 'additional_losses' in X else None

        labels = self._get_train_label(X)

        self.choice.prepare(
            model=X['network'],
            metrics=metrics,
            criterion=get_loss(X['dataset_properties'],
                               name=additional_losses),
            budget_tracker=self.budget_tracker,
            optimizer=X['optimizer'],
            device=get_device_from_fit_dictionary(X),
            metrics_during_training=X['metrics_during_training'],
            scheduler=X['lr_scheduler'],
            task_type=STRING_TO_TASK_TYPES[X['dataset_properties']['task_type']],
            labels=labels,
            step_interval=X['step_interval']
        )

    def get_budget_tracker(self, X: Dict) -> BudgetTracker:
        return BudgetTracker(
            budget_type=X['budget_type'],
            max_runtime=X['runtime'] if 'runtime' in X else None,
            max_epochs=X['epochs'] if 'epochs' in X else None,
        )

    def _fit(self, X: Dict[str, Any], y: Any = None, **kwargs: Any) -> 'TrainerChoice':
        """
        Fits a component by using an input dictionary with pre-requisites

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API

        Returns:
            A instance of self
        """

        # Comply with mypy
        # Notice that choice here stands for the component choice framework,
        # where we dynamically build the configuration space by selecting the available
        # component choices. In this case, is what trainer choices are available
        assert self.choice is not None

        # Setup a Logger and other logging support
        # Writer is not pickable -- make sure it is not saved in self
        writer = None
        if 'use_tensorboard_logger' in X and X['use_tensorboard_logger']:
            writer = SummaryWriter(log_dir=X['backend'].temporary_directory)

        if X["torch_num_threads"] > 0:
            torch.set_num_threads(X["torch_num_threads"])

        self.budget_tracker = self.get_budget_tracker(X)

        self.prepare_trainer(X)

        total_parameter_count, trainable_parameter_count = self.count_parameters(X['network'])
        self.run_summary = RunSummary(
            total_parameter_count,
            trainable_parameter_count,
            optimize_metric=None if not X['metrics_during_training'] else X.get('optimize_metric'),
        )

        if X['val_data_loader'] is not None:
            self.early_stopping_split_type = 'val'
        else:
            self.early_stopping_split_type = 'train'

        epoch = 1

        while True:

            # prepare epoch
            start_time = time.time()

            self.choice.on_epoch_start(X=X, epoch=epoch)

            # training
            train_loss, train_metrics = self.choice.train_epoch(
                train_loader=X['train_data_loader'],
                epoch=epoch,
                writer=writer,
            )

            # its fine if train_loss is None due to `is_max_time_reached()`
            if train_loss is None:
                if self.budget_tracker.is_max_time_reached():
                    break
                else:
                    raise RuntimeError("Got an unexpected None in `train_loss`.")

            val_loss, val_metrics, test_loss, test_metrics = None, {}, None, {}
            if self.eval_valid_each_epoch(X):
                if X['val_data_loader']:
                    val_loss, val_metrics = self.choice.evaluate(X['val_data_loader'], epoch, writer)
                if 'test_data_loader' in X and X['test_data_loader']:
                    test_loss, test_metrics = self.choice.evaluate(X['test_data_loader'], epoch, writer)

            # Save training information
            self.run_summary.add_performance(
                epoch=epoch,
                start_time=start_time,
                end_time=time.time(),
                train_loss=train_loss,
                val_loss=val_loss,
                test_loss=test_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
            )

            # Save the weights of the best model and, if patience
            # exhausted break training
            if self.early_stop_handler(X):
                break

            if self.choice.on_epoch_end(X=X, epoch=epoch):
                break

            self.logger.debug(self.run_summary.repr_last_epoch())

            # Reached max epoch on next iter, don't even go there
            if (
                    self.budget_tracker.is_max_epoch_reached(epoch + 1)
                    or self.budget_tracker.is_max_time_reached()
            ):
                break

            epoch += 1

            if 'cuda' in X['device']:
                torch.cuda.empty_cache()

        if self.run_summary.is_empty():
            raise RuntimeError("Budget exhausted without finishing an epoch.")

        # wrap up -- add score if not evaluating every epoch
        if not self.eval_valid_each_epoch(X):
            if X['val_data_loader']:
                val_loss, val_metrics = self.choice.evaluate(X['val_data_loader'], epoch, writer)
            if 'test_data_loader' in X and X['val_data_loader']:
                test_loss, test_metrics = self.choice.evaluate(X['test_data_loader'], epoch, writer)
            self.run_summary.add_performance(
                epoch=epoch,
                start_time=start_time,
                end_time=time.time(),
                train_loss=train_loss,
                val_loss=val_loss,
                test_loss=test_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
            )
            self.save_model_for_ensemble()

        # As training have finished, load the best weight
        if self.checkpoint_dir is not None:
            self._load_best_weights_and_clean_checkpoints(X)

        self.logger.info(f"Finished training with {self.run_summary.repr_last_epoch()}")

        # Tag as fitted
        self.fitted_ = True

        return self

    def _get_train_label(self, X: Dict[str, Any]) -> List[int]:
        """
        Verifies and validates the labels from train split.
        """
        # Ensure that the split is not missing any class.
        labels: List[int] = X['y_train'][X['backend'].load_datamanager().splits[X['split_id']][0]]
        if STRING_TO_TASK_TYPES[X['dataset_properties']['task_type']] in CLASSIFICATION_TASKS:
            unique_labels = len(np.unique(labels))
            if unique_labels < X['dataset_properties']['output_shape']:
                raise ValueError(f"Expected number of unique labels {unique_labels} in train split: {X['split_id']}"
                                 f" to be = num_classes {X['dataset_properties']['output_shape']}."
                                 f" Consider using stratified splitting strategies.")

        return labels

    def _load_best_weights_and_clean_checkpoints(self, X: Dict[str, Any]) -> None:
        """
        Load the best model until the last epoch and delete all the files for checkpoints.

        Args:
            X (Dict[str, Any]): Dependencies needed by current component to perform fit
        """
        assert self.checkpoint_dir is not None  # mypy
        assert self.run_summary is not None  # mypy
        assert self.early_stopping_split_type is not None  # mypy

        best_path = os.path.join(self.checkpoint_dir, 'best.pth')
        best_epoch = self.run_summary.get_best_epoch(split_type=self.early_stopping_split_type)
        self.logger.debug(f" Early stopped model {X['num_run']} on epoch {best_epoch}")
        # We will stop the training. Load the last best performing weights
        X['network'].load_state_dict(torch.load(best_path))

        # Clean the temp dir
        shutil.rmtree(self.checkpoint_dir)
        self.checkpoint_dir = None

    def early_stop_handler(self, X: Dict[str, Any]) -> bool:
        """
        If early stopping is enabled, this procedure stops the training after a
        given patience
        Args:
            X (Dict[str, Any]): Dictionary with fitted parameters. It is a message passing
                mechanism, in which during a transform, a components adds relevant information
                so that further stages can be properly fitted

        Returns:
            bool: If true, training should be stopped
        """
        assert self.run_summary is not None
        assert self.early_stopping_split_type is not None  # mypy

        # Allow to disable early stopping
        if X['early_stopping'] is None or X['early_stopping'] < 0:
            return False

        # Store the best weights seen so far:
        if self.checkpoint_dir is None:
            self.checkpoint_dir = tempfile.mkdtemp(dir=X['backend'].temporary_directory)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        last_epoch = self.run_summary.get_last_epoch()
        best_epoch = self.run_summary.get_best_epoch(split_type=self.early_stopping_split_type)
        epochs_since_best = last_epoch - best_epoch

        # Save the checkpoint if there is a new best epoch
        best_path = os.path.join(self.checkpoint_dir, 'best.pth')
        if epochs_since_best == 0:
            torch.save(X['network'].state_dict(), best_path)

        return epochs_since_best > cast(int, X['early_stopping'])

    def eval_valid_each_epoch(self, X: Dict[str, Any]) -> bool:
        """
        Returns true if we are supposed to evaluate the model on every epoch,
        on the validation data. Usually, we only validate the data at the end,
        but in the case of early stopping, is appealing to evaluate each epoch.
        Args:
            X (Dict[str, Any]): Dictionary with fitted parameters. It is a message passing
                mechanism, in which during a transform, a components adds relevant information
                so that further stages can be properly fitted

        Returns:
            bool: if True, the model is evaluated in every epoch

        """
        if 'early_stopping' in X and X['early_stopping']:
            return True

        # We need to know if we should reduce the rate based on val loss
        if 'ReduceLROnPlateau' in X['lr_scheduler'].__class__.__name__:
            return True

        return False

    def check_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
        """
        A mechanism in code to ensure the correctness of the fit dictionary
        It recursively makes sure that the children and parent level requirements
        are honored before fit.

        Args:
            X (Dict[str, Any]): Dictionary with fitted parameters. It is a message passing
                mechanism, in which during a transform, a components adds relevant information
                so that further stages can be properly fitted
        """

        # make sure the parent requirements are honored
        super().check_requirements(X, y)

        # We need a working dir in where to put our data
        if 'backend' not in X:
            raise ValueError('Need a backend to provide the working directory, '
                             "yet 'backend' was not found in the fit dictionary")

        # Whether we should evaluate metrics during training or no
        if 'metrics_during_training' not in X:
            raise ValueError('Missing metrics_during_training in the fit dictionary')

        # Setup Components
        if 'lr_scheduler' not in X:
            raise ValueError("Learning rate scheduler not found in the fit dictionary!")

        if 'network' not in X:
            raise ValueError("Network not found in the fit dictionary!")

        if 'optimizer' not in X:
            raise ValueError("Optimizer not found in the fit dictionary!")

        # Training Components
        if 'train_data_loader' not in X:
            raise ValueError("train_data_loader not found in the fit dictionary!")

        if 'val_data_loader' not in X:
            raise ValueError("val_data_loader not found in the fit dictionary!")

        if 'budget_type' not in X:
            raise ValueError("Budget type not found in the fit dictionary!")
        else:
            if 'epochs' not in X or 'runtime' not in X or 'epoch_or_time' not in X:
                if X['budget_type'] in ['epochs', 'epoch_or_time'] and 'epochs' not in X:
                    raise ValueError("Budget type is epochs but "
                                     "no epochs was not found in the fit dictionary!")
                elif X['budget_type'] in ['runtime', 'epoch_or_time'] and 'runtime' not in X:
                    raise ValueError("Budget type is runtime but "
                                     "no maximum number of seconds was provided!")
            else:
                raise ValueError("Unsupported budget type provided: {}".format(
                    X['budget_type']
                ))

        if 'num_run' not in X:
            raise ValueError('To fit a trainer, expected fit dictionary to have a num_run')

        for config_option in ["torch_num_threads", 'device']:
            if config_option not in X:
                raise ValueError("To fit a trainer, expected fit dictionary to have a {}".format(
                    config_option
                ))

        # For early stopping, we need to know the patience
        if 'early_stopping' not in X:
            raise ValueError('To fit a Trainer, expected fit dictionary to have early_stopping')

    @staticmethod
    def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
        """
        A method to get the total/trainable parameter count from the model

        Args:
            model (torch.nn.Module): the module from which to count parameters

        Returns:
            total_parameter_count: the total number of parameters of the model
            trainable_parameter_count: only the parameters being optimized
        """
        total_parameter_count = sum(
            p.numel() for p in model.parameters())
        trainable_parameter_count = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        return total_parameter_count, trainable_parameter_count

    def save_model_for_ensemble(self) -> str:
        raise NotImplementedError()

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = str(self.run_summary)
        return string
