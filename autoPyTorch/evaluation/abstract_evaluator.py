import logging.handlers
import time
import warnings
from multiprocessing.queues import Queue
from typing import Any, Dict, List, NamedTuple, Optional, Union, no_type_check

from ConfigSpace import Configuration

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier

from smac.tae import StatusType

from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.constants import (
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
    STRING_TO_TASK_TYPES
)
from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.evaluation.pipeline_class_collection import (
    get_default_pipeline_config,
    get_pipeline_class
)
from autoPyTorch.evaluation.utils import (
    DisableFileOutputParameters,
    VotingRegressorWrapper,
    ensure_prediction_array_sizes
)
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.utils import (
    calculate_loss,
    get_metrics,
)
from autoPyTorch.utils.common import dict_repr
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from autoPyTorch.utils.logging_ import PicklableClientLogger, get_named_client_logger
from autoPyTorch.utils.pipeline import get_dataset_requirements

__all__ = [
    'AbstractEvaluator',
    'EvaluationResults',
    'fit_pipeline'
]


def get_default_budget_type(choice: str = 'default') -> str:
    pipeline_config = get_default_pipeline_config(choice=choice)
    return str(pipeline_config['budget_type'])


def get_default_budget(choice: str = 'default') -> int:
    pipeline_config = get_default_pipeline_config(choice=choice)
    return int(pipeline_config[get_default_budget_type()])


def _get_send_warnings_to_log(logger: PicklableClientLogger) -> Any:
    @no_type_check
    def send_warnings_to_log(message, category, filename, lineno,
                             file=None, line=None) -> None:
        logger.debug(f'{filename}:{lineno}: {category.__name__}:{message}')
        return

    return send_warnings_to_log


def fit_pipeline(logger: PicklableClientLogger, pipeline: BaseEstimator,
                 X: Dict[str, Any], y: Any) -> BaseEstimator:

    send_warnings_to_log = _get_send_warnings_to_log(logger)
    with warnings.catch_warnings():
        warnings.showwarning = send_warnings_to_log
        # X is a fit dictionary and y is usually None for the compatibility
        pipeline.fit(X, y)

    return pipeline


class EvaluationResults(NamedTuple):
    """
    Attributes:
        opt_loss (Dict[str, float]):
            The optimization loss, calculated on the validation set. This will
            be the cost used in SMAC
        train_loss (Dict[str, float]):
            The train loss, calculated on the train set
        opt_pred (np.ndarray):
            The predictions on the validation set. This validation set is created
            from the resampling strategy
        valid_pred (Optional[np.ndarray]):
            Predictions on a user provided validation set
        test_pred (Optional[np.ndarray]):
            Predictions on a user provided test set
        additional_run_info (Optional[Dict]):
            A dictionary with additional run information, like duration or
            the crash error msg, if any.
        status (StatusType):
            The status of the run, following SMAC StatusType syntax.
        pipeline (Optional[BaseEstimator]):
            The fitted pipeline.
    """
    opt_loss: Dict[str, float]
    train_loss: Dict[str, float]
    opt_pred: np.ndarray
    status: StatusType
    pipeline: Optional[BaseEstimator] = None
    valid_pred: Optional[np.ndarray] = None
    test_pred: Optional[np.ndarray] = None
    additional_run_info: Optional[Dict] = None


class FixedPipelineParams(NamedTuple):
    """
    Attributes:
        backend (Backend):
            An object to interface with the disk storage. In particular, allows to
            access the train and test datasets
        metric (autoPyTorchMetric):
            A scorer object that is able to evaluate how good a pipeline was fit. It
            is a wrapper on top of the actual score method (a wrapper on top of scikit
            lean accuracy for example) that formats the predictions accordingly.
        budget_type  (str):
            The budget type, which can be epochs or time
        pipeline_config (Optional[Dict[str, Any]]):
            Defines the content of the pipeline being evaluated. For example, it
            contains pipeline specific settings like logging name, or whether or not
            to use tensorboard.
        seed (int):
            A integer that allows for reproducibility of results
        save_y_opt (bool):
            Whether this worker should output the target predictions, so that they are
            stored on disk. Fundamentally, the resampling strategy might shuffle the
            Y_train targets, so we store the split in order to re-use them for ensemble
            selection.
        include (Optional[Dict[str, Any]]):
            An optional dictionary to include components of the pipeline steps.
        exclude (Optional[Dict[str, Any]]):
            An optional dictionary to exclude components of the pipeline steps.
        disable_file_output (Optional[List[Union[str, DisableFileOutputParameters]]]):
            Used as a list to pass more fine-grained
            information on what to save. Must be a member of `DisableFileOutputParameters`.
            Allowed elements in the list are:

            + `y_opt`:
                do not save the predictions for the optimization set,
                which would later on be used to build an ensemble. Note that SMAC
                optimizes a metric evaluated on the optimization set.
            + `model`:
                do not save any individual pipeline files
            + `cv_model`:
                In case of cross validation, disables saving the joint model of the
                pipelines fit on each fold.
            + `y_test`:
                do not save the predictions for the test set.
            + `all`:
                do not save any of the above.
            For more information check `autoPyTorch.evaluation.utils.DisableFileOutputParameters`.
        logger_port (Optional[int]):
            Logging is performed using a socket-server scheme to be robust against many
            parallel entities that want to write to the same file. This integer states the
            socket port for the communication channel. If None is provided, a traditional
            logger is used.
        all_supported_metrics (bool):
            Whether all supported metric should be calculated for every configuration.
        search_space_updates (Optional[HyperparameterSearchSpaceUpdates]):
            An object used to fine tune the hyperparameter search space of the pipeline
    """
    def __init__(self, backend: Backend,
                 queue: Queue,
                 metric: autoPyTorchMetric,
                 budget: float,
                 configuration: Union[int, str, Configuration],
                 budget_type: str = None,
                 pipeline_config: Optional[Dict[str, Any]] = None,
                 seed: int = 1,
                 output_y_hat_optimization: bool = True,
                 num_run: Optional[int] = None,
                 include: Optional[Dict[str, Any]] = None,
                 exclude: Optional[Dict[str, Any]] = None,
                 disable_file_output: Optional[List[Union[str, DisableFileOutputParameters]]] = None,
                 init_params: Optional[Dict[str, Any]] = None,
                 logger_port: Optional[int] = None,
                 all_supported_metrics: bool = True,
                 search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
                 ) -> None:

        self.starttime = time.time()

        self.configuration = configuration
        self.backend: Backend = backend
        self.queue = queue

        self.include = include
        self.exclude = exclude
        self.search_space_updates = search_space_updates

        self.metric = metric


        self._init_datamanager_info()

        # Flag to save target for ensemble
        self.output_y_hat_optimization = output_y_hat_optimization

    An evaluator is an object that:
        + constructs a pipeline (i.e. a classification or regression estimator) for a given
          pipeline_config and run settings (budget, seed)
        + Fits and trains this pipeline (TrainEvaluator) or tests a given
          configuration (TestEvaluator)

    The provided configuration determines the type of pipeline created. For more
    details, please read the get_pipeline() method.

    Args:
        queue (Queue):
            Each worker available will instantiate an evaluator, and after completion,
            it will append the result to a multiprocessing queue
        fixed_pipeline_params (FixedPipelineParams):
            Fixed parameters for a pipeline.
        evaluator_params (EvaluatorParams):
            The parameters for an evaluator.
    """
    def __init__(self, queue: Queue, fixed_pipeline_params: FixedPipelineParams, evaluator_params: EvaluatorParams):
        self.y_opt: Optional[np.ndarray] = None
        self.starttime = time.time()
        self.queue = queue
        self.fixed_pipeline_params = fixed_pipeline_params
        self.evaluator_params = evaluator_params
        self._init_miscellaneous()
        self.logger.debug(f"Fit dictionary in Abstract evaluator: {dict_repr(self.fit_dictionary)}")
        self.logger.debug(f"Search space updates : {self.fixed_pipeline_params.search_space_updates}")

    def _init_miscellaneous(self) -> None:
        num_run = self.evaluator_params.num_run
        self.num_run = 0 if num_run is None else num_run
        self._init_dataset_properties()
        self._init_additional_metrics()
        self._init_fit_dictionary()

        disable_file_output = self.fixed_pipeline_params.disable_file_output
        if disable_file_output is not None:
            DisableFileOutputParameters.check_compatibility(disable_file_output)
            self.disable_file_output = disable_file_output
        else:
            if isinstance(self.configuration, int):
                self.pipeline_class = DummyClassificationPipeline
            elif isinstance(self.configuration, str):
                if self.task_type in TABULAR_TASKS:
                    self.pipeline_class = MyTraditionalTabularClassificationPipeline
                else:
                    raise ValueError("Only tabular tasks are currently supported with traditional methods")
            elif isinstance(self.configuration, Configuration):
                if self.task_type in TABULAR_TASKS:
                    self.pipeline_class = autoPyTorch.pipeline.tabular_classification.TabularClassificationPipeline
                elif self.task_type in IMAGE_TASKS:
                    self.pipeline_class = autoPyTorch.pipeline.image_classification.ImageClassificationPipeline
                else:
                    raise ValueError('task {} not available'.format(self.task_type))
            self.predict_function = self._predict_proba

        self.X_train, self.y_train = datamanager.train_tensors
        self.unique_train_labels = [
            list(np.unique(self.y_train[train_indices])) for train_indices, _ in self.splits
        ]
        self.X_valid, self.y_valid, self.X_test, self.y_test = None, None, None, None
        if datamanager.val_tensors is not None:
            self.X_valid, self.y_valid = datamanager.val_tensors

        if datamanager.test_tensors is not None:
            self.X_test, self.y_test = datamanager.test_tensors

    def _init_additional_metrics(self) -> None:
        all_supported_metrics = self.fixed_pipeline_params.all_supported_metrics
        metric = self.fixed_pipeline_params.metric
        self.additional_metrics: Optional[List[autoPyTorchMetric]] = None
        self.metrics_dict: Optional[Dict[str, List[str]]] = None

        if all_supported_metrics:
            self.additional_metrics = get_metrics(dataset_properties=self.dataset_properties,
                                                  all_supported_metrics=all_supported_metrics)
            self.metrics_dict = {'additional_metrics': [m.name for m in [metric] + self.additional_metrics]}

    def _init_datamanager_info(
        self,
    ) -> None:
        """
        Initialises instance attributes that come from the datamanager.
        For example,
            X_train, y_train, etc.
        """

        datamanager: BaseDataset = self.backend.load_datamanager()

        assert datamanager.task_type is not None, \
            "Expected dataset {} to have task_type got None".format(datamanager.__class__.__name__)
        self.task_type = STRING_TO_TASK_TYPES[datamanager.task_type]
        self.output_type = STRING_TO_OUTPUT_TYPES[datamanager.output_type]
        self.issparse = datamanager.issparse

        self.X_train, self.y_train = datamanager.train_tensors

        if datamanager.val_tensors is not None:
            self.X_valid, self.y_valid = datamanager.val_tensors
        else:
            self.X_valid, self.y_valid = None, None

        if datamanager.test_tensors is not None:
            self.X_test, self.y_test = datamanager.test_tensors
        else:
            self.X_test, self.y_test = None, None

        self.resampling_strategy = datamanager.resampling_strategy

        self.num_classes: Optional[int] = getattr(datamanager, "num_classes", None)

        self.dataset_properties = datamanager.get_dataset_properties(
            get_dataset_requirements(info=datamanager.get_required_dataset_info(),
                                     include=self.include,
                                     exclude=self.exclude,
                                     search_space_updates=self.search_space_updates
                                     ))
        self.splits = datamanager.splits
        if self.splits is None:
            raise AttributeError(f"create_splits on {datamanager.__class__.__name__} must be called "
                                 f"before the instantiation of {self.__class__.__name__}")

        # delete datamanager from memory
        del datamanager

    def _init_fit_dictionary(
        self,
        logger_port: int,
        pipeline_config: Dict[str, Any],
        metrics_dict: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        Initialises the fit dictionary

        Args:
            logger_port (int):
                Logging is performed using a socket-server scheme to be robust against many
                parallel entities that want to write to the same file. This integer states the
                socket port for the communication channel.
            pipeline_config (Dict[str, Any]):
                Defines the content of the pipeline being evaluated. For example, it
                contains pipeline specific settings like logging name, or whether or not
                to use tensorboard.
            metrics_dict (Optional[Dict[str, List[str]]]):
            Contains a list of metric names to be evaluated in Trainer with key `additional_metrics`. Defaults to None.

        Returns:
            None
        """
        logger_name = f"{self.__class__.__name__.split('.')[-1]}({self.fixed_pipeline_params.seed})"
        logger_port = self.fixed_pipeline_params.logger_port
        logger_port = logger_port if logger_port is not None else logging.handlers.DEFAULT_TCP_LOGGING_PORT
        self.logger = get_named_client_logger(name=logger_name, port=logger_port)

        self.fit_dictionary: Dict[str, Any] = dict(
            dataset_properties=self.dataset_properties,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            backend=self.fixed_pipeline_params.backend,
            logger_port=logger_port,
            optimize_metric=self.fixed_pipeline_params.metric.name,
            **((lambda: {} if self.metrics_dict is None else self.metrics_dict)())
        )
        self.fit_dictionary.update(**self.fixed_pipeline_params.pipeline_config)

        budget, budget_type = self.evaluator_params.budget, self.fixed_pipeline_params.budget_type
        # If the budget is epochs, we want to limit that in the fit dictionary
        if budget_type == 'epochs':
            self.fit_dictionary['epochs'] = budget
            self.fit_dictionary.pop('runtime', None)
        elif budget_type == 'runtime':
            self.fit_dictionary['runtime'] = budget
            self.fit_dictionary.pop('epochs', None)
        else:
            raise ValueError(f"budget type must be `epochs` or `runtime`, but got {budget_type}")

    def predict(
        self,
        X: Optional[np.ndarray],
        pipeline: BaseEstimator,
        unique_train_labels: Optional[List[int]] = None
    ) -> Optional[np.ndarray]:
        """
        A wrapper function to handle the prediction of regression or classification tasks.

        Args:
            X (np.ndarray):
                A set of features to feed to the pipeline
            pipeline (BaseEstimator):
                A model that will take the features X return a prediction y
            unique_train_labels (Optional[List[int]]):
                The unique labels included in the train split.

        Returns:
            (np.ndarray):
                The predictions of pipeline for the given features X
        """

        if X is None:
            return None

        send_warnings_to_log = _get_send_warnings_to_log(self.logger)
        with warnings.catch_warnings():
            warnings.showwarning = send_warnings_to_log
            if self.task_type in REGRESSION_TASKS:
                # To comply with scikit-learn VotingRegressor requirement, if the estimator
                # predicts a (N,) shaped array, it is converted to (N, 1)
                pred = pipeline.predict(X, batch_size=1000)
                pred = pred.reshape((-1, 1)) if len(pred.shape) == 1 else pred
            else:
                pred = pipeline.predict_proba(X, batch_size=1000)
                pred = ensure_prediction_array_sizes(
                    prediction=pred,
                    num_classes=self.num_classes,
                    output_type=self.output_type,
                    unique_train_labels=unique_train_labels
                )

        return pred

    def _get_pipeline(self) -> BaseEstimator:
        """
        Implements a pipeline object based on the self.configuration attribute.
        int: A dummy classifier/dummy regressor is created. This estimator serves
             as a baseline model to ignore all models that perform worst than this
             fixed estimator. Also, in the worst case scenario, this is the final
             estimator created (for instance, in case not enough memory was allocated).
        str: A pipeline with traditional classifiers like random forest, SVM, etc is created,
             as the configuration will contain an estimator name defining the configuration
             to use, for example 'RandomForest'
        Configuration: A pipeline object matching this configuration is created. This
             is the case of neural architecture search, where different backbones
             and head can be passed in the form of a configuration object.

        Returns
            pipeline (BaseEstimator):
                A scikit-learn compliant pipeline which is not yet fit to the data.
        """
        config = self.evaluator_params.configuration
        if not isinstance(config, (int, str, Configuration)):
            raise TypeError("The type of configuration must be either (int, str, Configuration), "
                            f"but got type {type(config)}")

        kwargs = dict(
            config=config,
            random_state=np.random.RandomState(self.fixed_pipeline_params.seed),
            init_params=self.evaluator_params.init_params
        )
        pipeline_class = get_pipeline_class(config=config, task_type=self.task_type)

        if isinstance(config, int):
            return pipeline_class(**kwargs)
        elif isinstance(config, str):
            return pipeline_class(dataset_properties=self.dataset_properties, **kwargs)
        elif isinstance(config, Configuration):
            return pipeline_class(dataset_properties=self.dataset_properties,
                                  include=self.fixed_pipeline_params.include,
                                  exclude=self.fixed_pipeline_params.exclude,
                                  search_space_updates=self.fixed_pipeline_params.search_space_updates,
                                  **kwargs)

    def _loss(self, labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
        """SMAC follows a minimization goal, so the make_scorer
        sign is used as a guide to obtain the value to reduce.
        The calculate_loss internally translate a score function to
        a minimization problem

        Args:
            labels (np.ndarray):
                The expect labels given by the original dataset
            preds (np.ndarray):
                The prediction of the current pipeline being fit
        Returns:
            (Dict[str, float]):
                A dictionary with metric_name -> metric_loss, for every
                supported metric
        """

        metric = self.fixed_pipeline_params.metric
        if isinstance(self.evaluator_params.configuration, int):
            # We do not calculate performance of the dummy configurations
            return {metric.name: metric._optimum - metric._sign * metric._worst_possible_result}

        metrics = self.additional_metrics if self.additional_metrics is not None else [metric]

        return calculate_loss(target=labels, prediction=preds, task_type=self.task_type, metrics=metrics)

    def record_evaluation(self, results: EvaluationResults) -> None:
        """This function does everything necessary after the fitting:
        1. Evaluate of loss for each metric
        2. Save the files for the ensembles_statistics
        3. Add evaluations to queue for SMAC
        We use it as the signal handler so we can recycle the code for the
        normal usecase and when the runsolver kills us here :)

        Args:
            results (EvaluationResults):
                The results from fitting a pipeline.
        """

        opt_pred, valid_pred, test_pred = results.opt_pred, results.valid_pred, results.test_pred

        if not self._save_to_backend(opt_pred, valid_pred, test_pred):
            # If we CANNOT save, nothing to pass to SMAC thus early-return
            return

        cost = results.opt_loss[self.fixed_pipeline_params.metric.name]
        additional_run_info = {} if results.additional_run_info is None else results.additional_run_info
        update_dict = dict(
            train_loss=results.train_loss,
            validation_loss=self._get_transformed_metrics(pred=valid_pred, inference_name='valid'),
            test_loss=self._get_transformed_metrics(pred=test_pred, inference_name='test'),
            opt_loss=results.opt_loss,
            duration=time.time() - self.starttime,
            num_run=self.num_run
        )
        additional_run_info.update({k: v for k, v in update_dict.items() if v is not None})

        rval_dict = {'loss': cost, 'additional_run_info': additional_run_info, 'status': results.status}
        self.queue.put(rval_dict)

    def _get_transformed_metrics(self, pred: Optional[np.ndarray], inference_name: str) -> Optional[Dict[str, float]]:
        """
        A helper function to calculate the performance estimate of the
        current pipeline in the user provided validation/test set.

        Args:
            pred (Optional[np.ndarray]):
                predictions on a validation set provided by the user,
                matching self.y_{valid or test}
            inference_name (str):
                Which inference duration either `valid` or `test`

        Returns:
            loss_dict (Optional[Dict[str, float]]):
                Various losses available on the dataset for the specified duration.
        """
        duration_choices = ('valid', 'test')
        if inference_name not in duration_choices:
            raise ValueError(f'inference_name must be in {duration_choices}, but got {inference_name}')

        labels = getattr(self, f'y_{inference_name}', None)
        return None if pred is None or labels is None else self._loss(labels, pred)

    def _get_prediction(self, pred: Optional[np.ndarray], name: str) -> Optional[np.ndarray]:
        return pred if name not in self.disable_file_output else None

    def _fetch_voting_pipeline(self) -> Optional[Union[VotingClassifier, VotingRegressorWrapper]]:
        pipelines = [pl for pl in self.pipelines if pl is not None]
        if len(pipelines) == 0:
            return None

        if self.task_type in CLASSIFICATION_TASKS:
            voting_pipeline = VotingClassifier(estimators=None, voting='soft')
        else:
            voting_pipeline = VotingRegressorWrapper(estimators=None)

        voting_pipeline.estimators_ = self.pipelines

        return voting_pipeline

    def _save_to_backend(
        self,
        opt_pred: np.ndarray,
        valid_pred: Optional[np.ndarray],
        test_pred: Optional[np.ndarray]
    ) -> bool:
        """ Return False if we CANNOT save due to some issues """
        if not self._is_output_possible(opt_pred, valid_pred, test_pred):
            return False
        if self.y_opt is None or 'all' in self.disable_file_output:
            # self.y_opt can be None if we use partial-cv ==> no output to save
            return True

        backend = self.fixed_pipeline_params.backend
        # This file can be written independently of the others down bellow
        if 'y_opt' not in self.disable_file_output and self.fixed_pipeline_params.save_y_opt:
            backend.save_targets_ensemble(self.y_opt)

        seed, budget = self.fixed_pipeline_params.seed, self.evaluator_params.budget
        self.logger.debug(f"Saving directory {seed}, {self.num_run}, {budget}")
        backend.save_numrun_to_dir(
            seed=int(seed),
            idx=int(self.num_run),
            budget=float(budget),
            model=self.pipelines[0] if 'model' not in self.disable_file_output else None,
            cv_model=self._fetch_voting_pipeline() if 'cv_model' not in self.disable_file_output else None,
            ensemble_predictions=self._get_prediction(opt_pred, 'y_opt'),
            valid_predictions=self._get_prediction(valid_pred, 'y_valid'),
            test_predictions=self._get_prediction(test_pred, 'y_test')
        )
        return True

    def _is_output_possible(
        self,
        opt_pred: np.ndarray,
        valid_pred: Optional[np.ndarray],
        test_pred: Optional[np.ndarray]
    ) -> bool:

        if self.y_opt is None:  # mypy check
            return True

        if self.y_opt.shape[0] != opt_pred.shape[0]:
            return False

        y_dict = {'optimization': opt_pred, 'validation': valid_pred, 'test': test_pred}
        for y in y_dict.values():
            if y is not None and not np.all(np.isfinite(y)):
                return False  # Model predictions contains NaNs

        Args:
            prediction (np.ndarray):
                The un-formatted predictions of a pipeline
            Y_train (np.ndarray):
                The labels from the dataset to give an intuition of the expected
                predictions dimensionality
        Returns:
            (np.ndarray):
                The formatted prediction
        """
        assert self.num_classes is not None, "Called function on wrong task"

        if self.output_type == MULTICLASS and \
                prediction.shape[1] < self.num_classes:
            if Y_train is None:
                raise ValueError('Y_train must not be None!')
            classes = list(np.unique(Y_train))

            mapping = dict()
            for class_number in range(self.num_classes):
                if class_number in classes:
                    index = classes.index(class_number)
                    mapping[index] = class_number
            new_predictions = np.zeros((prediction.shape[0], self.num_classes),
                                       dtype=np.float32)

            for index in mapping:
                class_index = mapping[index]
                new_predictions[:, class_index] = prediction[:, index]

            return new_predictions

        return prediction
