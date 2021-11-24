import os
import uuid
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np

import pandas as pd

from autoPyTorch.api.base_task import BaseTask
from autoPyTorch.constants import (
    TASK_TYPES_TO_STRING,
    TIMESERIES_FORECASTING,
)
from autoPyTorch.data.time_series_forecasting_validator import TimeSeriesForecastingInputValidator
from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
    HoldoutValTypes,
)
from autoPyTorch.datasets.time_series_dataset import TimeSeriesForecastingDataset
from autoPyTorch.pipeline.time_series_forecasting import TimeSeriesForecastingPipeline
from autoPyTorch.utils.backend import Backend
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from autoPyTorch.constants_forecasting import MAX_WINDOW_SIZE_BASE, SEASONALITY_MAP


class TimeSeriesForecastingTask(BaseTask):
    """
    Time Series Forcasting API to the pipelines.
    Args:
        seed (int): seed to be used for reproducibility.
        n_jobs (int), (default=1): number of consecutive processes to spawn.
        logging_config (Optional[Dict]): specifies configuration
            for logging, if None, it is loaded from the logging.yaml
        ensemble_size (int), (default=50): Number of models added to the ensemble built by
            Ensemble selection from libraries of models.
            Models are drawn with replacement.
        ensemble_nbest (int), (default=50): only consider the ensemble_nbest
            models to build the ensemble
        max_models_on_disc (int), (default=50): maximum number of models saved to disc.
            Also, controls the size of the ensemble as any additional models will be deleted.
            Must be greater than or equal to 1.
        temporary_directory (str): folder to store configuration output and log file
        output_directory (str): folder to store predictions for optional test set
        delete_tmp_folder_after_terminate (bool): determines whether to delete the temporary directory,
            when finished
        include_components (Optional[Dict]): If None, all possible components are used.
            Otherwise specifies set of components to use.
        exclude_components (Optional[Dict]): If None, all possible components are used.
            Otherwise specifies set of components not to use. Incompatible with include
            components
    """

    def __init__(
            self,
            seed: int = 1,
            n_jobs: int = 1,
            logging_config: Optional[Dict] = None,
            ensemble_size: int = 50,
            ensemble_nbest: int = 50,
            max_models_on_disc: int = 50,
            temporary_directory: Optional[str] = None,
            output_directory: Optional[str] = None,
            delete_tmp_folder_after_terminate: bool = True,
            delete_output_folder_after_terminate: bool = True,
            include_components: Optional[Dict] = None,
            exclude_components: Optional[Dict] = None,
            resampling_strategy: Union[
                CrossValTypes, HoldoutValTypes] = HoldoutValTypes.time_series_hold_out_validation,
            resampling_strategy_args: Optional[Dict[str, Any]] = None,
            backend: Optional[Backend] = None,
            search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None,
    ):
        super().__init__(
            seed=seed,
            n_jobs=n_jobs,
            logging_config=logging_config,
            ensemble_size=ensemble_size,
            ensemble_nbest=ensemble_nbest,
            max_models_on_disc=max_models_on_disc,
            temporary_directory=temporary_directory,
            output_directory=output_directory,
            delete_tmp_folder_after_terminate=delete_tmp_folder_after_terminate,
            delete_output_folder_after_terminate=delete_output_folder_after_terminate,
            include_components=include_components,
            exclude_components=exclude_components,
            backend=backend,
            resampling_strategy=resampling_strategy,
            resampling_strategy_args=resampling_strategy_args,
            search_space_updates=search_space_updates,
            task_type=TASK_TYPES_TO_STRING[TIMESERIES_FORECASTING],
        )
        # here fraction of subset could be number of images, tabular data or resolution of time-series datasets.
        # TODO if budget type resolution is applied to all datasets, we will put it to configs
        self.pipeline_options.update({"min_resolution": 0.1,
                                      "full_resolution": 1.0})

        self.customized_window_size = False
        if self.search_space_updates is not None:
            for update in self.search_space_updates.updates:
                # user has already specified a window_size range
                if update.node_name == 'data_loader' and update.hyperparameter == 'window_size':
                    self.customized_window_size = True
        self.time_series_prediction = True

    def _get_required_dataset_properties(self, dataset: BaseDataset) -> Dict[str, Any]:
        if not isinstance(dataset, TimeSeriesForecastingDataset):
            raise ValueError("Dataset is incompatible for the given task,: {}".format(
                type(dataset)
            ))
        return dataset.get_required_dataset_info()

    def build_pipeline(self, dataset_properties: Dict[str, Any]) -> TimeSeriesForecastingPipeline:
        return TimeSeriesForecastingPipeline(dataset_properties=dataset_properties)

    def search(
            self,
            optimize_metric: str,
            X_train: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
            y_train: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
            X_test: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
            y_test: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
            target_variables: Optional[Union[Tuple[int], Tuple[str], np.ndarray]] = None,
            n_prediction_steps: int = 1,
            freq: Optional[Union[str, int, List[int]]] = None,
            dataset_name: Optional[str] = None,
            budget_type: Optional[str] = None,
            budget: Optional[float] = None,
            total_walltime_limit: int = 100,
            func_eval_time_limit: int = 60,
            traditional_per_total_budget: float = 0.,
            memory_limit: Optional[int] = 4096,
            smac_scenario_args: Optional[Dict[str, Any]] = None,
            get_smac_object_callback: Optional[Callable] = None,
            all_supported_metrics: bool = True,
            precision: int = 32,
            disable_file_output: List = [],
            load_models: bool = True,
            shift_input_data: bool = True,
            normalize_y: bool = True,
    ) -> 'BaseTask':
        """
        Search for the best pipeline configuration for the given dataset.

        Fit both optimizes the machine learning models and builds an ensemble out of them.
        To disable ensembling, set ensemble_size==0.
        using the optimizer.
        Args:
            X_train, y_train, X_test, y_test: Union[np.ndarray, List, pd.DataFrame]
                A pair of features (X_train) and targets (y_train) used to fit a
                pipeline. Additionally, a holdout of this pairs (X_test, y_test) can
                be provided to track the generalization performance of each stage.
            target_variables: Optional[Union[Tuple[int], Tuple[str], np.ndarray]] = None,
                (used for multi-variable prediction), indicates which value needs to be predicted
            n_prediction_steps: int
                How many steps in advance we need to predict
            freq: Optional[Union[str, int, List[int]]]
                frequency information, it determines the configuration space of the window size, if it is not given,
                we will use the default configuration
            dataset_name: Optional[str],
                dataset name
            optimize_metric (str): name of the metric that is used to
                evaluate a pipeline.
            budget_type (Optional[str]):
                Type of budget to be used when fitting the pipeline.
                Either 'epochs' or 'runtime' or 'resolution'. If not provided, uses
                the default in the pipeline config ('epochs')
            budget (Optional[float]):
                Budget to fit a single run of the pipeline. If not
                provided, uses the default in the pipeline config
            total_walltime_limit (int), (default=100): Time limit
                in seconds for the search of appropriate models.
                By increasing this value, autopytorch has a higher
                chance of finding better models.
            func_eval_time_limit (int), (default=60): Time limit
                for a single call to the machine learning model.
                Model fitting will be terminated if the machine
                learning algorithm runs over the time limit. Set
                this value high enough so that typical machine
                learning algorithms can be fit on the training
                data.
            traditional_per_total_budget (float), (default=0.1):
                Percent of total walltime to be allocated for
                running traditional classifiers.
            memory_limit (Optional[int]), (default=4096): Memory
                limit in MB for the machine learning algorithm. autopytorch
                will stop fitting the machine learning algorithm if it tries
                to allocate more than memory_limit MB. If None is provided,
                no memory limit is set. In case of multi-processing, memory_limit
                will be per job. This memory limit also applies to the ensemble
                creation process.
            smac_scenario_args (Optional[Dict]): Additional arguments inserted
                into the scenario of SMAC. See the
                [SMAC documentation] (https://automl.github.io/SMAC3/master/options.html?highlight=scenario#scenario)
            get_smac_object_callback (Optional[Callable]): Callback function
                to create an object of class
                [smac.optimizer.smbo.SMBO](https://automl.github.io/SMAC3/master/apidoc/smac.optimizer.smbo.html).
                The function must accept the arguments scenario_dict,
                instances, num_params, runhistory, seed and ta. This is
                an advanced feature. Use only if you are familiar with
                [SMAC](https://automl.github.io/SMAC3/master/index.html).
            all_supported_metrics (bool), (default=True): if True, all
                metrics supporting current task will be calculated
                for each pipeline and results will be available via cv_results
            precision (int), (default=32): Numeric precision used when loading
                ensemble data. Can be either '16', '32' or '64'.
            disable_file_output (Union[bool, List]):
            load_models (bool), (default=True): Whether to load the
                models after fitting AutoPyTorch.
            shift_input_data: bool
                if the input data needs to be shifted
            normalize_y: bool
                if the input y values need to be normalized
        Returns:
            self

        """
        if dataset_name is None:
            dataset_name = str(uuid.uuid1(clock_seq=os.getpid()))

        # we have to create a logger for at this point for the validator
        self._logger = self._get_logger(dataset_name)
        #TODO we will only consider target variables as int here
        self.target_variables = target_variables
        # Create a validator object to make sure that the data provided by
        # the user matches the autopytorch requirements
        self.InputValidator = TimeSeriesForecastingInputValidator(
            is_classification=False,
            logger_port=self._logger_port,
        )

        # Fit a input validator to check the provided data
        # Also, an encoder is fit to both train and test data,
        # to prevent unseen categories during inference
        self.InputValidator.fit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        self.dataset = TimeSeriesForecastingDataset(
            X=X_train, Y=y_train,
            X_test=X_test, Y_test=y_test,
            freq=freq,
            validator=self.InputValidator,
            resampling_strategy=self.resampling_strategy,
            resampling_strategy_args=self.resampling_strategy_args,
            n_prediction_steps=n_prediction_steps,
            shift_input_data=shift_input_data,
            normalize_y=normalize_y,
        )

        if self.dataset.freq_value is not None or not self.customized_window_size:
            base_window_size = int(np.ceil(self.dataset.freq_value))
            # we don't want base window size to large, which might cause a too long computation time, in which case
            # we will use n_prediction_step instead (which is normally smaller than base_window_size)
            if base_window_size > self.dataset.upper_window_size or base_window_size > MAX_WINDOW_SIZE_BASE:
                # TODO considering padding to allow larger upper_window_size !!!
                base_window_size = int(np.ceil(min(n_prediction_steps, self.dataset.upper_window_size)))
            if base_window_size > MAX_WINDOW_SIZE_BASE:
                base_window_size = 50  # TODO this value comes from setting of solar dataset, do we have a better choice?
            if self.search_space_updates is None:
                self.search_space_updates = HyperparameterSearchSpaceUpdates()

            window_size_scales = [1, 2]

            self.search_space_updates.append(node_name="data_loader",
                                             hyperparameter="window_size",
                                             value_range=[window_size_scales[0] * base_window_size,
                                                          window_size_scales[1] * base_window_size],
                                             default_value=int(np.ceil(1.25 * base_window_size)),
                                             )

        if traditional_per_total_budget > 0.:
            self._logger.warning("Time series Forecasting for now does not support traditional classifiers. "
                                 "Setting traditional_per_total_budget to 0.")
            traditional_per_total_budget = 0.

        seasonality = SEASONALITY_MAP.get(self.dataset.freq, 1)
        if isinstance(seasonality, list):
            seasonality = min(seasonality)  # Use to calculate MASE
        self._metrics_kwargs = {'sp': seasonality,
                                'n_prediction_steps': n_prediction_steps}

        return self._search(
            dataset=self.dataset,
            optimize_metric=optimize_metric,
            budget_type=budget_type,
            budget=budget,
            total_walltime_limit=total_walltime_limit,
            func_eval_time_limit=func_eval_time_limit,
            traditional_per_total_budget=traditional_per_total_budget,
            memory_limit=memory_limit,
            smac_scenario_args=smac_scenario_args,
            get_smac_object_callback=get_smac_object_callback,
            all_supported_metrics=all_supported_metrics,
            precision=precision,
            disable_file_output=disable_file_output,
            load_models=load_models,
            time_series_prediction=self.time_series_prediction
        )

    def predict(
            self,
            X_test: List[np.ndarray],
            batch_size: Optional[int] = None,
            n_jobs: int = 1,
            y_train: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
                    target_variables: Optional[Union[Tuple[int], Tuple[str], np.ndarray]] = None,
                (used for multi-variable prediction), indicates which value needs to be predicted
        """
        y_pred = np.ones([len(X_test), self.dataset.n_prediction_steps])
        for seq_idx, seq in enumerate(X_test):
            if self.dataset.normalize_y:
                if pd.DataFrame(seq).shape[-1] > 1:
                    if self.target_variables is None and y_train is None:
                        raise ValueError(
                            'For multi-variant prediction task, either target_variables or y_train needs to '
                            'be provided!')
                    if y_train is None:
                        y_train = seq[self.target_variables]
                else:
                    y_train = seq
                if self.dataset.shift_input_data:
                    # if input data is shifted, we must compute the mean and standard deviation with the shifted data.
                    # This is helpful when the
                    mean_seq = np.mean(y_train[self.dataset.n_prediction_steps:])
                    std_seq = np.std(y_train[self.dataset.n_prediction_steps:])
                else:
                    mean_seq = np.mean(y_train)
                    std_seq = np.std(y_train)

                seq_pred = super(TimeSeriesForecastingTask, self).predict(seq, batch_size, n_jobs).flatten()

                seq_pred = seq_pred * std_seq + mean_seq
            else:
                seq_pred = super(TimeSeriesForecastingTask, self).predict(seq, batch_size, n_jobs).flatten()
            y_pred[seq_idx] = seq_pred
        return y_pred
