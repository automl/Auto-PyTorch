import os
import uuid
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np

import pandas as pd

from autoPyTorch.api.base_task import BaseTask
from autoPyTorch.constants import TASK_TYPES_TO_STRING, TIMESERIES_FORECASTING
from autoPyTorch.data.time_series_forecasting_validator import TimeSeriesForecastingInputValidator
from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
    HoldoutValTypes,
)
from autoPyTorch.datasets.time_series_dataset import TimeSeriesForecastingDataset
from autoPyTorch.pipeline.time_series_forecasting import TimeSeriesForecastingPipeline
from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from autoPyTorch.constants_forecasting import MAX_WINDOW_SIZE_BASE


class TimeSeriesForecastingTask(BaseTask):
    """
    Time Series Forecasting API to the pipelines.
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
        self.time_series_forecasting = True

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
            budget_type: str = 'epochs',
            min_budget: Union[int, str] = 5,
            max_budget: Union[int, str] = 50,
            total_walltime_limit: int = 100,
            func_eval_time_limit_secs: Optional[int] = None,
            enable_traditional_pipeline: bool = False,
            memory_limit: Optional[int] = 4096,
            smac_scenario_args: Optional[Dict[str, Any]] = None,
            get_smac_object_callback: Optional[Callable] = None,
            all_supported_metrics: bool = True,
            precision: int = 32,
            disable_file_output: List = [],
            load_models: bool = True,
            portfolio_selection: Optional[str] = None,
            normalize_y: bool = True,
            suggested_init_models: Optional[List[str]] = None,
            custom_init_setting_path: Optional[str] = None,
            min_num_test_instances: Optional[int] = None,
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
            budget_type (str):
                Type of budget to be used when fitting the pipeline.
                It can be one of:

                + `epochs`: The training of each pipeline will be terminated after
                    a number of epochs have passed. This number of epochs is determined by the
                    budget argument of this method.
                + `runtime`: The training of each pipeline will be terminated after
                    a number of seconds have passed. This number of seconds is determined by the
                    budget argument of this method. The overall fitting time of a pipeline is
                    controlled by func_eval_time_limit_secs. 'runtime' only controls the allocated
                    time to train a pipeline, but it does not consider the overall time it takes
                    to create a pipeline (data loading and preprocessing, other i/o operations, etc.).
                    budget_type will determine the units of min_budget/max_budget. If budget_type=='epochs'
                    is used, min_budget will refer to epochs whereas if budget_type=='runtime' then
                    min_budget will refer to seconds.
                + 'resolution': The sample resolution of time series, for instance, if a time series sequence is
                [0, 1, 2, 3, 4] with resolution 0.5, the sequence fed to the network is [0, 2, 4]
            min_budget Union[int, str]:
                Auto-PyTorch uses `Hyperband <https://arxiv.org/abs/1603.06560>`_ to
                trade-off resources between running many pipelines at min_budget and
                running the top performing pipelines on max_budget.
                min_budget states the minimum resource allocation a pipeline should have
                so that we can compare and quickly discard bad performing models.
                For example, if the budget_type is epochs, and min_budget=5, then we will
                run every pipeline to a minimum of 5 epochs before performance comparison.
            max_budget Union[int, str]:
                Auto-PyTorch uses `Hyperband <https://arxiv.org/abs/1603.06560>`_ to
                trade-off resources between running many pipelines at min_budget and
                running the top performing pipelines on max_budget.
                max_budget states the maximum resource allocation a pipeline is going to
                be ran. For example, if the budget_type is epochs, and max_budget=50,
                then the pipeline training will be terminated after 50 epochs.

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
            normalize_y: bool
                if the input y values need to be normalized
            suggested_init_models: Optional[List[str]]
                suggested initial models with their default configurations setting
            custom_init_setting_path: Optional[str]
                path to a json file that contains the initial configuration suggested by the users
            min_num_test_instances: Optional[int]
                if it is set None, then full validation sets will be evaluated in each fidelity. Otherwise, the number
                of instances in the test sets should be a value that is at least as great as this value, otherwise, the
                number of test instance is proportional to its fidelity
        Returns:
            self

        """
        if dataset_name is None:
            dataset_name = str(uuid.uuid1(clock_seq=os.getpid()))

        # we have to create a logger for at this point for the validator
        self._logger = self._get_logger(dataset_name)
        # TODO we will only consider target variables as int here
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
            normalize_y=normalize_y,
        )

        if self.dataset.base_window_size is not None or not self.customized_window_size:
            base_window_size = int(np.ceil(self.dataset.base_window_size))
            # we don't want base window size to large, which might cause a too long computation time, in which case
            # we will use n_prediction_step instead (which is normally smaller than base_window_size)
            if base_window_size > MAX_WINDOW_SIZE_BASE:
                # TODO considering padding to allow larger upper_window_size !!!
                if n_prediction_steps > MAX_WINDOW_SIZE_BASE:
                    base_window_size = 50
                else:
                    base_window_size = n_prediction_steps

            if self.search_space_updates is None:
                self.search_space_updates = HyperparameterSearchSpaceUpdates()

            window_size_scales = [1, 3]

            self.search_space_updates.append(node_name="data_loader",
                                             hyperparameter="window_size",
                                             value_range=[int(window_size_scales[0] * base_window_size),
                                                          int(window_size_scales[1] * base_window_size)],
                                             default_value=int(np.ceil(1.25 * base_window_size)),
                                             )

        self._metrics_kwargs = {'sp': self.dataset.seasonality,
                                'n_prediction_steps': n_prediction_steps}

        forecasting_kwargs = dict(suggested_init_models=suggested_init_models,
                                  custom_init_setting_path=custom_init_setting_path,
                                  min_num_test_instances=min_num_test_instances)

        return self._search(
            dataset=self.dataset,
            optimize_metric=optimize_metric,
            budget_type=budget_type,
            min_budget=min_budget,
            max_budget=max_budget,
            total_walltime_limit=total_walltime_limit,
            func_eval_time_limit_secs=func_eval_time_limit_secs,
            enable_traditional_pipeline=enable_traditional_pipeline,
            memory_limit=memory_limit,
            smac_scenario_args=smac_scenario_args,
            get_smac_object_callback=get_smac_object_callback,
            all_supported_metrics=all_supported_metrics,
            precision=precision,
            disable_file_output=disable_file_output,
            load_models=load_models,
            portfolio_selection=portfolio_selection,
            time_series_forecasting=self.time_series_forecasting,
            **forecasting_kwargs,
        )

    def predict(
            self,
            X_test: Optional[Union[Union[List[np.ndarray]], pd.DataFrame, Dict]]=None,
            batch_size: Optional[int] = None,
            n_jobs: int = 1,
            past_targets: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
                    target_variables: Optional[Union[Tuple[int], Tuple[str], np.ndarray]] = None,
                (used for multi-variable prediction), indicates which value needs to be predicted
        """
        if not self.dataset.is_uni_variant:
            if past_targets is None:
                if not isinstance(X_test, Dict) or "past_targets" not in X_test:
                    raise ValueError("Past Targets must be given")
            else:
                X_test = {"features": X_test,
                          "past_targets": past_targets}
        flattened_res = super(TimeSeriesForecastingTask, self).predict(X_test, batch_size, n_jobs)
        if self.dataset.num_target == 1:
            return flattened_res.reshape([-1, self.dataset.n_prediction_steps])
        return flattened_res.reshape([-1, self.dataset.n_prediction_steps, self.dataset.num_target])
