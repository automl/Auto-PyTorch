from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

import pandas as pd

from autoPyTorch.api.base_task import BaseTask
from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.constants import MAX_WINDOW_SIZE_BASE, TASK_TYPES_TO_STRING, TIMESERIES_FORECASTING
from autoPyTorch.data.time_series_forecasting_validator import TimeSeriesForecastingInputValidator
from autoPyTorch.data.utils import (
    DatasetCompressionSpec,
    get_dataset_compression_mapping
)
from autoPyTorch.datasets.base_dataset import (
    BaseDataset,
    BaseDatasetPropertiesType
)
from autoPyTorch.datasets.resampling_strategy import (
    HoldoutValTypes,
    ResamplingStrategies
)
from autoPyTorch.datasets.time_series_dataset import TimeSeriesForecastingDataset, TimeSeriesSequence
from autoPyTorch.pipeline.time_series_forecasting import TimeSeriesForecastingPipeline
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates


class TimeSeriesForecastingTask(BaseTask):
    """
    Time Series Forecasting API to the pipelines.

    Args:
        seed (int):
            seed to be used for reproducibility.
        n_jobs (int), (default=1):
            number of consecutive processes to spawn.
        logging_config (Optional[Dict]):
            specifies configuration for logging, if None, it is loaded from the logging.yaml
        ensemble_size (int), (default=50):
            Number of models added to the ensemble built by Ensemble selection from libraries of models.
            Models are drawn with replacement.
        ensemble_nbest (int), (default=50):
            only consider the ensemble_nbest models to build the ensemble
        max_models_on_disc (int), (default=50):
            maximum number of models saved to disc. Also, controls the size of the ensemble as any additional models
             will be deleted. Must be greater than or equal to 1.
        temporary_directory (str):
            folder to store configuration output and log file
        output_directory (str):
            folder to store predictions for optional test set
        delete_tmp_folder_after_terminate (bool):
            determines whether to delete the temporary directory, when finished
        include_components (Optional[Dict]):
            If None, all possible components are used. Otherwise specifies set of components to use.
        exclude_components (Optional[Dict]):
            If None, all possible components are used. Otherwise specifies set of components not to use.
            Incompatible with include components
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
        resampling_strategy: ResamplingStrategies = HoldoutValTypes.time_series_hold_out_validation,
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

        self.customized_window_size = False
        if self.search_space_updates is not None:
            for update in self.search_space_updates.updates:
                # user has already specified a window_size range
                if (
                    update.node_name == "data_loader"
                    and update.hyperparameter == "window_size"
                ):
                    self.customized_window_size = True

    def _get_required_dataset_properties(self, dataset: BaseDataset) -> Dict[str, Any]:
        if not isinstance(dataset, TimeSeriesForecastingDataset):
            raise ValueError(
                "Dataset is incompatible for the given task,: {}".format(type(dataset))
            )
        return dataset.get_required_dataset_info()

    def build_pipeline(
        self,
        dataset_properties: Dict[str, BaseDatasetPropertiesType],
        include_components: Optional[Dict[str, Any]] = None,
        exclude_components: Optional[Dict[str, Any]] = None,
        search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None,
    ) -> TimeSeriesForecastingPipeline:
        """
        Build pipeline according to current task
        and for the passed dataset properties

        Args:
            dataset_properties (Dict[str, Any]):
                Characteristics of the dataset to guide the pipeline
                choices of components
            include_components (Optional[Dict[str, Any]]):
                Dictionary containing components to include. Key is the node
                name and Value is an Iterable of the names of the components
                to include. Only these components will be present in the
                search space.
            exclude_components (Optional[Dict[str, Any]]):
                Dictionary containing components to exclude. Key is the node
                name and Value is an Iterable of the names of the components
                to exclude. All except these components will be present in
                the search space.
            search_space_updates (Optional[HyperparameterSearchSpaceUpdates]):
                Search space updates that can be used to modify the search
                space of particular components or choice modules of the pipeline

        Returns:
            TimeSeriesForecastingPipeline:

        """
        return TimeSeriesForecastingPipeline(
            dataset_properties=dataset_properties,
            include=include_components,
            exclude=exclude_components,
            search_space_updates=search_space_updates,
        )

    def _get_dataset_input_validator(
        self,
        X_train: Union[List, pd.DataFrame, np.ndarray],
        y_train: Union[List, pd.DataFrame, np.ndarray],
        X_test: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
        resampling_strategy: Optional[ResamplingStrategies] = None,
        resampling_strategy_args: Optional[Dict[str, Any]] = None,
        dataset_name: Optional[str] = None,
        dataset_compression: Optional[DatasetCompressionSpec] = None,
        freq: Optional[Union[str, int, List[int]]] = None,
        start_times: Optional[List[pd.DatetimeIndex]] = None,
        series_idx: Optional[Union[List[Union[str, int]], str, int]] = None,
        n_prediction_steps: int = 1,
        known_future_features: Union[Tuple[Union[int, str]], Tuple[()]] = (),
        **forecasting_dataset_kwargs: Any,
    ) -> Tuple[TimeSeriesForecastingDataset, TimeSeriesForecastingInputValidator]:
        """
        Returns an object of `TimeSeriesForecastingDataset` and an object of
        `TimeSeriesForecastingInputValidator` according to the current task.

        Args:
            X_train (Union[List, pd.DataFrame, np.ndarray]):
                Training feature set.
            y_train (Union[List, pd.DataFrame, np.ndarray]):
                Training target set.
            X_test (Optional[Union[List, pd.DataFrame, np.ndarray]]):
                Testing feature set
            y_test (Optional[Union[List, pd.DataFrame, np.ndarray]]):
                Testing target set
            resampling_strategy (Optional[RESAMPLING_STRATEGIES]):
                Strategy to split the training data. if None, uses
                HoldoutValTypes.holdout_validation.
            resampling_strategy_args (Optional[Dict[str, Any]]):
                arguments required for the chosen resampling strategy. If None, uses
                the default values provided in DEFAULT_RESAMPLING_PARAMETERS
                in ```datasets/resampling_strategy.py```.
            dataset_name (Optional[str]):
                name of the dataset, used as experiment name.
            dataset_compression (Optional[DatasetCompressionSpec]):
                specifications for dataset compression. For more info check
                documentation for `BaseTask.get_dataset`.
            freq (Optional[Union[str, int, List[int]]]):
                frequency information, it determines the configuration space of the window size, if it is not given,
                we will use the default configuration
            start_times (Optional[List[pd.DatetimeIndex]]):
                starting time of each series when they are sampled. If it is not given, we simply start with a fixed
                timestamp.
            series_idx (Optional[Union[List[Union[str, int]], str, int]]):
                (only works if X is stored as pd.DataFrame). This value is applied to identify to which series the data
                belongs if the data is presented as a "chunk" dataframe
            n_prediction_steps (int):
                The number of steps you want to forecast into the future (forecast horizon)
            known_future_features (Optional[Union[Tuple[Union[str, int]], Tuple[()]]]):
                future features that are known in advance. For instance, holidays.
            forecasting_kwargs (Any)
                kwargs for forecasting dataset, for more details, please check
                ```datasets/time_series_dataset.py```
        Returns:
            TimeSeriesForecastingDataset:
                the dataset object.
            TimeSeriesForecastingInputValidator:
                the input validator fitted on the data.
        """

        resampling_strategy = (
            resampling_strategy
            if resampling_strategy is not None
            else self.resampling_strategy
        )
        resampling_strategy_args = (
            resampling_strategy_args
            if resampling_strategy_args is not None
            else self.resampling_strategy_args
        )

        # Create a validator object to make sure that the data provided by
        # the user matches the autopytorch requirements
        input_validator = TimeSeriesForecastingInputValidator(
            is_classification=False,
            logger_port=self._logger_port,
            dataset_compression=dataset_compression,
        )

        # Fit an input validator to check the provided data
        # Also, an encoder is fit to both train and test data,
        # to prevent unseen categories during inference
        input_validator.fit(
            X_train=X_train,
            y_train=y_train,
            start_times=start_times,
            series_idx=series_idx,
            X_test=X_test,
            y_test=y_test,
        )

        dataset = TimeSeriesForecastingDataset(
            X=X_train,
            Y=y_train,
            X_test=X_test,
            Y_test=y_test,
            dataset_name=dataset_name,
            freq=freq,
            start_times=start_times,
            series_idx=series_idx,
            validator=input_validator,
            resampling_strategy=resampling_strategy,
            resampling_strategy_args=resampling_strategy_args,
            n_prediction_steps=n_prediction_steps,
            known_future_features=known_future_features,
            **forecasting_dataset_kwargs,
        )

        return dataset, input_validator

    def search(
        self,
        optimize_metric: str,
        X_train: Optional[Union[List, pd.DataFrame]] = None,
        y_train: Optional[Union[List, pd.DataFrame]] = None,
        X_test: Optional[Union[List, pd.DataFrame]] = None,
        y_test: Optional[Union[List, pd.DataFrame]] = None,
        n_prediction_steps: int = 1,
        freq: Optional[Union[str, int, List[int]]] = None,
        start_times: Optional[List[pd.DatetimeIndex]] = None,
        series_idx: Optional[Union[List[Union[str, int]], str, int]] = None,
        dataset_name: Optional[str] = None,
        budget_type: str = "epochs",
        min_budget: Union[int, float] = 5,
        max_budget: Union[int, float] = 50,
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
        suggested_init_models: Optional[List[str]] = None,
        custom_init_setting_path: Optional[str] = None,
        min_num_test_instances: Optional[int] = None,
        dataset_compression: Union[Mapping[str, Any], bool] = False,
        **forecasting_dataset_kwargs: Any,
    ) -> "BaseTask":
        """
        Search for the best pipeline configuration for the given dataset.

        Fit both optimizes the machine learning models and builds an ensemble out of them.
        To disable ensembling, set ensemble_size==0.
        using the optimizer.

        Args:
            optimize_metric (str):
                name of the metric that is used to evaluate a pipeline.
            X_train: Optional[Union[List, pd.DataFrame]]
                A pair of features (X_train) and targets (y_train) used to fit a
                pipeline. Additionally, a holdout of this pairs (X_test, y_test) can
                be provided to track the generalization performance of each stage.
            y_train: Union[List, pd.DataFrame]
                training target, must be given
            X_test: Optional[Union[List, pd.DataFrame]]
                Test Features, Test series need to end at one step before forecasting
            y_test: Optional[Union[List, pd.DataFrame]]
                Test Targets
            n_prediction_steps: int
                How many steps in advance we need to predict
            freq: Optional[Union[str, int, List[int]]]
                frequency information, it determines the configuration space of the window size, if it is not given,
                we will use the default configuration
            start_times: : List[pd.DatetimeIndex]
                A list indicating the start time of each series in the training sets
            series_idx: Optional[Union[List[Union[str, int]], str, int]]
                variable in X indicating series indices
            dataset_name: Optional[str],
                dataset name
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
            min_budget Union[int, float]:
                Auto-PyTorch uses `Hyperband <https://arxiv.org/abs/1603.06560>`_ to
                trade-off resources between running many pipelines at min_budget and
                running the top performing pipelines on max_budget.
                min_budget states the minimum resource allocation a pipeline should have
                so that we can compare and quickly discard bad performing models.
                For example, if the budget_type is epochs, and min_budget=5, then we will
                run every pipeline to a minimum of 5 epochs before performance comparison.
            max_budget Union[int, float]:
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
            disable_file_output (Optional[List[Union[str, DisableFileOutputParameters]]]):
                Used as a list to pass more fine-grained
                information on what to save. Must be a member of `DisableFileOutputParameters`.
                Allowed elements in the list are:

                + `y_optimization`:
                    do not save the predictions for the optimization set,
                    which would later on be used to build an ensemble. Note that SMAC
                    optimizes a metric evaluated on the optimization set.
                + `pipeline`:
                    do not save any individual pipeline files
                + `pipelines`:
                    In case of cross validation, disables saving the joint model of the
                    pipelines fit on each fold.
                + `y_test`:
                    do not save the predictions for the test set.
                + `all`:
                    do not save any of the above.
                For more information check `autoPyTorch.evaluation.utils.DisableFileOutputParameters`.
            load_models (bool), (default=True): Whether to load the
                models after fitting AutoPyTorch.
            suggested_init_models: Optional[List[str]]
                suggested initial models with their default configurations setting
            custom_init_setting_path: Optional[str]
                path to a json file that contains the initial configuration suggested by the users
            min_num_test_instances: Optional[int]
                if it is set None, then full validation sets will be evaluated in each fidelity. Otherwise, the number
                of instances in the test sets should be a value that is at least as great as this value, otherwise, the
                number of test instance is proportional to its fidelity
            forecasting_dataset_kwargs: Dict[Any]
                Forecasting dataset kwargs used to initialize forecasting dataset
        Returns:
            self

        """
        if memory_limit is not None:
            self._dataset_compression = get_dataset_compression_mapping(
                memory_limit, dataset_compression
            )
        else:
            self._dataset_compression = None

        self.dataset, self.input_validator = self._get_dataset_input_validator(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            resampling_strategy=self.resampling_strategy,
            resampling_strategy_args=self.resampling_strategy_args,
            dataset_name=dataset_name,
            dataset_compression=self._dataset_compression,
            freq=freq,
            start_times=start_times,
            series_idx=series_idx,
            n_prediction_steps=n_prediction_steps,
            **forecasting_dataset_kwargs,
        )

        if not self.customized_window_size:
            self.update_sliding_window_size(n_prediction_steps=n_prediction_steps)

        self._metrics_kwargs = {
            "sp": self.dataset.seasonality,
            "n_prediction_steps": n_prediction_steps,
        }

        forecasting_kwargs = dict(
            suggested_init_models=suggested_init_models,
            custom_init_setting_path=custom_init_setting_path,
            min_num_test_instances=min_num_test_instances,
        )

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
            **forecasting_kwargs,  # type: ignore[arg-type]
        )

    def predict(
        self,
        X_test: List[Union[np.ndarray, pd.DataFrame, TimeSeriesSequence]] = None,
        batch_size: Optional[int] = None,
        n_jobs: int = 1,
        past_targets: Optional[List[np.ndarray]] = None,
        future_targets: Optional[List[Union[np.ndarray, pd.DataFrame, TimeSeriesSequence]]] = None,
        start_times: List[pd.DatetimeIndex] = [],
    ) -> np.ndarray:
        """
        Predict the future varaibles

        Args:
            X_test (List[Union[np.ndarray, pd.DataFrame, TimeSeriesSequence]])
                if it is a list of TimeSeriesSequence, then it is the series to be forecasted. Otherwise, it is the
                known future features
            batch_size: Optional[int]
                batch size
            n_jobs (int):
                number of jobs
            past_targets (Optional[List[np.ndarray]])
                past observed targets, required when X_test is not a list of TimeSeriesSequence
            future_targets (Optional[List[Union[np.ndarray, pd.DataFrame, TimeSeriesSequence]]]):
                future targets (test sets)
            start_times (List[pd.DatetimeIndex]):
                starting time of each series when they are sampled. If it is not given, we simply start with a fixed
                timestamp.

        Return:
            np.ndarray
                predicted value, it needs to be with shape (B, H, N),
                B is the number of series, H is forecasting horizon (n_prediction_steps), N is the number of targets
        """
        if X_test is None or not isinstance(X_test[0], TimeSeriesSequence):
            assert past_targets is not None
            # Validate and construct TimeSeriesSequence
            X_test, _, _, _ = self.dataset.transform_data_into_time_series_sequence(
                X=X_test,
                Y=past_targets,
                X_test=future_targets,
                start_times=start_times,
                is_test_set=True,
            )
        flattened_res = super(TimeSeriesForecastingTask, self).predict(
            X_test, batch_size, n_jobs
        )
        # forecasting result from each series is stored as an array
        if self.dataset.num_targets == 1:
            forecasting = flattened_res.reshape([-1, self.dataset.n_prediction_steps])
        else:
            forecasting = flattened_res.reshape(
                [-1, self.dataset.n_prediction_steps, self.dataset.num_target]
            )
        if self.dataset.normalize_y:
            mean = np.repeat(
                self.dataset.y_mean.values(), self.dataset.n_prediction_steps
            )
            std = np.repeat(
                self.dataset.y_std.values(), self.dataset.n_prediction_steps
            )
            return forecasting * std + mean
        return forecasting

    def update_sliding_window_size(self, n_prediction_steps: int) -> None:
        """
        the size of the sliding window is heavily dependent on the dataset,
        so we only update them when we get the information from the

        Args:
            n_prediction_steps (int):
                forecast horizon. Sometimes we could also make our base sliding window size based on the
                forecast horizon
        """
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

        self.search_space_updates.append(
            node_name="data_loader",
            hyperparameter="window_size",
            value_range=[
                int(window_size_scales[0] * base_window_size),
                int(window_size_scales[1] * base_window_size),
            ],
            default_value=int(np.ceil(1.25 * base_window_size)),
        )
