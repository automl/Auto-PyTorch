from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

import pandas as pd

from autoPyTorch.api.base_task import BaseTask
from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.constants import (
    TABULAR_REGRESSION,
    TASK_TYPES_TO_STRING
)
from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.data.utils import (
    DatasetCompressionSpec,
    get_dataset_compression_mapping,
)
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.datasets.resampling_strategy import (
    HoldoutValTypes,
    ResamplingStrategies,
)
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.evaluation.utils import DisableFileOutputParameters
from autoPyTorch.pipeline.tabular_regression import TabularRegressionPipeline
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates


class TabularRegressionTask(BaseTask):
    """
    Tabular Regression API to the pipelines.

    Args:
        seed (int: default=1):
            seed to be used for reproducibility.
        n_jobs (int: default=1):
            number of consecutive processes to spawn.
        n_threads (int: default=1):
            number of threads to use for each process.
        logging_config (Optional[Dict]):
            Specifies configuration for logging, if None, it is loaded from the logging.yaml
        ensemble_size (int: default=50):
            Number of models added to the ensemble built by
            Ensemble selection from libraries of models.
            Models are drawn with replacement.
        ensemble_nbest (int: default=50):
            Only consider the ensemble_nbest
            models to build the ensemble
        max_models_on_disc (int: default=50):
            Maximum number of models saved to disc.
            Also, controls the size of the ensemble
            as any additional models will be deleted.
            Must be greater than or equal to 1.
        temporary_directory (str):
            Folder to store configuration output and log file
        output_directory (str):
            Folder to store predictions for optional test set
        delete_tmp_folder_after_terminate (bool):
            Determines whether to delete the temporary directory,
            when finished
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
        resampling_strategy resampling_strategy (RESAMPLING_STRATEGIES),
                (default=HoldoutValTypes.holdout_validation):
                strategy to split the training data.
        resampling_strategy_args (Optional[Dict[str, Any]]): arguments
            required for the chosen resampling strategy. If None, uses
            the default values provided in DEFAULT_RESAMPLING_PARAMETERS
            in ```datasets/resampling_strategy.py```.
        search_space_updates (Optional[HyperparameterSearchSpaceUpdates]):
            Search space updates that can be used to modify the search
            space of particular components or choice modules of the pipeline
    """

    def __init__(
        self,
        seed: int = 1,
        n_jobs: int = 1,
        n_threads: int = 1,
        logging_config: Optional[Dict] = None,
        ensemble_size: int = 50,
        ensemble_nbest: int = 50,
        max_models_on_disc: int = 50,
        temporary_directory: Optional[str] = None,
        output_directory: Optional[str] = None,
        delete_tmp_folder_after_terminate: bool = True,
        delete_output_folder_after_terminate: bool = True,
        include_components: Optional[Dict[str, Any]] = None,
        exclude_components: Optional[Dict[str, Any]] = None,
        resampling_strategy: ResamplingStrategies = HoldoutValTypes.holdout_validation,
        resampling_strategy_args: Optional[Dict[str, Any]] = None,
        backend: Optional[Backend] = None,
        search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
    ):
        super().__init__(
            seed=seed,
            n_jobs=n_jobs,
            n_threads=n_threads,
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
            task_type=TASK_TYPES_TO_STRING[TABULAR_REGRESSION],
        )

    def build_pipeline(
        self,
        dataset_properties: Dict[str, BaseDatasetPropertiesType],
        include_components: Optional[Dict[str, Any]] = None,
        exclude_components: Optional[Dict[str, Any]] = None,
        search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
    ) -> TabularRegressionPipeline:
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
            TabularRegressionPipeline:

        """
        return TabularRegressionPipeline(dataset_properties=dataset_properties,
                                         include=include_components,
                                         exclude=exclude_components,
                                         search_space_updates=search_space_updates)

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
        **kwargs: Any
    ) -> Tuple[TabularDataset, TabularInputValidator]:
        """
        Returns an object of `TabularDataset` and an object of
        `TabularInputValidator` according to the current task.

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
            kwargs (Any):
                Currently for tabular tasks, expect `feat_types: (Optional[List[str]]` which
                specifies whether a feature is 'numerical' or 'categorical'.
        Returns:
            TabularDataset:
                the dataset object.
            TabularInputValidator:
                the input validator fitted on the data.
        """

        resampling_strategy = resampling_strategy if resampling_strategy is not None else self.resampling_strategy
        resampling_strategy_args = resampling_strategy_args if resampling_strategy_args is not None else \
            self.resampling_strategy_args

        feat_types = kwargs.pop('feat_types', None)
        # Create a validator object to make sure that the data provided by
        # the user matches the autopytorch requirements
        input_validator = TabularInputValidator(
            is_classification=False,
            logger_port=self._logger_port,
            dataset_compression=dataset_compression,
            feat_types=feat_types
        )

        # Fit a input validator to check the provided data
        # Also, an encoder is fit to both train and test data,
        # to prevent unseen categories during inference
        input_validator.fit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        dataset = TabularDataset(
            X=X_train, Y=y_train,
            X_test=X_test, Y_test=y_test,
            validator=input_validator,
            resampling_strategy=resampling_strategy,
            resampling_strategy_args=resampling_strategy_args,
            dataset_name=dataset_name
        )

        return dataset, input_validator

    def search(
        self,
        optimize_metric: str,
        X_train: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
        y_train: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
        X_test: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
        dataset_name: Optional[str] = None,
        feat_types: Optional[List[str]] = None,
        budget_type: str = 'epochs',
        min_budget: int = 5,
        max_budget: int = 50,
        total_walltime_limit: int = 100,
        func_eval_time_limit_secs: Optional[int] = None,
        enable_traditional_pipeline: bool = True,
        memory_limit: int = 4096,
        smac_scenario_args: Optional[Dict[str, Any]] = None,
        get_smac_object_callback: Optional[Callable] = None,
        all_supported_metrics: bool = True,
        precision: int = 32,
        disable_file_output: Optional[List[Union[str, DisableFileOutputParameters]]] = None,
        load_models: bool = True,
        portfolio_selection: Optional[str] = None,
        dataset_compression: Union[Mapping[str, Any], bool] = False,
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
            feat_types (Optional[List[str]]):
                Description about the feature types of the columns.
                Accepts `numerical` for integers, float data and `categorical`
                for categories, strings and bool. Defaults to None.
            optimize_metric (str):
                Name of the metric that is used to evaluate a pipeline.
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
            min_budget (int):
                Auto-PyTorch uses `Hyperband <https://arxiv.org/abs/1603.06560>`_ to
                trade-off resources between running many pipelines at min_budget and
                running the top performing pipelines on max_budget.
                min_budget states the minimum resource allocation a pipeline should have
                so that we can compare and quickly discard bad performing models.
                For example, if the budget_type is epochs, and min_budget=5, then we will
                run every pipeline to a minimum of 5 epochs before performance comparison.
            max_budget (int):
                Auto-PyTorch uses `Hyperband <https://arxiv.org/abs/1603.06560>`_ to
                trade-off resources between running many pipelines at min_budget and
                running the top performing pipelines on max_budget.
                max_budget states the maximum resource allocation a pipeline is going to
                be ran. For example, if the budget_type is epochs, and max_budget=50,
                then the pipeline training will be terminated after 50 epochs.
            total_walltime_limit (int: default=100):
                Time limit in seconds for the search of appropriate models.
                By increasing this value, autopytorch has a higher
                chance of finding better models.
            func_eval_time_limit_secs (Optional[int]):
                Time limit for a single call to the machine learning model.
                Model fitting will be terminated if the machine
                learning algorithm runs over the time limit. Set
                this value high enough so that typical machine
                learning algorithms can be fit on the training
                data.
                When set to None, this time will automatically be set to
                total_walltime_limit // 2 to allow enough time to fit
                at least 2 individual machine learning algorithms.
                Set to np.inf in case no time limit is desired.
            enable_traditional_pipeline (bool: default=True):
                We fit traditional machine learning algorithms
                (LightGBM, CatBoost, RandomForest, ExtraTrees, KNN, SVM)
                prior building PyTorch Neural Networks. You can disable this
                feature by turning this flag to False. All machine learning
                algorithms that are fitted during search() are considered for
                ensemble building.
            memory_limit (int: default=4096):
                Memory limit in MB for the machine learning algorithm.
                Autopytorch will stop fitting the machine learning algorithm
                if it tries to allocate more than memory_limit MB. If None
                is provided, no memory limit is set. In case of multi-processing,
                memory_limit will be per job. This memory limit also applies to
                the ensemble creation process.
            smac_scenario_args (Optional[Dict]):
                Additional arguments inserted into the scenario of SMAC. See the
                `SMAC documentation <https://automl.github.io/SMAC3/master/options.html?highlight=scenario#scenario>`_
                for a list of available arguments.
            get_smac_object_callback (Optional[Callable]):
                Callback function to create an object of class
                `smac.optimizer.smbo.SMBO <https://automl.github.io/SMAC3/master/apidoc/smac.optimizer.smbo.html>`_.
                The function must accept the arguments scenario_dict,
                instances, num_params, runhistory, seed and ta. This is
                an advanced feature. Use only if you are familiar with
                `SMAC <https://automl.github.io/SMAC3/master/index.html>`_.
            tae_func (Optional[Callable]):
                TargetAlgorithm to be optimised. If None, `eval_function`
                available in autoPyTorch/evaluation/train_evaluator is used.
                Must be child class of AbstractEvaluator.
            all_supported_metrics (bool: default=True):
                If True, all metrics supporting current task will be calculated
                for each pipeline and results will be available via cv_results
            precision (int: default=32):
                Numeric precision used when loading ensemble data.
                Can be either '16', '32' or '64'.
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
            load_models (bool: default=True):
                Whether to load the models after fitting AutoPyTorch.
            portfolio_selection (Optional[str]):
                This argument controls the initial configurations that
                AutoPyTorch uses to warm start SMAC for hyperparameter
                optimization. By default, no warm-starting happens.
                The user can provide a path to a json file containing
                configurations, similar to (...herepathtogreedy...).
                Additionally, the keyword 'greedy' is supported,
                which would use the default portfolio from
                `AutoPyTorch Tabular <https://arxiv.org/abs/2006.13799>`_.
            dataset_compression: Union[bool, Mapping[str, Any]] = True
                We compress datasets so that they fit into some predefined amount of memory.
                **NOTE**

                Default configuration when left as ``True``:
                .. code-block:: python
                    {
                        "memory_allocation": 0.1,
                        "methods": ["precision"]
                    }
                You can also pass your own configuration with the same keys and choosing
                from the available ``"methods"``.
                The available options are described here:
                **memory_allocation**
                    By default, we attempt to fit the dataset into ``0.1 * memory_limit``. This
                    float value can be set with ``"memory_allocation": 0.1``. We also allow for
                    specifying absolute memory in MB, e.g. 10MB is ``"memory_allocation": 10``.
                    The memory used by the dataset is checked after each reduction method is
                    performed. If the dataset fits into the allocated memory, any further methods
                    listed in ``"methods"`` will not be performed.

                **methods**
                    We currently provide the following methods for reducing the dataset size.
                    These can be provided in a list and are performed in the order as given.
                    *   ``"precision"`` -
                        We reduce floating point precision as follows:
                            *   ``np.float128 -> np.float64``
                            *   ``np.float96 -> np.float64``
                            *   ``np.float64 -> np.float32``
                            *   pandas dataframes are reduced using the downcast option of `pd.to_numeric`
                                to the lowest possible precision.
                    *   ``subsample`` -
                        We subsample data such that it **fits directly into
                        the memory allocation** ``memory_allocation * memory_limit``.
                        Therefore, this should likely be the last method listed in
                        ``"methods"``.
                        Subsampling takes into account classification labels and stratifies
                        accordingly. We guarantee that at least one occurrence of each
                        label is included in the sampled set.

        Returns:
            self

        """

        self._dataset_compression = get_dataset_compression_mapping(memory_limit, dataset_compression)

        self.dataset, self.input_validator = self._get_dataset_input_validator(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            resampling_strategy=self.resampling_strategy,
            resampling_strategy_args=self.resampling_strategy_args,
            dataset_name=dataset_name,
            dataset_compression=self._dataset_compression,
            feat_types=feat_types)

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
        )

    def predict(
            self,
            X_test: np.ndarray,
            batch_size: Optional[int] = None,
            n_jobs: int = 1
    ) -> np.ndarray:
        if self.input_validator is None or not self.input_validator._is_fitted:
            raise ValueError("predict() is only supported after calling search. Kindly call first "
                             "the estimator search() method.")

        X_test = self.input_validator.feature_validator.transform(X_test)
        predicted_values = super().predict(X_test, batch_size=batch_size,
                                           n_jobs=n_jobs)

        # Allow to predict in the original domain -- that is, the user is not interested
        # in our encoded values
        return self.input_validator.target_validator.inverse_transform(predicted_values)
