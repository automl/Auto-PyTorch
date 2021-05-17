import os
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

import pandas as pd

from autoPyTorch.api.base_task import BaseTask
from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.constants import (
    TABULAR_REGRESSION,
    TASK_TYPES_TO_STRING
)
from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
    HoldoutValTypes,
)
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.pipeline.tabular_regression import TabularRegressionPipeline
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates


class TabularRegressionTask(BaseTask):
    """
    Tabular Regression API to the pipelines.
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
        search_space_updates (Optional[HyperparameterSearchSpaceUpdates]):
            search space updates that can be used to modify the search
            space of particular components or choice modules of the pipeline
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
            resampling_strategy: Union[CrossValTypes, HoldoutValTypes] = HoldoutValTypes.holdout_validation,
            resampling_strategy_args: Optional[Dict[str, Any]] = None,
            backend: Optional[Backend] = None,
            search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
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
            task_type=TASK_TYPES_TO_STRING[TABULAR_REGRESSION],
        )

    def _get_required_dataset_properties(self, dataset: BaseDataset) -> Dict[str, Any]:
        if not isinstance(dataset, TabularDataset):
            raise ValueError("Dataset is incompatible for the given task,: {}".format(
                type(dataset)
            ))
        return {'task_type': dataset.task_type,
                'output_type': dataset.output_type,
                'issparse': dataset.issparse,
                'numerical_columns': dataset.numerical_columns,
                'categorical_columns': dataset.categorical_columns}

    def build_pipeline(self, dataset_properties: Dict[str, Any]) -> TabularRegressionPipeline:
        return TabularRegressionPipeline(dataset_properties=dataset_properties)

    def search(
        self,
        optimize_metric: str,
        X_train: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
        y_train: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
        X_test: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
        dataset_name: Optional[str] = None,
        budget_type: Optional[str] = None,
        budget: Optional[float] = None,
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
        portfolio_selection: str = "none"
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
            optimize_metric (str): name of the metric that is used to
                evaluate a pipeline.
            budget_type (Optional[str]):
                Type of budget to be used when fitting the pipeline.
                Either 'epochs' or 'runtime'. If not provided, uses
                the default in the pipeline config ('epochs')
            budget (Optional[float]):
                Budget to fit a single run of the pipeline. If not
                provided, uses the default in the pipeline config
            total_walltime_limit (int), (default=100): Time limit
                in seconds for the search of appropriate models.
                By increasing this value, autopytorch has a higher
                chance of finding better models.
            func_eval_time_limit_secs (int), (default=None): Time limit
                for a single call to the machine learning model.
                Model fitting will be terminated if the machine
                learning algorithm runs over the time limit. Set
                this value high enough so that typical machine
                learning algorithms can be fit on the training
                data.
                When set to None, this time will automatically be set to
                total_walltime_limit // 2 to allow enough time to fit
                at least 2 individual machine learning algorithms.
                Set to np.inf in case no time limit is desired.
            enable_traditional_pipeline (bool), (default=False):
                Not enabled for regression. This flag is here to comply
                with the API.
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
            portfolio_selection (str), (default="none"): If "greedy",
                runs initial configurations present in
                'autoPyTorch/configs/greedy_portfolio.json'.
                These configurations are the best performing configurations
                when search was performed on meta training datasets.
                For more info refer to `AutoPyTorch Tabular <https://arxiv.org/abs/2006.13799>`
        Returns:
            self

        """
        if dataset_name is None:
            dataset_name = str(uuid.uuid1(clock_seq=os.getpid()))

        # we have to create a logger for at this point for the validator
        self._logger = self._get_logger(dataset_name)

        # Create a validator object to make sure that the data provided by
        # the user matches the autopytorch requirements
        self.InputValidator = TabularInputValidator(
            is_classification=False,
            logger_port=self._logger_port,
        )

        # Fit a input validator to check the provided data
        # Also, an encoder is fit to both train and test data,
        # to prevent unseen categories during inference
        self.InputValidator.fit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        self.dataset = TabularDataset(
            X=X_train, Y=y_train,
            X_test=X_test, Y_test=y_test,
            validator=self.InputValidator,
            resampling_strategy=self.resampling_strategy,
            resampling_strategy_args=self.resampling_strategy_args,
        )

        return self._search(
            dataset=self.dataset,
            optimize_metric=optimize_metric,
            budget_type=budget_type,
            budget=budget,
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
            portfolio_selection=portfolio_selection
        )

    def predict(
            self,
            X_test: np.ndarray,
            batch_size: Optional[int] = None,
            n_jobs: int = 1
    ) -> np.ndarray:
        if self.InputValidator is None or not self.InputValidator._is_fitted:
            raise ValueError("predict() is only supported after calling search. Kindly call first "
                             "the estimator fit() method.")

        X_test = self.InputValidator.feature_validator.transform(X_test)
        predicted_values = super().predict(X_test, batch_size=batch_size,
                                           n_jobs=n_jobs)

        # Allow to predict in the original domain -- that is, the user is not interested
        # in our encoded values
        return self.InputValidator.target_validator.inverse_transform(predicted_values)
