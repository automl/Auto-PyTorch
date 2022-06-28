# -*- encoding: utf-8 -*-
from typing import Any, Dict, List, Optional

from ConfigSpace.configuration_space import ConfigurationSpace

from autoPyTorch.constants import (
    CLASSIFICATION_TASKS,
    FORECASTING_TASKS,
    ForecastingDependenciesNotInstalledMSG,
    IMAGE_TASKS,
    REGRESSION_TASKS,
    STRING_TO_TASK_TYPES,
    TABULAR_TASKS,
)
from autoPyTorch.pipeline.image_classification import ImageClassificationPipeline
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline
from autoPyTorch.pipeline.tabular_regression import TabularRegressionPipeline
try:
    from autoPyTorch.pipeline.time_series_forecasting import TimeSeriesForecastingPipeline
    forecasting_dependencies_installed = True
except ModuleNotFoundError:
    forecasting_dependencies_installed = False
from autoPyTorch.utils.common import FitRequirement
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates

__all__ = [
    'get_dataset_requirements',
    'get_configuration_space'
]


def get_dataset_requirements(info: Dict[str, Any],
                             include: Optional[Dict] = None,
                             exclude: Optional[Dict] = None,
                             search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
                             ) -> List[FitRequirement]:
    """

    This function is used to return the dataset
    property requirements which are needed to fit
    a pipeline created based on the constraints
    specified using include, exclude and
    search_space_updates

    Args:
        info (Dict[str, Any]):
             A dictionary that specifies the required information
             about the dataset to instantiate a pipeline. For more
             info check the get_required_dataset_info of the
             appropriate dataset in autoPyTorch/datasets
        include (Optional[Dict]):
            If None, all possible components are used.
            Otherwise specifies set of components to use.
        exclude (Optional[Dict]):
            If None, all possible components are used.
            Otherwise specifies set of components not to use.
            Incompatible with include.
        search_space_updates (Optional[HyperparameterSearchSpaceUpdates]):
            search space updates that can be used to modify the search
            space of particular components or choice modules of the pipeline

    Returns:
        List[FitRequirement]:
            List of requirements that should be in the fit
            dictionary used to fit the pipeline.
    """
    task_type: int = STRING_TO_TASK_TYPES[info['task_type']]
    if task_type in REGRESSION_TASKS:
        return _get_regression_dataset_requirements(info,
                                                    include if include is not None else {},
                                                    exclude if exclude is not None else {},
                                                    search_space_updates=search_space_updates
                                                    )
    elif task_type in CLASSIFICATION_TASKS:
        return _get_classification_dataset_requirements(info,
                                                        include if include is not None else {},
                                                        exclude if exclude is not None else {},
                                                        search_space_updates=search_space_updates
                                                        )
    else:
        if not forecasting_dependencies_installed:
            raise ModuleNotFoundError(ForecastingDependenciesNotInstalledMSG)
        return _get_forecasting_dataset_requirements(info,
                                                     include if include is not None else {},
                                                     exclude if exclude is not None else {},
                                                     search_space_updates=search_space_updates
                                                     )


def _get_regression_dataset_requirements(info: Dict[str, Any],
                                         include: Optional[Dict] = None,
                                         exclude: Optional[Dict] = None,
                                         search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
                                         ) -> List[FitRequirement]:
    task_type = STRING_TO_TASK_TYPES[info['task_type']]
    if task_type in TABULAR_TASKS:
        return TabularRegressionPipeline(
            dataset_properties=info,
            include=include,
            exclude=exclude,
            search_space_updates=search_space_updates
        ).get_dataset_requirements()

    else:
        raise ValueError("Task_type not supported")


def _get_classification_dataset_requirements(info: Dict[str, Any],
                                             include: Optional[Dict] = None,
                                             exclude: Optional[Dict] = None,
                                             search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
                                             ) -> List[FitRequirement]:
    task_type = STRING_TO_TASK_TYPES[info['task_type']]

    if task_type in TABULAR_TASKS:
        return TabularClassificationPipeline(
            dataset_properties=info,
            include=include, exclude=exclude,
            search_space_updates=search_space_updates
        ).get_dataset_requirements()
    elif task_type in IMAGE_TASKS:
        return ImageClassificationPipeline(
            dataset_properties=info,
            include=include, exclude=exclude,
            search_space_updates=search_space_updates
        ).get_dataset_requirements()
    else:
        raise ValueError("Task_type not supported")


def _get_forecasting_dataset_requirements(info: Dict[str, Any],
                                          include: Optional[Dict] = None,
                                          exclude: Optional[Dict] = None,
                                          search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
                                          ) -> List[FitRequirement]:
    task_type = STRING_TO_TASK_TYPES[info['task_type']]

    if task_type in FORECASTING_TASKS:
        if not forecasting_dependencies_installed:
            raise ModuleNotFoundError(ForecastingDependenciesNotInstalledMSG)
        return TimeSeriesForecastingPipeline(
            dataset_properties=info,
            include=include,
            exclude=exclude,
            search_space_updates=search_space_updates
        ).get_dataset_requirements()
    else:
        raise ValueError("Task_type not supported")


def get_configuration_space(info: Dict[str, Any],
                            include: Optional[Dict] = None,
                            exclude: Optional[Dict] = None,
                            search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
                            ) -> ConfigurationSpace:
    """

    This function is used to return the configuration
    space of the pipeline created based on the constraints
    specified using include, exclude and search_space_updates

    Args:
        info (Dict[str, Any]):
             A dictionary that specifies the required information
             about the dataset to instantiate a pipeline. For more
             info check the get_required_dataset_info of the
             appropriate dataset in autoPyTorch/datasets
        include (Optional[Dict]):
            If None, all possible components are used.
            Otherwise specifies set of components to use.
        exclude (Optional[Dict]):
            If None, all possible components are used.
            Otherwise specifies set of components not to use.
            Incompatible with include.
        search_space_updates (Optional[HyperparameterSearchSpaceUpdates]):
            search space updates that can be used to modify the search
            space of particular components or choice modules of the pipeline

    Returns:
        ConfigurationSpace

    """
    task_type: int = STRING_TO_TASK_TYPES[info['task_type']]

    if task_type in REGRESSION_TASKS:
        return _get_regression_configuration_space(info,
                                                   include if include is not None else {},
                                                   exclude if exclude is not None else {},
                                                   search_space_updates=search_space_updates
                                                   )
    elif task_type in FORECASTING_TASKS:
        return _get_forecasting_configuration_space(info,
                                                    include if include is not None else {},
                                                    exclude if exclude is not None else {},
                                                    search_space_updates=search_space_updates
                                                    )
    else:
        return _get_classification_configuration_space(info,
                                                       include if include is not None else {},
                                                       exclude if exclude is not None else {},
                                                       search_space_updates=search_space_updates
                                                       )


def _get_regression_configuration_space(info: Dict[str, Any], include: Dict[str, List[str]],
                                        exclude: Dict[str, List[str]],
                                        search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
                                        ) -> ConfigurationSpace:
    if STRING_TO_TASK_TYPES[info['task_type']] in TABULAR_TASKS:
        pipeline = TabularRegressionPipeline(
            dataset_properties=info,
            include=include,
            exclude=exclude,
            search_space_updates=search_space_updates)
        return pipeline.get_hyperparameter_search_space()

    else:
        raise ValueError("Task_type not supported")


def _get_classification_configuration_space(info: Dict[str, Any], include: Dict[str, List[str]],
                                            exclude: Dict[str, List[str]],
                                            search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
                                            ) -> ConfigurationSpace:
    if STRING_TO_TASK_TYPES[info['task_type']] in TABULAR_TASKS:
        pipeline = TabularClassificationPipeline(dataset_properties=info,
                                                 include=include, exclude=exclude,
                                                 search_space_updates=search_space_updates)
        return pipeline.get_hyperparameter_search_space()

    elif STRING_TO_TASK_TYPES[info['task_type']] in IMAGE_TASKS:
        return ImageClassificationPipeline(
            dataset_properties=info,
            include=include, exclude=exclude,
            search_space_updates=search_space_updates). \
            get_hyperparameter_search_space()
    else:
        raise ValueError("Task_type not supported")


def _get_forecasting_configuration_space(info: Dict[str, Any], include: Dict[str, List[str]],
                                         exclude: Dict[str, List[str]],
                                         search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
                                         ) -> ConfigurationSpace:
    pipeline = TimeSeriesForecastingPipeline(dataset_properties=info,
                                             include=include, exclude=exclude,
                                             search_space_updates=search_space_updates)
    return pipeline.get_hyperparameter_search_space()
