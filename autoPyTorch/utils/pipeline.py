# -*- encoding: utf-8 -*-
from typing import Any, Dict, List, Optional

from ConfigSpace.configuration_space import ConfigurationSpace

from autoPyTorch.constants import (
    CLASSIFICATION_TASKS,
    IMAGE_TASKS,
    REGRESSION_TASKS,
    STRING_TO_TASK_TYPES,
    TABULAR_TASKS,
    TIMESERIES_TASKS,
)
from autoPyTorch.pipeline.image_classification import ImageClassificationPipeline
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline
from autoPyTorch.pipeline.tabular_regression import TabularRegressionPipeline
from autoPyTorch.pipeline.time_series_classification import TimeSeriesClassificationPipeline
from autoPyTorch.pipeline.time_series_regression import TimeSeriesRegressionPipeline
from autoPyTorch.utils.common import FitRequirement
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates

__all__ = [
    'get_dataset_requirements',
    'get_configuration_space'
]


def get_dataset_requirements(info: Dict[str, Any],
                             include_estimators: Optional[List[str]] = None,
                             exclude_estimators: Optional[List[str]] = None,
                             include_preprocessors: Optional[List[str]] = None,
                             exclude_preprocessors: Optional[List[str]] = None
                             ) -> List[FitRequirement]:
    exclude = dict()
    include = dict()
    if include_preprocessors is not None and \
            exclude_preprocessors is not None:
        raise ValueError('Cannot specify include_preprocessors and '
                         'exclude_preprocessors.')
    elif include_preprocessors is not None:
        include['feature_preprocessor'] = include_preprocessors
    elif exclude_preprocessors is not None:
        exclude['feature_preprocessor'] = exclude_preprocessors

    task_type: int = STRING_TO_TASK_TYPES[info['task_type']]
    if include_estimators is not None and \
            exclude_estimators is not None:
        raise ValueError('Cannot specify include_estimators and '
                         'exclude_estimators.')
    elif include_estimators is not None:
        if task_type in CLASSIFICATION_TASKS:
            include['classifier'] = include_estimators
        elif task_type in REGRESSION_TASKS:
            include['regressor'] = include_estimators
        else:
            raise ValueError(info['task_type'])
    elif exclude_estimators is not None:
        if task_type in CLASSIFICATION_TASKS:
            exclude['classifier'] = exclude_estimators
        elif task_type in REGRESSION_TASKS:
            exclude['regressor'] = exclude_estimators
        else:
            raise ValueError(info['task_type'])

    if task_type in REGRESSION_TASKS:
        return _get_regression_dataset_requirements(info, include, exclude)
    else:
        return _get_classification_dataset_requirements(info, include, exclude)


def _get_regression_dataset_requirements(info: Dict[str, Any], include: Dict[str, List[str]],
                                         exclude: Dict[str, List[str]]) -> List[FitRequirement]:
    task_type = STRING_TO_TASK_TYPES[info['task_type']]
    if task_type in TABULAR_TASKS:
        return TabularRegressionPipeline(
            dataset_properties=info,
            include=include,
            exclude=exclude
        ).get_dataset_requirements()

    elif task_type in TIMESERIES_TASKS:
        return TimeSeriesRegressionPipeline(
            dataset_properties=info,
            include=include,
            exclude=exclude
        ).get_dataset_requirements()

    else:
        raise ValueError("Task_type not supported")


def _get_classification_dataset_requirements(info: Dict[str, Any],
                                             include: Dict[str, List[str]],
                                             exclude: Dict[str, List[str]]) -> List[FitRequirement]:
    task_type = STRING_TO_TASK_TYPES[info['task_type']]

    if task_type in TABULAR_TASKS:
        return TabularClassificationPipeline(
            dataset_properties=info,
            include=include,
            exclude=exclude
        ).get_dataset_requirements()

    elif task_type in TIMESERIES_TASKS:
        return TimeSeriesClassificationPipeline(
            dataset_properties=info,
            include=include,
            exclude=exclude,
        ).get_dataset_requirements()

    elif task_type in IMAGE_TASKS:
        return ImageClassificationPipeline(
            dataset_properties=info,
            include=include,
            exclude=exclude
        ).get_dataset_requirements()

    else:
        raise ValueError("Task_type not supported")


def get_configuration_space(info: Dict[str, Any],
                            include: Optional[Dict] = None,
                            exclude: Optional[Dict] = None,
                            search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
                            ) -> ConfigurationSpace:
    task_type: int = STRING_TO_TASK_TYPES[info['task_type']]

    if task_type in REGRESSION_TASKS:
        return _get_regression_configuration_space(info,
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
        pipeline = TabularRegressionPipeline(dataset_properties=info,
                                             include=include,
                                             exclude=exclude,
                                             search_space_updates=search_space_updates)
        return pipeline.get_hyperparameter_search_space()

    elif STRING_TO_TASK_TYPES[info['task_type']] in TIMESERIES_TASKS:
        pipeline = TimeSeriesRegressionPipeline(dataset_properties=info,
                                                include=include, exclude=exclude,
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

    elif STRING_TO_TASK_TYPES[info['task_type']] in TIMESERIES_TASKS:
        pipeline = TimeSeriesClassificationPipeline(dataset_properties=info,
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
