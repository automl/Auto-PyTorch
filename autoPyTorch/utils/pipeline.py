# -*- encoding: utf-8 -*-
"""TODO: reduce strings as much as possible"""
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

from ConfigSpace.configuration_space import ConfigurationSpace

from autoPyTorch.constants import (
    CLASSIFICATION_TASKS,
    IMAGE_TASKS,
    REGRESSION_TASKS,
    STRING_TO_TASK_TYPES,
    TABULAR_TASKS,
)
from autoPyTorch.pipeline.image_classification import ImageClassificationPipeline
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline
from autoPyTorch.pipeline.tabular_regression import TabularRegressionPipeline
from autoPyTorch.utils.common import FitRequirement
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates

__all__ = [
    'get_dataset_requirements',
    'get_configuration_space'
]


class _PipeLineParameters(NamedTuple):
    dataset_properties: Dict[str, Any]
    include: Dict[str, List[str]]
    exclude: Dict[str, List[str]]


def _check_preprocessor(include: Dict[str, Any],
                        exclude: Dict[str, Any],
                        include_preprocessors: List[str],
                        exclude_preprocessors: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    if None not in [include_preprocessors, exclude_preprocessors]:
        raise ValueError('Cannot specify include_preprocessors and '
                         'exclude_preprocessors.')
    elif include_preprocessors is not None:
        include['feature_preprocessor'] = include_preprocessors
    elif exclude_preprocessors is not None:
        exclude['feature_preprocessor'] = exclude_preprocessors

    return include, exclude


def _check_estimators(task_name: str,
                      task_type: int,
                      include: Dict[str, Any],
                      exclude: Dict[str, Any],
                      include_estimators: List[str],
                      exclude_estimators: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    no_task_type = task_type not in (CLASSIFICATION_TASKS + REGRESSION_TASKS)

    if None not in [include_estimators, exclude_estimators]:
        raise ValueError('Cannot specify include_estimators and '
                         'exclude_estimators.')
    elif no_task_type and (include_estimators is not None or
                           exclude_estimators is not None):
        raise ValueError(f"The task_type {task_name} is not supported.")

    if include_estimators is not None:
        if task_type in CLASSIFICATION_TASKS:
            include['classifier'] = include_estimators
        elif task_type in REGRESSION_TASKS:
            include['regressor'] = include_estimators

    elif exclude_estimators is not None:
        if task_type in CLASSIFICATION_TASKS:
            exclude['classifier'] = exclude_estimators
        elif task_type in REGRESSION_TASKS:
            exclude['regressor'] = exclude_estimators

    return include, exclude


def _check_dataset_requirements(task_type: int, pipeline_params: Dict[str, Any]) -> List[FitRequirement]:
    """TODO: rewrite it nicer. Especially, the relation of REGRESSION TASKS and the supported tasks."""
    if task_type in REGRESSION_TASKS:
        if task_type in TABULAR_TASKS:
            return TabularRegressionPipeline(**pipeline_params).get_dataset_requirements()
        else:
            raise ValueError("Task_type not supported")
    elif task_type in CLASSIFICATION_TASKS:
        if task_type in TABULAR_TASKS:
            return TabularClassificationPipeline(**pipeline_params).get_dataset_requirements()
        elif task_type in IMAGE_TASKS:
            return ImageClassificationPipeline(**pipeline_params).get_dataset_requirements()
        else:
            raise ValueError("Task_type not supported")
    else:
        raise ValueError("The given task_type is not supported.")


def get_dataset_requirements(info: Dict[str, Any],
                             include_estimators: Optional[List[str]] = None,
                             exclude_estimators: Optional[List[str]] = None,
                             include_preprocessors: Optional[List[str]] = None,
                             exclude_preprocessors: Optional[List[str]] = None
                             ) -> List[FitRequirement]:
    include, exclude, task_name = dict(), dict(), info['task_type']

    try:
        task_type: int = STRING_TO_TASK_TYPES[task_name]
        include, exclude = _check_preprocessor(include, exclude,
                                               include_preprocessors, exclude_preprocessors)
        include, exclude = _check_estimators(task_name, task_type, include, exclude,
                                             include_estimators, exclude_estimators)

        pipeline_params = _PipeLineParameters(dataset_properties=info, include=include, exclude=exclude)._asdict()

        return _check_dataset_requirements(task_type, pipeline_params)

    except ValueError:
        raise ValueError(f"Error occurred during getting the requirements of {task_name}")
    except KeyError:
        raise KeyError(f"No match for task_type '{task_name}'")


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
        configuration_space = TabularRegressionPipeline(
            dataset_properties=info,
            include=include,
            exclude=exclude,
            search_space_updates=search_space_updates
        ).get_hyperparameter_search_space()
        return configuration_space
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
            search_space_updates=search_space_updates).\
            get_hyperparameter_search_space()
    else:
        raise ValueError("Task_type not supported")
