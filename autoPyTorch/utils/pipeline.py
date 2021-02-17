# -*- encoding: utf-8 -*-
"""TODO: reduce strings as much as possible"""
from typing import Any, Dict, List, Optional, Tuple, NamedTuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace

from autoPyTorch.constants import (
    RegressionTypes,
    ClassificationTypes
)

from autoPyTorch.pipeline.image_classification import ImageClassificationPipeline
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline
from autoPyTorch.pipeline.tabular_regression import TabularRegressionPipeline
from autoPyTorch.utils.common import FitRequirement
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates

"""TODO: now rewriting
from autoPyTorch.constants import (
    CLASSIFICATION_TASKS,
    IMAGE_TASKS,
    REGRESSION_TASKS,
    STRING_TO_TASK_TYPES,
    TABULAR_TASKS,
)
"""

__all__ = [
    'get_dataset_requirements',
    'get_configuration_space'
]


class _PipeLineParameters(NamedTuple):
    dataset_properties: Dict[str, Any]
    include: Dict[str, List[str]]
    exclude: Dict[str, List[str]]
    search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None


"""TODO: consider better ways to refactor TaskDict, SupportedPipelines"""
TaskDict = {'classifier': ClassificationTypes, 'regressor': RegressionTypes}
SupportedPipelines = {
    'regressor': {
        RegressionTypes.tabular.name: TabularRegressionPipeline
    },
    'classifier': {
        ClassificationTypes.tabular.name: TabularClassificationPipeline,
        ClassificationTypes.image.name: ImageClassificationPipeline
    }
}


def _check_supported_tasks(task_type: Union[RegressionTypes, ClassificationTypes]) -> None:
    supported_tasks = TaskDict.values()

    if not any(isinstance(task_type, supported_tasks) for supported_task in supported_tasks):
        raise TypeError(f"task_type must be supported class type, but got '{type(task_type)}'")


def _check_preprocessor(include: Dict[str, Any],
                        exclude: Dict[str, Any],
                        include_preprocessors: List[str],
                        exclude_preprocessors: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    if None not in [include_preprocessors, exclude_preprocessors]:
        raise ValueError('Cannot specify include_preprocessors and '
                         'exclude_preprocessors.')
    elif include_preprocessors is not None:
        """TODO: what is include and exclude? Why don't we use NamedTuple?"""
        include['feature_preprocessor'] = include_preprocessors
    elif exclude_preprocessors is not None:
        exclude['feature_preprocessor'] = exclude_preprocessors

    return include, exclude


def _check_estimators(task_type: Union[RegressionTypes, ClassificationTypes],
                      include: Dict[str, Any],
                      exclude: Dict[str, Any],
                      include_estimators: List[str],
                      exclude_estimators: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    if None not in [include_estimators, exclude_estimators]:
        raise ValueError('Cannot specify include_estimators and '
                         'exclude_estimators.')

    if include_estimators is not None:
        for task_name, task_class in TaskDict.items():
            if isinstance(task_type, task_class):
                include[task_name] = include_estimators

    elif exclude_estimators is not None:
        for task_name, task_class in TaskDict.items():
            if isinstance(task_type, task_class):
                exclude[task_name] = exclude_estimators

    return include, exclude


def _check_dataset_requirements(task_type: Union[RegressionTypes, ClassificationTypes],
                                pipeline_params: Dict[str, Any]) -> List[FitRequirement]:

    if not task_type.is_supported():
        raise ValueError(f"The given task_type '{task_type}' is not supported.")

    dataset_type = task_type.name
    for task_name, task_class in TaskDict.items():
        if isinstance(task_type, task_class):
            pipeline = SupportedPipelines[task_name][dataset_type]
            return pipeline(**pipeline_params).get_dataset_requirements()


def _check_configspace(task_type: Union[RegressionTypes, ClassificationTypes],
                       pipeline_params: Dict[str, Any]) -> ConfigurationSpace:

    if not task_type.is_supported():
        raise ValueError(f"The given task_type '{task_type}' is not supported.")

    dataset_type = task_type.name
    for task_name, task_class in TaskDict.items():
        if isinstance(task_type, task_class):
            pipeline = SupportedPipelines[task_name][dataset_type]
            return pipeline(**pipeline_params).get_hyperparameter_search_space()


def get_dataset_requirements(dataset_properties: 'DatasetProperties',  # temporal name
                             include_estimators: Optional[List[str]] = None,
                             exclude_estimators: Optional[List[str]] = None,
                             include_preprocessors: Optional[List[str]] = None,
                             exclude_preprocessors: Optional[List[str]] = None
                             ) -> List[FitRequirement]:
    """TODO: make info NamedTuple"""
    """TODO: to be compatible with other files using get_dataset_requirements"""
    include, exclude = dict(), dict()
    task_type = dataset_properties.Task_type

    try:
        _check_supported_tasks(task_type)

        include, exclude = _check_preprocessor(include, exclude,
                                               include_preprocessors, exclude_preprocessors)
        include, exclude = _check_estimators(task_type, include, exclude,
                                             include_estimators, exclude_estimators)

        pipeline_params = _PipeLineParameters(dataset_properties=dataset_properties._asdict(),
                                              include=include, exclude=exclude)._asdict()

        return _check_dataset_requirements(task_type, pipeline_params)

    except ValueError:
        raise ValueError(f"Error occurred during getting the requirements of {task_type}")
    except TypeError:
        raise TypeError


def get_configuration_space(dataset_properties: 'DatasetProperties',  # temporal name
                            include: Optional[Dict] = None,
                            exclude: Optional[Dict] = None,
                            search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
                            ) -> ConfigurationSpace:

    task_type = dataset_properties.Task_type

    try:
        _check_supported_tasks(task_type)

        pipeline_params = _PipeLineParameters(dataset_properties=dataset_properties._asdict(),
                                              include=include, exclude=exclude,
                                              search_space_updates=search_space_updates)._asdict()
        _check_supported_tasks(task_type, pipeline_params)

    except ValueError:
        raise ValueError(f"Error occurred during getting the config space of {task_type}")
    except TypeError:
        raise TypeError
