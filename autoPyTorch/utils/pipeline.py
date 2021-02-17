# -*- encoding: utf-8 -*-
"""TODO: reduce strings as much as possible"""
from typing import Any, Dict, List, Optional, Tuple, NamedTuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace

from autoPyTorch.constants import (
    RegressionTypes,
    ClassificationTypes,
    SupportedTaskTypes
)

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
    search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None


def _check_supported_tasks(task_type: Union[RegressionTypes, ClassificationTypes]) -> None:
    if not any(isinstance(task_type, supported_task_type) for supported_task_type in SupportedTaskTypes):
        raise TypeError(f"task_type must be supported class type, but got '{type(task_type)}'")
    elif not task_type.is_supported():
        raise TypeError(f"The given task_type '{task_type}' is not supported.")


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
        include[task_type.task_name] = include_estimators
    elif exclude_estimators is not None:
        exclude[task_type.task_name] = exclude_estimators

    return include, exclude


def get_dataset_requirements(dataset_properties: 'DatasetProperties',  # temporal name
                             include_estimators: Optional[List[str]] = None,
                             exclude_estimators: Optional[List[str]] = None,
                             include_preprocessors: Optional[List[str]] = None,
                             exclude_preprocessors: Optional[List[str]] = None
                             ) -> List[FitRequirement]:
    """TODO: make 'info' (older argument) NamedTuple"""
    """TODO: to be compatible with other files using get_dataset_requirements"""
    """TODO: DatasetProperties can be merged in BaseDataset in my opinion."""
    include, exclude = dict(), dict()
    task_type = dataset_properties.task_type

    _check_supported_tasks(task_type)

    include, exclude = _check_preprocessor(include, exclude,
                                           include_preprocessors, exclude_preprocessors)
    include, exclude = _check_estimators(task_type, include, exclude,
                                         include_estimators, exclude_estimators)

    pipeline_params = _PipeLineParameters(dataset_properties=dataset_properties._asdict(),
                                          include=include, exclude=exclude)._asdict()

    return task_type.pipeline(**pipeline_params).get_dataset_requirements()


def get_configuration_space(dataset_properties: 'DatasetProperties',  # temporal name
                            include: Optional[Dict] = None,
                            exclude: Optional[Dict] = None,
                            search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
                            ) -> ConfigurationSpace:

    task_type = dataset_properties.task_type

    _check_supported_tasks(task_type)

    pipeline_params = _PipeLineParameters(dataset_properties=dataset_properties._asdict(),
                                          include=include, exclude=exclude,
                                          search_space_updates=search_space_updates)._asdict()

    return task_type.pipeline(**pipeline_params).get_hyperparameter_search_space()
