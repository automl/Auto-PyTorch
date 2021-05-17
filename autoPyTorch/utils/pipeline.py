# -*- encoding: utf-8 -*-
from typing import Any, Dict, List, Optional

from ConfigSpace.configuration_space import ConfigurationSpace

from autoPyTorch.constants import (
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


def get_dataset_requirements(info: Dict[str, Any],
                             include: Optional[Dict] = None,
                             exclude: Optional[Dict] = None,
                             search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
                             ) -> List[FitRequirement]:
    task_type: int = STRING_TO_TASK_TYPES[info['task_type']]
    if task_type in REGRESSION_TASKS:
        return _get_regression_dataset_requirements(info,
                                                    include if include is not None else {},
                                                    exclude if exclude is not None else {},
                                                    search_space_updates=search_space_updates
                                                    )
    else:
        return _get_classification_dataset_requirements(info,
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
        fit_requirements = TabularRegressionPipeline(
            dataset_properties=info,
            include=include,
            exclude=exclude,
            search_space_updates=search_space_updates
        ).get_dataset_requirements()
        return fit_requirements
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
            search_space_updates=search_space_updates). \
            get_dataset_requirements()
    elif task_type in IMAGE_TASKS:
        return ImageClassificationPipeline(
            dataset_properties=info,
            include=include, exclude=exclude,
            search_space_updates=search_space_updates). \
            get_dataset_requirements()
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
            search_space_updates=search_space_updates). \
            get_hyperparameter_search_space()
    else:
        raise ValueError("Task_type not supported")
