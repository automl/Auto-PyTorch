import warnings
from math import ceil, floor
from typing import Dict, List, Optional, Sequence

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.utils.common import HyperparameterSearchSpace, HyperparameterValueType


NoneType_ = Optional[str]  # This typing is exclusively for Literal["none", "None", None]
# TODO: when we drop support for 3.7 use the following line
# NoneType_ = Optional[Literal["none", "None"]]


def filter_score_func_choices(
    class_name: str,
    score_func: HyperparameterSearchSpace,
    dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
) -> HyperparameterSearchSpace:
    """
    In the context of select rates classification or select percentile classification,
    some score functions are not compatible with sparse or signed data.
    This function filters out those score function from the search space of the component
    depending on the dataset.

    Args:
        score_func (HyperparameterSearchSpace)
        dataset_properties (Optional[Dict[str, BaseDatasetPropertiesType]]):
            Information about the dataset. Defaults to None.

    Raises:
        ValueError:
            if none of the score function choices are incompatible with the dataset

    Returns:
        HyperparameterSearchSpace:
            updated score function search space
    """
    value_range = list(score_func.value_range)
    if dataset_properties is not None:
        if dataset_properties.get("issigned", False):
            value_range = [value for value in value_range if value not in ("chi2", "mutual_info_classif")]
        if dataset_properties.get("issparse", False):
            value_range = [value for value in value_range if value != "f_classif"]

    if sorted(value_range) != sorted(list(score_func.value_range)):
        warnings.warn(f"Given choices for `score_func` are not compatible with the dataset. "
                      f"Updating choices to {value_range}")

    if len(value_range) == 0:
        raise ValueError(f"`{class_name}` is not compatible with the"
                         f" current dataset as it is both `signed` and `sparse`")
    default_value = score_func.default_value if score_func.default_value in value_range else value_range[-1]
    score_func = HyperparameterSearchSpace(hyperparameter="score_func",
                                           value_range=value_range,
                                           default_value=default_value,
                                           )
    return score_func


def percentage_value_range_to_integer_range(
    hyperparameter_search_space: HyperparameterSearchSpace,
    default_value_range: Sequence[HyperparameterValueType],
    default_value: int,
    dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
) -> HyperparameterSearchSpace:
    """
    For some feature preprocessors, the value of an integer hyperparameter
    needs to be lower than the number of features. To facilitate this,
    autoPyTorch uses a value range based on the percentage of the number
    of features. This function converts that hyperparameter search space
    to an integer value range as is required by the underlying sklearn
    preprocessors.
    """
    hyperparameter_name = hyperparameter_search_space.hyperparameter
    if dataset_properties is not None:
        n_features = len(dataset_properties['numerical_columns']) if isinstance(
            dataset_properties['numerical_columns'], List) else 0
        if n_features == 1:
            # log=True is not supported in ConfigSpace when the value range consists of 0
            # raising ValueError: Negative lower bound (0) for log-scale hyperparameter is forbidden.
            log = False
        else:
            log = hyperparameter_search_space.log
        hyperparameter_search_space = HyperparameterSearchSpace(
            hyperparameter=hyperparameter_name,
            value_range=(
                floor(float(hyperparameter_search_space.value_range[0]) * n_features),
                floor(float(hyperparameter_search_space.value_range[1]) * n_features)),
            default_value=ceil(float(hyperparameter_search_space.default_value) * n_features),
            log=log)
    else:
        hyperparameter_search_space = HyperparameterSearchSpace(hyperparameter=hyperparameter_name,
                                                                value_range=default_value_range,
                                                                default_value=default_value,
                                                                log=hyperparameter_search_space.log)

    return hyperparameter_search_space
