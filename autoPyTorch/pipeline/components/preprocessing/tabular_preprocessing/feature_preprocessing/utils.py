from typing import Dict, Optional
import warnings

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.utils.common import HyperparameterSearchSpace


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