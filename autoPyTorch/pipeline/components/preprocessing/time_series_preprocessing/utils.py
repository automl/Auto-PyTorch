from typing import Any, Dict, List

from sklearn.base import BaseEstimator

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.utils import get_tabular_preprocessers


def get_time_series_preprocessors(X: Dict[str, Any]) -> Dict[str, List[BaseEstimator]]:
    """
    This function simply rename tabular preprocessor to time series preprocessor.
    """
    return get_tabular_preprocessers(X)


def get_time_series_target_preprocessers(X: Dict[str, Any]) -> Dict[str, List[BaseEstimator]]:
    """
    Expects fit_dictionary(X) to have target preprocessors
    I leave here interface to target categorical
    (fitted numerical/categorical preprocessing nodes) that will build a pipeline in the TimeSeriesTransformer.
    This function parses X and extracts such components.
    Creates a dictionary with two keys,
    numerical- containing list of numerical preprocessors
    categorical- containing list of categorical preprocessors

    Args:
        X: fit dictionary

    Returns:
        (Dict[str, List[BaseEstimator]]): dictionary with list of numerical and categorical preprocessors
    """
    preprocessor = dict(target_numerical=list(), target_categorical=list())  # type: Dict[str, List[BaseEstimator]]
    for key, value in X.items():
        if isinstance(value, dict):
            # as each preprocessor is child of BaseEstimator
            if 'target_numerical' in value and isinstance(value['target_numerical'], BaseEstimator):
                preprocessor['target_numerical'].append(value['target_numerical'])
            if 'target_categorical' in value and isinstance(value['target_categorical'], BaseEstimator):
                preprocessor['target_categorical'].append(value['target_categorical'])
    return preprocessor
