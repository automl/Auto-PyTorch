from typing import Any, Dict, List

from sklearn.base import BaseEstimator


def get_tabular_preprocessers(X: Dict[str, Any]) -> Dict[str, List[BaseEstimator]]:
    """
    Expects fit_dictionary(X) to have numerical/categorical preprocessors
    (fited numerical/categorical preprocessing nodes) that will build the
    column transformer in the TabularColumnTransformer. This function
    parses X and extracts such components.
    Creates a dictionary with two keys,
    numerical- containing list of numerical preprocessors
    categorical- containing list of categorical preprocessors
    Args:
        X: fit dictionary
    Returns:
        (Dict[str, List[BaseEstimator]]): dictionary with list of numerical and categorical preprocessors
    """
    preprocessor: Dict[str, List[BaseEstimator]] = dict(numerical=list(), categorical=list())
    for key, value in X.items():
        if isinstance(value, dict):
            # as each preprocessor is child of BaseEstimator
            if 'numerical' in value and isinstance(value['numerical'], BaseEstimator):
                preprocessor['numerical'].append(value['numerical'])
            if 'categorical' in value and isinstance(value['categorical'], BaseEstimator):
                preprocessor['categorical'].append(value['categorical'])

    return preprocessor
