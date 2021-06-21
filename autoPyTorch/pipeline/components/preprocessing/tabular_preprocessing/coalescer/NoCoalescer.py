from typing import Any, Dict, Optional, Union

import numpy as np

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.coalescer.base_coalescer import BaseCoalescer


class NoCoalescer(BaseCoalescer):
    """
    Don't perform NoCoalescer on categorical features
    """
    def __init__(self,
                 random_state: np.random.RandomState,
                 ):
        super().__init__()
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Optional[Any] = None) -> BaseCoalescer:
        """
        As no coalescing happens, the input fit dictionary is unchanged.

        Args:
        X (Dict[str, Any]):
            input fit dictionary
        y (Optional[Any]):
            Parameter to comply with scikit-learn API. Not used.

        Returns:
            instance of self
        """
        self.check_requirements(X, y)

        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add self into the 'X' dictionary and return the modified dict.
        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        X.update({'coalescer': self.preprocessor})
        return X

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'NoCoalescer',
            'name': 'No Coalescer',
            'handles_sparse': True
        }
