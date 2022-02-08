from typing import Any, Dict, Optional, Union

import numpy as np

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.coalescer.base_coalescer import BaseCoalescer


class NoCoalescer(BaseCoalescer):
    def __init__(self, random_state: np.random.RandomState):
        super().__init__()
        self.random_state = random_state
        self._processing = False

    def fit(self, X: Dict[str, Any], y: Optional[Any] = None) -> BaseCoalescer:
        """
        As no coalescing happens, only check the requirements.

        Args:
            X (Dict[str, Any]):
                fit dictionary
            y (Optional[Any]):
                Parameter to comply with scikit-learn API. Not used.

        Returns:
            instance of self
        """
        self.check_requirements(X, y)

        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'NoCoalescer',
            'name': 'NoCoalescer',
            'handles_sparse': True
        }
