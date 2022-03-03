from typing import Any, Dict, List, Optional

import numpy as np

from sklearn.utils import check_random_state

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.base_tabular_preprocessing import (
    autoPyTorchTabularPreprocessingComponent
)


class autoPyTorchFeaturePreprocessingComponent(autoPyTorchTabularPreprocessingComponent):
    _required_properties: List[str] = [
        'handles_sparse', 'handles_classification', 'handles_regression']

    def __init__(self, random_state: Optional[np.random.RandomState] = None):
        if random_state is None:
            # A trainer components need a random state for
            # sampling -- for example in MixUp training
            self.random_state = check_random_state(1)
        else:
            self.random_state = random_state
        super().__init__()

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the fitted feature preprocessor into the 'X' dictionary and returns it.
        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        if self.preprocessor['numerical'] is None:
            raise AttributeError("{} can't tranform without fitting first"
                                 .format(self.__class__.__name__))
        X.update({'feature_preprocessor': self.preprocessor})
        return X
