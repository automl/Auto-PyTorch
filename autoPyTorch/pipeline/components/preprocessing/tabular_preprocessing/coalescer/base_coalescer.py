from typing import Any, Dict, List

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.base_tabular_preprocessing import (
    autoPyTorchTabularPreprocessingComponent
)
from autoPyTorch.utils.common import FitRequirement


class BaseCoalescer(autoPyTorchTabularPreprocessingComponent):
    def __init__(self) -> None:
        super().__init__()
        self._processing = True
        self.add_fit_requirements([
            FitRequirement('categorical_columns', (List,), user_defined=True, dataset_property=True),
            FitRequirement('categories', (List,), user_defined=True, dataset_property=True)
        ])

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add the preprocessor to the provided fit dictionary `X`.

        Args:
            X (Dict[str, Any]): fit dictionary in sklearn

        Returns:
            X (Dict[str, Any]): the updated fit dictionary
        """
        if self._processing and self.preprocessor['categorical'] is None:
            # If we apply minority coalescer, we must have categorical preprocessor!
            raise RuntimeError(f"fit() must be called before transform() on {self.__class__.__name__}")

        X.update({'coalescer': self.preprocessor})
        return X
