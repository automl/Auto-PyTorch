from typing import Any, Dict, List

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.base_tabular_preprocessing import (
    autoPyTorchTabularPreprocessingComponent
)
from autoPyTorch.utils.common import FitRequirement


class BaseEncoder(autoPyTorchTabularPreprocessingComponent):
    """
    Base class for encoder
    """
    def __init__(self) -> None:
        super().__init__()
        self.add_fit_requirements([
            FitRequirement('encode_columns', (List,), user_defined=True, dataset_property=False)])

    @staticmethod
    def _has_encode_columns(X: Dict[str, Any]):
        return len(X.get('encode_columns', [])) > 0

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the self into the 'X' dictionary and returns it.
        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        X.update({'encoder': self.preprocessor})
        return X
