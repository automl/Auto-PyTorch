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
            FitRequirement('categorical_columns', (List,), user_defined=True, dataset_property=True),
            FitRequirement('categories', (List,), user_defined=True, dataset_property=True)])

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the self into the 'X' dictionary and returns it.
        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        if self.preprocessor['numerical'] is None and self.preprocessor['categorical'] is None:
            raise ValueError("cant call transform on {} without fitting first."
                             .format(self.__class__.__name__))
        X.update({'encoder': self.preprocessor})
        return X
