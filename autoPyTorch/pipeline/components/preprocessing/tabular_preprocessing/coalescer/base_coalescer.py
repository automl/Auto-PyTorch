from typing import Any, Dict, List

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.base_tabular_preprocessing import (
    autoPyTorchTabularPreprocessingComponent
)
from autoPyTorch.utils.common import FitRequirement


class BaseCoalescer(autoPyTorchTabularPreprocessingComponent):
    """
    Base class for coalescing
    """
    def __init__(self) -> None:
        super().__init__()
        self.add_fit_requirements([
            FitRequirement('categorical_columns', (List,), user_defined=True, dataset_property=True),
            FitRequirement('categories', (List,), user_defined=True, dataset_property=True)])

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        The input X is the fit dictionary, that contains both the train data as
        well as fit directives. For example, it indicates whether or not to use the gpu
        or perform a cpu only run.

        This method add the self into the 'X' dictionary and return it.
        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        if self.preprocessor['numerical'] is None and self.preprocessor['categorical'] is None:
            raise ValueError("Cannot call transform() on {} without calling fit() first."
                             .format(self.__class__.__name__))
        X.update({'coalescer': self.preprocessor})
        return X
