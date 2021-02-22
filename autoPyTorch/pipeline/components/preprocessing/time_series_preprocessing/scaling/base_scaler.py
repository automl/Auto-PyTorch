from typing import Any, Dict, List

from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.base_time_series_preprocessing import (
    autoPyTorchTimeSeriesPreprocessingComponent
)
from autoPyTorch.utils.common import FitRequirement


class BaseScaler(autoPyTorchTimeSeriesPreprocessingComponent):
    """
    Provides abstract class interface for time series scalers in AutoPytorch
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_fit_requirements([
            FitRequirement('numerical_features', (List,), user_defined=True, dataset_property=True)])

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the fitted scalar into the 'X' dictionary and returns it.
        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        if self.preprocessor['numerical'] is None and self.preprocessor['categorical'] is None:
            raise ValueError(f"can not call transform on {self.__class__.__name__} without fitting first.")
        X.update({'scaler': self.preprocessor})
        return X
