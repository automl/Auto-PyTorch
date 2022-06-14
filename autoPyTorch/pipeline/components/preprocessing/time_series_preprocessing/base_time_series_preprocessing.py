from typing import Dict, Optional, Union

from sklearn.base import BaseEstimator

from autoPyTorch.pipeline.components.preprocessing.base_preprocessing import (
    autoPyTorchPreprocessingComponent, autoPyTorchTargetPreprocessingComponent)


class autoPyTorchTimeSeriesPreprocessingComponent(autoPyTorchPreprocessingComponent):
    """
     Provides abstract interface for time series preprocessing algorithms in AutoPyTorch.
    """

    def __init__(self) -> None:
        super().__init__()
        self.preprocessor: Union[Dict[str, Optional[BaseEstimator]], BaseEstimator] = dict(
            numerical=None, categorical=None)

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.__class__.__name__
        return string


class autoPyTorchTimeSeriesTargetPreprocessingComponent(autoPyTorchTargetPreprocessingComponent):
    """
     Provides abstract interface for time series target preprocessing algorithms in AutoPyTorch.
     Currently only numerical target preprocessing is supported.
     # TODO add support for categorical targets!
     # TODO define inverse transformation for each inversible numerical transformation (log, deseasonalization, etc. )
    """
    def __init__(self) -> None:
        super().__init__()
        self.preprocessor: Union[Dict[str, Optional[BaseEstimator]], BaseEstimator] = dict(
            numerical=None, categorical=None)

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.__class__.__name__
        return string
