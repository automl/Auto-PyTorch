import logging
from typing import Optional, Union

import pandas as pd
import numpy as np
from scipy.sparse import issparse

from autoPyTorch.utils.logging_ import PicklableClientLogger
from autoPyTorch.data.base_target_validator import SupportedTargetTypes
from autoPyTorch.data.tabular_target_validator import TabularTargetValidator, ArrayType


class TimeSeriesTargetValidator(TabularTargetValidator):
    def __init__(self,
                 is_classification: bool = False,
                 logger: Optional[
                     Union[PicklableClientLogger, logging.Logger]
                 ] = None,
                 ):
        if is_classification:
            raise NotImplementedError("Classification is currently not supported for forecasting tasks!")
        super().__init__(is_classification, logger)

    def transform(self,
                  y: SupportedTargetTypes,
                  index: Optional[Union[pd.Index, np.ndarray]] = None,
                  ) -> pd.DataFrame:
        """
        Validates and fit a categorical encoder (if needed) to the features.
        The supported data types are List, numpy arrays and pandas DataFrames.

        Args:
            y (SupportedTargetTypes)
                A set of targets that are going to be encoded if the current task
                is classification
            index (Optional[Union[pd.Index], np.ndarray]):
                index indentifying which series the data belongs to

        Returns:
            pd.DataFrame:
                The transformed array
        """
        y: ArrayType = super().transform(y)

        if index is None:
            index = np.array([0] * y.shape[0])
        if y.ndim == 1:
            y = np.expand_dims(y, -1)
        if isinstance(y, np.ndarray):
            y: pd.DataFrame = pd.DataFrame(y)
        else:
            y: pd.DataFrame = pd.DataFrame.sparse.from_spmatrix(y)
        y.index = index
        return y

    @property
    def allow_missing_values(self) -> bool:
        return True
