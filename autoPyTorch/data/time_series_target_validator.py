import logging
from typing import Optional, Union


import numpy as np

import pandas as pd

from scipy.sparse import issparse

from sklearn.base import BaseEstimator


from autoPyTorch.data.base_target_validator import SupportedTargetTypes
from autoPyTorch.data.tabular_target_validator import ArrayType, TabularTargetValidator
from autoPyTorch.utils.logging_ import PicklableClientLogger


class TimeSeriesTargetValidator(TabularTargetValidator):
    def __init__(
        self,
        is_classification: bool = False,
        logger: Optional[Union[PicklableClientLogger, logging.Logger]] = None,
    ):
        if is_classification:
            raise NotImplementedError(
                "Classification is currently not supported for forecasting tasks!"
            )
        super().__init__(is_classification, logger)

    def fit(
        self,
        y_train: SupportedTargetTypes,
        y_test: Optional[SupportedTargetTypes] = None,
    ) -> BaseEstimator:
        if issparse(y_train):
            # TODO fix this
            raise NotImplementedError(
                "Sparse Target is unsupported for forecasting task!"
            )
        return super().fit(y_train, y_test)

    def transform(
        self,
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
        y_has_idx = isinstance(y, pd.DataFrame)
        if y_has_idx and index is None:
            index = y.index  # type: ignore[union-attr]
        y: ArrayType = super().transform(y)  # type: ignore[no-redef]

        if index is None:
            if not y_has_idx:
                index = np.array([0] * y.shape[0])  # type: ignore[union-attr]
        else:
            if len(index) != y.shape[0]:  # type: ignore[union-attr]
                raise ValueError("Index must have length as the input targets!")
        if y.ndim == 1:  # type: ignore[union-attr]
            y = np.expand_dims(y, -1)
        y: pd.DataFrame = pd.DataFrame(y)  # type: ignore[no-redef]
        y.index = index  # type: ignore
        return y

    @property
    def allow_missing_values(self) -> bool:
        return True
