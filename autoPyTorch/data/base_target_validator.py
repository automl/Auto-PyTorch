import logging
from typing import List, Optional, Union, cast

import numpy as np

import pandas as pd

from scipy.sparse import spmatrix

from sklearn.base import BaseEstimator

from autoPyTorch.utils.logging_ import PicklableClientLogger


SupportedTargetTypes = Union[List, pd.Series, pd.DataFrame, np.ndarray, spmatrix]


class BaseTargetValidator(BaseEstimator):
    """
    A class to pre-process targets. It validates the data provided during fit (to make sure
    it matches AutoPyTorch expectation) as well as encoding the targets in case of classification

    Attributes:
        is_classification (bool):
            A bool that indicates if the validator should operate in classification mode.
            During classification, the targets are encoded.
        encoder (Optional[BaseEstimator]):
            Host a encoder object if the data requires transformation (for example,
            if provided a categorical column in a pandas DataFrame)
        enc_columns (List[str])
            List of columns that where encoded
    """
    def __init__(self,
                 is_classification: bool = False,
                 logger: Optional[Union[PicklableClientLogger,
                                        logging.Logger
                                        ]
                                  ] = None,
                 ):
        self.is_classification = is_classification

        self.data_type: Optional[type] = None

        self.encoder: Optional[BaseEstimator] = None

        self.out_dimensionality: Optional[int] = None
        self.type_of_target: Optional[str] = None

        self.logger: Union[
            PicklableClientLogger, logging.Logger
        ] = logger if logger is not None else logging.getLogger(__name__)

        # Store the dtype for remapping to correct type
        self.dtype: Optional[type] = None

        self._is_fitted = False

    def fit(
        self,
        y_train: SupportedTargetTypes,
        y_test: Optional[SupportedTargetTypes] = None,
    ) -> BaseEstimator:
        """
        Validates and fit a categorical encoder (if needed) to the targets
        The supported data types are List, numpy arrays and pandas DataFrames.

        Args:
            y_train (SupportedTargetTypes)
                A set of targets set aside for training
            y_test (Union[SupportedTargetTypes])
                A hold out set of data used of the targets. It is also used to fit the
                categories of the encoder.
        """
        # Check that the data is valid
        self._check_data(y_train)

        shape = np.shape(y_train)
        if y_test is not None:
            self._check_data(y_test)

            if len(shape) != len(np.shape(y_test)) or (
                    len(shape) > 1 and (shape[1] != np.shape(y_test)[1])):
                raise ValueError("The dimensionality of the train and test targets "
                                 "does not match train({}) != test({})".format(
                                     np.shape(y_train),
                                     np.shape(y_test)
                                 ))
            if isinstance(y_train, pd.DataFrame):
                y_test = cast(pd.DataFrame, y_test)
                if y_train.columns.tolist() != y_test.columns.tolist():
                    raise ValueError(
                        "Train and test targets must both have the same columns, yet "
                        "y={} and y_test={} ".format(
                            y_train.columns,
                            y_test.columns
                        )
                    )

                if list(y_train.dtypes) != list(y_test.dtypes):
                    raise ValueError("Train and test targets must both have the same dtypes")

        if self.out_dimensionality is None:
            self.out_dimensionality = 1 if len(shape) == 1 else shape[1]
        else:
            _n_outputs = 1 if len(shape) == 1 else shape[1]
            if self.out_dimensionality != _n_outputs:
                raise ValueError('Number of outputs changed from %d to %d!' %
                                 (self.out_dimensionality, _n_outputs))

        # Fit on the training data
        self._fit(y_train, y_test)

        self._is_fitted = True

        return self

    def _fit(
        self,
        y_train: SupportedTargetTypes,
        y_test: Optional[SupportedTargetTypes] = None,
    ) -> BaseEstimator:
        """
        Args:
            y_train (SupportedTargetTypes)
                The labels of the current task. They are going to be encoded in case
                of classification
            y_test (Optional[SupportedTargetTypes])
                A holdout set of labels
        """
        raise NotImplementedError()

    def transform(
        self,
        y: Union[SupportedTargetTypes],
    ) -> np.ndarray:
        """
        Args:
            y (SupportedTargetTypes)
                A set of targets that are going to be encoded if the current task
                is classification
        Returns:
            np.ndarray:
                The transformed array
        """
        raise NotImplementedError()

    def inverse_transform(
        self,
        y: SupportedTargetTypes,
    ) -> np.ndarray:
        """
        Revert any encoding transformation done on a target array

        Args:
            y (Union[np.ndarray, pd.DataFrame, pd.Series]):
                Target array to be transformed back to original form before encoding

        Returns:
            np.ndarray:
                The transformed array
        """
        raise NotImplementedError()

    @property
    def classes_(self) -> np.ndarray:
        """
        Complies with scikit learn classes_ attribute,
        which consist of a ndarray of shape (n_classes,)
        where n_classes are the number of classes seen while fitting
        a encoder to the targets.

        Returns:
            classes_: np.ndarray
                The unique classes seen during encoding of a classifier
        """
        if self.encoder is None:
            return np.array([])
        else:
            return self.encoder.categories_[0]

    def is_single_column_target(self) -> bool:
        """
        Output is encoded with a single column encoding
        """
        return self.out_dimensionality == 1

    @property
    def allow_missing_values(self) -> bool:
        return False
