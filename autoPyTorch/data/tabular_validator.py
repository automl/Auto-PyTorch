# -*- encoding: utf-8 -*-
import logging
from typing import List, Optional, Tuple, Union

import numpy as np

from scipy.sparse import issparse

from autoPyTorch.data.base_validator import BaseInputValidator
from autoPyTorch.data.tabular_feature_validator import SupportedFeatTypes, TabularFeatureValidator
from autoPyTorch.data.tabular_target_validator import SupportedTargetTypes, TabularTargetValidator
from autoPyTorch.data.utils import (
    DatasetCompressionInputType,
    DatasetCompressionSpec,
    DatasetDTypeContainerType,
    reduce_dataset_size_if_too_large
)
from autoPyTorch.utils.common import ispandas
from autoPyTorch.utils.logging_ import PicklableClientLogger, get_named_client_logger


class TabularInputValidator(BaseInputValidator):
    """
    Makes sure the input data complies with Auto-PyTorch requirements.
    Categorical inputs are encoded via an Encoder, if the input
    is a dataframe. This allow us to nicely predict string targets

    This class also perform checks for data integrity and flags the user
    via informative errors.

    Attributes:
        is_classification (bool):
            For classification task, this flag indicates that the target data
            should be encoded
        feature_validator (FeatureValidator):
            A FeatureValidator instance used to validate and encode feature columns to match
            sklearn expectations on the data
        target_validator (TargetValidator):
            A TargetValidator instance used to validate and encode (in case of classification)
            the target values
        dataset_compression (Optional[DatasetCompressionSpec]):
            specifications for dataset compression. For more info check
            documentation for `BaseTask.get_dataset`.
        feat_types (List[str]):
                Description about the feature types of the columns.
                Accepts `numerical` for integers, float data and `categorical`
                for categories, strings and bool
    """
    def __init__(
        self,
        is_classification: bool = False,
        logger_port: Optional[int] = None,
        dataset_compression: Optional[DatasetCompressionSpec] = None,
        feat_types: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.dataset_compression = dataset_compression
        self._reduced_dtype: Optional[DatasetDTypeContainerType] = None
        self.is_classification = is_classification
        self.logger_port = logger_port
        self.feat_types = feat_types
        self.seed = seed
        if self.logger_port is not None:
            self.logger: Union[logging.Logger, PicklableClientLogger] = get_named_client_logger(
                name='Validation',
                port=self.logger_port,
            )
        else:
            self.logger = logging.getLogger('Validation')

        self.feature_validator = TabularFeatureValidator(
            logger=self.logger,
            feat_types=self.feat_types)
        self.target_validator = TabularTargetValidator(
            is_classification=self.is_classification,
            logger=self.logger
        )
        self._is_fitted = False

    def _compress_dataset(
        self,
        X: DatasetCompressionInputType,
        y: SupportedTargetTypes,
    ) -> DatasetCompressionInputType:
        """
        Compress the dataset. This function ensures that
        the testing data is converted to the same dtype as
        the training data.
        See `autoPyTorch.data.utils.reduce_dataset_size_if_too_large`
        for more information.

        Args:
            X (DatasetCompressionInputType):
                features of dataset
            y (SupportedTargetTypes):
                targets of dataset
        Returns:
            DatasetCompressionInputType:
                Compressed dataset.
        """
        is_dataframe = ispandas(X)
        is_reducible_type = isinstance(X, np.ndarray) or issparse(X) or is_dataframe
        if not is_reducible_type or self.dataset_compression is None:
            return X, y
        elif self._reduced_dtype is not None:
            X = X.astype(self._reduced_dtype)
            return X, y
        else:
            X, y = reduce_dataset_size_if_too_large(
                X,
                y=y,
                is_classification=self.is_classification,
                random_state=self.seed,
                **self.dataset_compression  # type: ignore [arg-type]
            )
            self._reduced_dtype = dict(X.dtypes) if is_dataframe else X.dtype
            return X, y

    def transform(
        self,
        X: SupportedFeatTypes,
        y: Optional[SupportedTargetTypes] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        X, y = super().transform(X, y)
        X_reduced, y_reduced = self._compress_dataset(X, y)

        return X_reduced, y_reduced
