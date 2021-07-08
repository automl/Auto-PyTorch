# -*- encoding: utf-8 -*-
import logging
import typing

from sklearn.base import BaseEstimator

from autoPyTorch.data.base_feature_validator import SUPPORTED_FEAT_TYPES
from autoPyTorch.data.base_target_validator import SUPPORTED_TARGET_TYPES

from autoPyTorch.data.base_validator import BaseInputValidator
from autoPyTorch.data.time_series_feature_validator import TimeSeriesFeatureValidator
from autoPyTorch.data.time_series_target_validator import TimeSeriesTargetValidator
from autoPyTorch.utils.logging_ import PicklableClientLogger, get_named_client_logger


class TimeSeriesInputValidator(BaseInputValidator):
    """
    Makes sure the input data complies with Auto-PyTorch requirements.

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
    """

    def __init__(
        self,
        is_classification: bool = False,
        logger_port: typing.Optional[int] = None,
    ) -> None:
        self.is_classification = is_classification
        self.logger_port = logger_port
        if self.logger_port is not None:
            self.logger: typing.Union[logging.Logger, PicklableClientLogger] = get_named_client_logger(
                name='Validation',
                port=self.logger_port,
            )
        else:
            self.logger = logging.getLogger('Validation')

        self.feature_validator = TimeSeriesFeatureValidator(logger=self.logger)
        self.target_validator = TimeSeriesTargetValidator(
            is_classification=self.is_classification,
            logger=self.logger
        )

        self._is_fitted = False



