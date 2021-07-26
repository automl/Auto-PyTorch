from autoPyTorch.data.time_series_validator import TimeSeriesInputValidator

# -*- encoding: utf-8 -*-
import logging
import typing

from autoPyTorch.data.time_series_forecasting_feature_validator import TimeSeriesForecastingFeatureValidator
from autoPyTorch.data.time_series_forecasting_target_validator import TimeSeriesForecastingTargetValidator
from autoPyTorch.utils.logging_ import PicklableClientLogger, get_named_client_logger

# TODO create a minxin class to perform same operations on both feature and target validators

class TimeSeriesForecastingInputValidator(TimeSeriesInputValidator):
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
        super().__init__(is_classification=is_classification, logger_port=logger_port)
        self.is_classification = is_classification
        self.logger_port = logger_port
        if self.logger_port is not None:
            self.logger: typing.Union[logging.Logger, PicklableClientLogger] = get_named_client_logger(
                name='Validation',
                port=self.logger_port,
            )
        else:
            self.logger = logging.getLogger('Validation')

        self.feature_validator = TimeSeriesForecastingFeatureValidator(logger=self.logger)
        self.target_validator = TimeSeriesForecastingTargetValidator(
            is_classification=self.is_classification,
            logger=self.logger
        )

        self._is_fitted = False