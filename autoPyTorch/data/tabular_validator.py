# -*- encoding: utf-8 -*-
import logging
from typing import Optional, Union

from autoPyTorch.data.base_validator import BaseInputValidator
from autoPyTorch.data.tabular_feature_validator import TabularFeatureValidator
from autoPyTorch.data.tabular_target_validator import TabularTargetValidator
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
    """
    def __init__(
        self,
        is_classification: bool = False,
        logger_port: Optional[int] = None,
    ) -> None:
        self.is_classification = is_classification
        self.logger_port = logger_port
        if self.logger_port is not None:
            self.logger: Union[logging.Logger, PicklableClientLogger] = get_named_client_logger(
                name='Validation',
                port=self.logger_port,
            )
        else:
            self.logger = logging.getLogger('Validation')

        self.feature_validator = TabularFeatureValidator(logger=self.logger)
        self.target_validator = TabularTargetValidator(
            is_classification=self.is_classification,
            logger=self.logger
        )
        self._is_fitted = False
