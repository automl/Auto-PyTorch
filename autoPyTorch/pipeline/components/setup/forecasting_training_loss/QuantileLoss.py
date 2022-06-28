from functools import partial
from typing import Any, Dict, Optional, Union

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

import numpy as np

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.forecasting_training_loss.base_forecasting_loss import \
    ForecastingLossComponents
from autoPyTorch.pipeline.components.training.losses import QuantileLoss
from autoPyTorch.utils.common import (
    HyperparameterSearchSpace,
    add_hyperparameter
)


class NetworkQuantileLoss(ForecastingLossComponents):
    loss = QuantileLoss
    net_output_type = 'quantile'

    def __init__(self,
                 random_state: Optional[np.random.RandomState] = None,
                 lower_quantile: float = 0.1,
                 upper_quantile: float = 0.9,
                 ):
        super().__init__()
        self.random_state = random_state
        self.quantiles = [0.5, lower_quantile, upper_quantile]
        # To make it compatible with
        # autoPyTorch.pipeline.components.training.trainer.forecasting_trainer.forecasting_base_trainer
        self.loss = partial(QuantileLoss, quantiles=self.quantiles)  # type: ignore[assignment]

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X.update({"quantile_values": self.quantiles})
        return super().transform(X)

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'QuantileLoss',
            'name': 'QuantileLoss',
            "handles_tabular": False,
            "handles_image": False,
            "handles_time_series": True,
            'handles_regression': True,
            'handles_classification': False
        }

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
            lower_quantile: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='lower_quantile',
                                                                                  value_range=(0.0, 0.4),
                                                                                  default_value=0.1),
            upper_quantile: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='upper_quantile',
                                                                                  value_range=(0.6, 1.0),
                                                                                  default_value=0.9)

    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        add_hyperparameter(cs, lower_quantile, UniformFloatHyperparameter)
        add_hyperparameter(cs, upper_quantile, UniformFloatHyperparameter)
        return cs
