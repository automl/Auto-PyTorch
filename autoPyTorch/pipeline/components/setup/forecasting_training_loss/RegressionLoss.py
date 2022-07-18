from typing import Dict, Optional, Union

from ConfigSpace import CategoricalHyperparameter, ConfigurationSpace

import numpy as np

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.forecasting_training_loss.base_forecasting_loss import \
    ForecastingLossComponents
from autoPyTorch.pipeline.components.training.losses import (
    L1Loss,
    MAPELoss,
    MASELoss,
    MSELoss
)
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class RegressionLoss(ForecastingLossComponents):
    net_output_type = 'regression'

    def __init__(self,
                 loss_name: str,
                 random_state: Optional[np.random.RandomState] = None,
                 ):
        super(RegressionLoss, self).__init__()
        if loss_name == "l1":
            self.loss = L1Loss
        elif loss_name == 'mse':
            self.loss = MSELoss
        elif loss_name == 'mase':
            self.loss = MASELoss
        elif loss_name == 'mape':
            self.loss = MAPELoss
        else:
            raise ValueError(f"Unsupported loss type {loss_name}!")
        self.random_state = random_state

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'RegressionLoss',
            'name': 'RegressionLoss',
            "handles_tabular": True,
            "handles_image": True,
            "handles_time_series": True,
            'handles_regression': True,
            'handles_classification': False
        }

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
            loss_name: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="loss_name",
                                                                             value_range=('l1', 'mse', 'mase', 'mape'),
                                                                             default_value='mse'),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        add_hyperparameter(cs, loss_name, CategoricalHyperparameter)
        return cs
