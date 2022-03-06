from typing import Optional, Dict, Union, Any
import numpy as np

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.distribution import ALL_DISTRIBUTIONS
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.forecasting_training_loss.base_forecasting_loss import \
    ForecastingLossComponents
from autoPyTorch.pipeline.components.training.losses import QuantileLoss
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter, FitRequirement


class NetworkQuantileLoss(ForecastingLossComponents):
    loss = QuantileLoss
    required_net_out_put_type = 'quantile'

    def __init__(self,
                 random_state: Optional[np.random.RandomState] = None,
                 lower_quantile: float=0.1,
                 upper_quantile: float=0.9,
                 ):
        super(QuantileLoss, self).__init__()
        self.random_state = random_state
        self.loss = QuantileLoss(lower=lower_quantile, upper=upper_quantile)

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'QuantileLoss',
            'name': 'QuantileLoss',
            "handles_tabular": False,
            "handles_image": False,
            "handles_time_series": True,
            'handles_regression': False,
            'handles_classification': False
        }

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        required_padding_value = ALL_DISTRIBUTIONS[self.dist_cls].value_in_support
        X.update({"dist_cls": self.dist_cls,
                  "required_padding_value": required_padding_value})
        return super().transform(X)

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
