from typing import Any, Dict, Optional, Union

from ConfigSpace import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter

import numpy as np


from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.forecasting_training_loss.base_forecasting_loss import (
    ForecastingLossComponents
)
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.distribution import (
    ALL_DISTRIBUTIONS,
    DisForecastingStrategy
)
from autoPyTorch.pipeline.components.training.losses import LogProbLoss
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


class DistributionLoss(ForecastingLossComponents):
    loss = LogProbLoss
    net_output_type = 'distribution'  # type: ignore[assignment]

    def __init__(self,
                 dist_cls: str,
                 random_state: Optional[np.random.RandomState] = None,
                 forecast_strategy: str = "sample",
                 num_samples: int = 100,
                 aggregation: str = "mean",
                 ):
        super(DistributionLoss, self).__init__()
        self.dist_cls = dist_cls
        self.random_state = random_state
        self.forecasting_strategy = DisForecastingStrategy(dist_cls=dist_cls,
                                                           forecast_strategy=forecast_strategy,
                                                           num_samples=num_samples,
                                                           aggregation=aggregation)

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'DistributionLoss',
            'name': 'DistributionLoss',
            "handles_tabular": False,
            "handles_image": False,
            "handles_time_series": True,
            'handles_regression': True,
            'handles_classification': False
        }

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        required_padding_value = ALL_DISTRIBUTIONS[self.dist_cls].value_in_support
        X.update({"required_padding_value": required_padding_value,
                  "dist_forecasting_strategy": self.forecasting_strategy})
        return super().transform(X)

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
            dist_cls: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="dist_cls",
                value_range=tuple(ALL_DISTRIBUTIONS.keys()),
                default_value=list(ALL_DISTRIBUTIONS.keys())[0]),
            forecast_strategy: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='forecast_strategy',
                                                                                     value_range=('sample', 'mean'),
                                                                                     default_value='sample'),
            num_samples: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='num_samples',
                                                                               value_range=(50, 200),
                                                                               default_value=100),
            aggregation: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='aggregation',
                                                                               value_range=('mean', 'median'),
                                                                               default_value='mean')
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        add_hyperparameter(cs, dist_cls, CategoricalHyperparameter)

        forecast_strategy = get_hyperparameter(forecast_strategy, CategoricalHyperparameter)
        num_samples = get_hyperparameter(num_samples, UniformIntegerHyperparameter)
        aggregation = get_hyperparameter(aggregation, CategoricalHyperparameter)

        cs.add_hyperparameters([forecast_strategy, num_samples, aggregation])

        cond_n_samples = EqualsCondition(num_samples, forecast_strategy, 'sample')
        cond_agg = EqualsCondition(aggregation, forecast_strategy, 'sample')
        cs.add_conditions([cond_n_samples, cond_agg])
        return cs
