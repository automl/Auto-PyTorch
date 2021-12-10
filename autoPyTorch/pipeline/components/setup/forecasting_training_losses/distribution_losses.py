from typing import Optional, Dict, Union

from ConfigSpace import ConfigurationSpace

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.forecasting_training_losses.base_forecasting_losses import \
    ForecastingLossComponents
from autoPyTorch.pipeline.components.training.losses import LogProbLoss


class DistributionLoss(ForecastingLossComponents):
    loss = LogProbLoss
    required_net_out_put_type = 'distribution'

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

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        return cs
