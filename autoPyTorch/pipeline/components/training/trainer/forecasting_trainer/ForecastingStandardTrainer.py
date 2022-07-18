from typing import Dict, Optional, Union

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.training.trainer.StandardTrainer import StandardTrainer
from autoPyTorch.pipeline.components.training.trainer.forecasting_trainer.forecasting_base_trainer import \
    ForecastingBaseTrainerComponent


class ForecastingStandardTrainer(ForecastingBaseTrainerComponent, StandardTrainer):
    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'ForecastingStandardTrainer',
            'name': 'Forecasting Standard Trainer',
        }
