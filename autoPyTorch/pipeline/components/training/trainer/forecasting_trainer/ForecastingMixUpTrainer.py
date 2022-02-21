from typing import Dict, Optional, Union

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType

from autoPyTorch.pipeline.components.training.trainer.forecasting_trainer.forecasting_base_trainer import \
    ForecastingBaseTrainerComponent
from autoPyTorch.pipeline.components.training.trainer.MixUpTrainer import MixUpTrainer


class ForecastingMixUpTrainer(ForecastingBaseTrainerComponent, MixUpTrainer):
    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'ForecastingMixUpTrainer',
            'name': 'MixUp Regularized Trainer',
        }