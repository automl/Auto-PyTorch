from typing import Any, Dict, Optional, Union

from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.targets_preprocessing.forecasting_target_scaling import BaseTargetScaler


class TargetStandardScaler(BaseTargetScaler):
    @property
    def scaler_mode(self):
        return 'standard'

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'TargetStandardScaler',
            'name': 'TargetStandardScaler'
        }