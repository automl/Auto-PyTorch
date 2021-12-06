from typing import Any, Dict, Optional, Union

from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.\
    forecasting_target_scaling.base_target_scaler import BaseTargetScaler


class TargetNoScaler(BaseTargetScaler):
    @property
    def scaler_mode(self):
        return 'none'

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'TargetNoScaler',
            'name': 'TargetNoScaler'
        }
