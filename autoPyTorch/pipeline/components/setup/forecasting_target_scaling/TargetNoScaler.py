from typing import Any, Dict, Optional, Union

from autoPyTorch.pipeline.components.setup.forecasting_target_scaling import BaseTargetScaler


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
