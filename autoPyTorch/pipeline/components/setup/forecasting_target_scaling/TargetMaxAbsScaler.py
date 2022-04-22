from typing import Any, Dict, Optional, Union

from autoPyTorch.pipeline.components.setup.forecasting_target_scaling import BaseTargetScaler


class TargetMaxAbsScaler(BaseTargetScaler):
    @property
    def scaler_mode(self):
        return 'max_abs'

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'TargetMaxAbsScaler',
            'name': 'TargetMaxAbsScaler'
        }
