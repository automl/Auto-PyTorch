from typing import Any, Dict, Optional, Union

from autoPyTorch.pipeline.components.setup.forecasting_target_scaling import \
    BaseTargetScaler


class TargetMinMaxScaler(BaseTargetScaler):
    @property
    def scaler_mode(self) -> str:
        return 'min_max'

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'TargetMinMaxScaler',
            'name': 'TargetMinMaxScaler'
        }
