from typing import Any, Dict, Optional, Union

from autoPyTorch.pipeline.components.setup.forecasting_target_scaling import \
    BaseTargetScaler


class TargetMeanAbsScaler(BaseTargetScaler):
    @property
    def scaler_mode(self) -> str:
        return 'mean_abs'

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'TargetMeanAbsScaler',
            'name': 'TargetMeanAbsScaler'
        }
