from typing import Any, Callable, Dict, Optional

from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent
from autoPyTorch.utils.common import FitRequirement


class ForecastingLossComponents(autoPyTorchComponent):
    _required_properties = ["name", "handles_tabular", "handles_image", "handles_time_series",
                            'handles_regression', 'handles_classification']
    loss: Optional[Callable] = None
    net_output_type: Optional[str] = None

    def __init__(self,
                 **kwargs: Any):
        super().__init__()
        self.add_fit_requirements([
            FitRequirement('task_type', (str,), user_defined=True, dataset_property=True),
        ])

    def fit(self, X: Dict[str, Any], y: Any = None) -> "autoPyTorchComponent":
        self.check_requirements(X, y)
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X.update({"loss": self.loss,
                  'net_output_type': self.net_output_type})
        return X
