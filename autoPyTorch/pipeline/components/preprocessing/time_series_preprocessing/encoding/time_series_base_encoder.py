from typing import Any, Dict, List, Union

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.encoding.base_encoder import \
    BaseEncoder
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.base_time_series_preprocessing import \
    autoPyTorchTimeSeriesPreprocessingComponent
from autoPyTorch.utils.common import FitRequirement


class TimeSeriesBaseEncoder(autoPyTorchTimeSeriesPreprocessingComponent):
    """
    Base class for encoder
    """
    def __init__(self) -> None:
        super(TimeSeriesBaseEncoder, self).__init__()
        self.add_fit_requirements([
            FitRequirement('categorical_columns', (List,), user_defined=True, dataset_property=True),
            FitRequirement('categories', (List,), user_defined=True, dataset_property=True),
            FitRequirement('feature_names', (tuple,), user_defined=True, dataset_property=True),
            FitRequirement('feature_shapes', (Dict, ), user_defined=True, dataset_property=True),
        ])
        self.feature_shapes: Union[Dict[str, int]] = {}

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the self into the 'X' dictionary and returns it.

        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        X['dataset_properties'].update({'feature_shapes': self.feature_shapes})
        return BaseEncoder.transform(self, X)
