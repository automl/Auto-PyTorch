from typing import Any, Dict, List, Optional, Tuple

from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.TimeSeriesTransformer import (
    TimeSeriesTransformer, TimeSeriesTargetTransformer
)
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.encoding import (
    TimeSeriesEncoderChoice
)
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.imputation.TimeSeriesImputer import (
    TimeSeriesFeatureImputer,
    TimeSeriesTargetImputer,
)
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.scaling.base_scaler import BaseScaler
from autoPyTorch.pipeline.time_series_forecasting import TimeSeriesForecastingPipeline


class ForecastingPipeline(TimeSeriesForecastingPipeline):
    def _get_pipeline_steps(self, dataset_properties: Optional[Dict[str, Any]],
                            ) -> List[Tuple[str, autoPyTorchChoice]]:
        """
        Defines what steps a pipeline should follow.
        The step itself has choices given via autoPyTorchChoice.

        Returns:
            List[Tuple[str, autoPyTorchChoice]]: list of steps sequentially exercised
                by the pipeline.
        """
        steps: List[Tuple[str, autoPyTorchChoice]] = []

        default_dataset_properties = {'target_type': 'time_series_forecasting'}
        if dataset_properties is not None:
            default_dataset_properties.update(dataset_properties)

        steps.extend([("imputer", TimeSeriesFeatureImputer(random_state=self.random_state)),
                      ("scaler", BaseScaler(random_state=self.random_state)),
                      ('encoding', TimeSeriesEncoderChoice(default_dataset_properties,
                                                           random_state=self.random_state)),
                      ("time_series_transformer", TimeSeriesTransformer(random_state=self.random_state)),
                      ("target_imputer", TimeSeriesTargetImputer(random_state=self.random_state)),
                      ])
        return steps
