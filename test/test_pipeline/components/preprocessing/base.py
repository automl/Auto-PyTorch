from typing import Any, Dict, List, Optional, Tuple

from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.TabularColumnTransformer import \
    TabularColumnTransformer
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.coalescer import CoalescerChoice
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.encoding import EncoderChoice
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.imputation.SimpleImputer import SimpleImputer
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.scaling import ScalerChoice
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.variance_thresholding. \
    VarianceThreshold import VarianceThreshold
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline


class TabularPipeline(TabularClassificationPipeline):
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

        default_dataset_properties = {'target_type': 'tabular_classification'}
        if dataset_properties is not None:
            default_dataset_properties.update(dataset_properties)

        steps.extend([
            ("imputer", SimpleImputer()),
            ("variance_threshold", VarianceThreshold()),
            ("coalescer", CoalescerChoice(default_dataset_properties)),
            ("encoder", EncoderChoice(default_dataset_properties)),
            ("scaler", ScalerChoice(default_dataset_properties)),
            ("tabular_transformer", TabularColumnTransformer()),
        ])
        return steps
