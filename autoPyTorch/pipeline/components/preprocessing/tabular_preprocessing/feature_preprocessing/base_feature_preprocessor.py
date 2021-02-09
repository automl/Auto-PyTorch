from typing import Any, Dict, List

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.base_tabular_preprocessing import (
    autoPyTorchTabularPreprocessingComponent
)


class autoPyTorchFeaturePreprocessingComponent(autoPyTorchTabularPreprocessingComponent):
    _required_properties: List[str] = ['handles_sparse']

    def __init__(self) -> None:
        super().__init__()

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the fitted feature preprocessor into the 'X' dictionary and returns it.
        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        if self.preprocessor['numerical'] is None and self.preprocessor['categorical'] is None:
            raise AttributeError("{} can't tranform without fitting first"
                                 .format(self.__class__.__name__))
        X.update({'feature_preprocessor': self.preprocessor})
        return X
