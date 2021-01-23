from typing import Dict, List, Optional, Union

from sklearn.base import BaseEstimator

from autoPyTorch.pipeline.components.preprocessing.base_preprocessing import autoPyTorchPreprocessingComponent


class autoPyTorchTabularPreprocessingComponent(autoPyTorchPreprocessingComponent):
    """
     Provides abstract interface for preprocessing algorithms in AutoPyTorch.
    """
    _required_properties: List[str] = ['handles_sparse']

    def __init__(self) -> None:
        super().__init__()
        self.preprocessor: Union[Dict[str, Optional[BaseEstimator]], BaseEstimator] = dict(
            numerical=None, categorical=None)

    def get_preprocessor_dict(self) -> Dict[str, BaseEstimator]:
        """
        Returns early_preprocessor dictionary containing the sklearn numerical
        and categorical early_preprocessor with "numerical" and "categorical"
        keys. May contain None for a key if early_preprocessor does not
        handle the datatype defined by key

        Returns:
            Dict[str, BaseEstimator]: early_preprocessor dictionary
        """
        if (self.preprocessor['numerical'] and self.preprocessor['categorical']) is None:
            raise AttributeError("{} can't return early_preprocessor dict without fitting first"
                                 .format(self.__class__.__name__))
        return self.preprocessor

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.__class__.__name__
        return string
