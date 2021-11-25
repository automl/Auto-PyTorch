from abc import abstractmethod
from typing import Any, Callable, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
)

import numpy as np

import torch

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent
from autoPyTorch.utils.common import FitRequirement, HyperparameterSearchSpace, add_hyperparameter


class BaseNetworkInitializerComponent(autoPyTorchSetupComponent):
    """Provide an abstract interface for weight initialization
    strategies in Auto-Pytorch
    """

    def __init__(
        self,
        bias_strategy: str,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        super().__init__()
        self.bias_strategy = bias_strategy
        self.random_state = random_state
        self.add_fit_requirements([
            FitRequirement('network', (torch.nn.Module,), user_defined=False, dataset_property=False)])

    def fit(self, X: Dict[str, Any], y: Any = None) -> autoPyTorchSetupComponent:
        """
        Fits a component by using an input dictionary with pre-requisites

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API

        Returns:
            A instance of self
        """
        # Make sure that input dictionary X has the required
        # information to fit this stage
        self.check_requirements(X, y)

        X['network'].apply(self.weights_init())

        return self

    @abstractmethod
    def weights_init(self) -> Callable:
        """ A weight initialization strategy to be applied to the network. It can be a custom
        implementation, a method from torch.init or even pre-trained weights

        Returns:
            Callable: a function to apply to each module in the network
        """
        raise NotImplementedError()

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """The transform function calls the transform function of the
        underlying model and returns the transformed array.

        Args:
            X (np.ndarray): input features

        Returns:
            np.ndarray: Transformed features
        """
        return X

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        bias_strategy: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="bias_strategy",
                                                                             value_range=('Zero', 'Normal'),
                                                                             default_value='Normal')
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        # The strategy for bias initializations
        add_hyperparameter(cs, bias_strategy, CategoricalHyperparameter)
        return cs

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.__class__.__name__
        return string
