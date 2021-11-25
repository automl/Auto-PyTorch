from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

import torch
from torch import nn

from autoPyTorch.constants import CLASSIFICATION_TASKS, STRING_TO_TASK_TYPES
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.training.base_training import autoPyTorchTrainingComponent
from autoPyTorch.utils.common import FitRequirement, get_device_from_fit_dictionary


class NetworkComponent(autoPyTorchTrainingComponent):
    """
    Provide an abstract interface for networks
    in Auto-Pytorch
    """

    def __init__(
            self,
            network: Optional[torch.nn.Module] = None,
            random_state: Optional[np.random.RandomState] = None
    ) -> None:
        super(NetworkComponent, self).__init__()
        self.random_state = random_state
        self.device = None
        self.add_fit_requirements([
            FitRequirement("network_head", (torch.nn.Module,), user_defined=False, dataset_property=False),
            FitRequirement("network_backbone", (torch.nn.Module,), user_defined=False, dataset_property=False),
            FitRequirement("network_embedding", (torch.nn.Module,), user_defined=False, dataset_property=False),
        ])
        self.network = network
        self.final_activation: Optional[torch.nn.Module] = None

    def fit(self, X: Dict[str, Any], y: Any = None) -> autoPyTorchTrainingComponent:
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

        self.network = torch.nn.Sequential(X['network_embedding'], X['network_backbone'], X['network_head'])

        # Properly set the network training device
        if self.device is None:
            self.device = get_device_from_fit_dictionary(X)

        self.to(self.device)

        if STRING_TO_TASK_TYPES[X['dataset_properties']['task_type']] in CLASSIFICATION_TASKS:
            self.final_activation = nn.Softmax(dim=1)

        self.is_fitted_ = True

        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        The transform function updates the network in the X dictionary.
        """
        X.update({'network': self.network})
        return X

    def get_network(self) -> nn.Module:
        """
        Return the underlying network object.
        Returns:
            model : the underlying network object
        """
        assert self.network is not None, "No network was initialized"
        return self.network

    def check_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
        """
        This common utility makes sure that the input dictionary X,
        used to fit a given component class, contains the minimum information
        to fit the given component, and it's parents
        """

        # Honor the parent requirements
        super().check_requirements(X, y)

    def get_network_weights(self) -> torch.nn.parameter.Parameter:
        """Returns the weights of the network"""
        assert self.network is not None, "No network was initialized"
        return self.network.parameters()

    def to(self, device: Optional[torch.device] = None) -> None:
        """Setups the network in cpu or gpu"""
        assert self.network is not None, "No network was initialized"
        if device is not None:
            self.network = self.network.to(device)
        else:
            self.network = self.network.to(self.device)

    def predict(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Performs batched prediction given a loader object
        """
        assert self.network is not None
        self.network.eval()

        # Batch prediction
        Y_batch_preds = list()

        for i, (X_batch, Y_batch) in enumerate(loader):
            # Predict on batch
            X_batch = X_batch.float().to(self.device)

            with torch.no_grad():
                Y_batch_pred = self.network(X_batch)
                if self.final_activation is not None:
                    Y_batch_pred = self.final_activation(Y_batch_pred)

            Y_batch_preds.append(Y_batch_pred.cpu())

        return torch.cat(Y_batch_preds, 0).cpu().numpy()

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
                                        **kwargs: Any
                                        ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'nn.Sequential',
            'name': 'torch.nn.Sequential',
        }

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        network_name: str = self.network.__class__.__name__
        info = vars(self)
        # Remove unwanted info
        network_name += " (" + str(info) + ")"
        return network_name
