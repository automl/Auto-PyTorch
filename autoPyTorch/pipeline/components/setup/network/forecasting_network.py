from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

import torch
from torch import nn

from autoPyTorch.constants import CLASSIFICATION_TASKS, STRING_TO_TASK_TYPES
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.forecasting_target_scaling. \
    base_target_scaler import BaseTargetScaler
from autoPyTorch.utils.common import FitRequirement, get_device_from_fit_dictionary
from autoPyTorch.pipeline.components.setup.network.base_network import NetworkComponent


class ForecastingNetworkComponent(NetworkComponent):
    def __init__(
            self,
            network: Optional[torch.nn.Module] = None,
            random_state: Optional[np.random.RandomState] = None,
            auto_regressive: Optional[bool] = False,
    ) -> None:
        super(ForecastingNetworkComponent, self).__init__(network=network, random_state=random_state)
        self.auto_regressive = auto_regressive

    def predict(self, loader: torch.utils.data.DataLoader,
                target_scaler: Optional[BaseTargetScaler] = None) -> torch.Tensor:
        """
        Performs batched prediction given a loader object
        """
        assert self.network is not None
        self.network.eval()

        # Batch prediction
        Y_batch_preds = list()

        for i, (X_batch, Y_batch) in enumerate(loader):
            # Predict on batch
            X = X_batch['past_target']

            X = X.float()

            if target_scaler is None:
                loc = 0.
                scale = 1.
            else:
                X, loc, scale = target_scaler(X)

            X = X.to(self.device)

            with torch.no_grad():
                Y_batch_pred = self.network(X).mean
                if loc is not None or scale is not None:
                    if loc is None:
                        loc = 0.
                    if scale is None:
                        scale = 1.
                Y_batch_pred = Y_batch_pred.cpu() * scale + loc

            Y_batch_preds.append(Y_batch_pred.cpu())

        return torch.cat(Y_batch_preds, 0).cpu().numpy()

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
                                        **kwargs: Any
                                        ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        return cs
