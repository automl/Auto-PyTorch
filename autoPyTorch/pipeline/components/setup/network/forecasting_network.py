from typing import Any, Dict, Iterable, Optional

import numpy as np

import torch
from torch import nn

from autoPyTorch.constants import CLASSIFICATION_TASKS, STRING_TO_TASK_TYPES
from autoPyTorch.pipeline.components.setup.forecasting_target_scaling.base_target_scaler import BaseTargetScaler
from autoPyTorch.pipeline.components.setup.network.base_network import NetworkComponent
from autoPyTorch.pipeline.components.setup.network.forecasting_architecture import (
    ForecastingDeepARNet,
    ForecastingNet,
    ForecastingSeq2SeqNet,
    NBEATSNet
)
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.distribution import \
    DisForecastingStrategy
from autoPyTorch.pipeline.components.training.base_training import autoPyTorchTrainingComponent
from autoPyTorch.utils.common import (
    FitRequirement,
    get_device_from_fit_dictionary
)


class ForecastingNetworkComponent(NetworkComponent):
    def __init__(
            self,
            network: Optional[torch.nn.Module] = None,
            random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        super(ForecastingNetworkComponent, self).__init__(network=network, random_state=random_state)
        self._fit_requirements.clear()
        self.add_fit_requirements([
            FitRequirement('dataset_properties', (Dict,), user_defined=False, dataset_property=True),
            FitRequirement('window_size', (int,), user_defined=False, dataset_property=False),
            FitRequirement('network_structure', (Dict,), user_defined=False, dataset_property=False),
            FitRequirement("network_embedding", (torch.nn.Module,), user_defined=False, dataset_property=False),
            FitRequirement("network_encoder", (Dict,), user_defined=False,
                           dataset_property=False),
            FitRequirement("network_decoder", (Dict,), user_defined=False,
                           dataset_property=False),
            FitRequirement("network_head", (Optional[torch.nn.Module],), user_defined=False, dataset_property=False),
            FitRequirement("auto_regressive", (bool,), user_defined=False, dataset_property=False),
            FitRequirement("target_scaler", (BaseTargetScaler,), user_defined=False, dataset_property=False),
            FitRequirement("net_output_type", (str,), user_defined=False, dataset_property=False),
            FitRequirement("feature_names", (Iterable,), user_defined=False, dataset_property=True),
            FitRequirement("feature_shapes", (Iterable,), user_defined=False, dataset_property=True),
            FitRequirement('transform_time_features', (bool,), user_defined=False, dataset_property=False),
            FitRequirement('static_features', (tuple,), user_defined=True, dataset_property=True),
            FitRequirement('time_feature_names', (Iterable,), user_defined=True, dataset_property=True),
        ])

    def fit(self, X: Dict[str, Any], y: Any = None) -> autoPyTorchTrainingComponent:
        # Make sure that input dictionary X has the required
        # information to fit this stage
        self.check_requirements(X, y)

        network_structure = X['network_structure']
        network_encoder = X['network_encoder']
        network_decoder = X['network_decoder']

        net_output_type = X['net_output_type']

        feature_names = X['dataset_properties']['feature_names']
        feature_shapes = X['dataset_properties']['feature_shapes']
        transform_time_features = X['transform_time_features']
        known_future_features = X['dataset_properties']['known_future_features']
        if transform_time_features:
            time_feature_names = X['dataset_properties']['time_feature_names']
        else:
            time_feature_names = ()

        network_init_kwargs = dict(network_structure=network_structure,
                                   network_embedding=X['network_embedding'],
                                   network_encoder=network_encoder,
                                   network_decoder=network_decoder,
                                   temporal_fusion=X.get("temporal_fusion", None),
                                   network_head=X['network_head'],
                                   auto_regressive=X['auto_regressive'],
                                   window_size=X['window_size'],
                                   dataset_properties=X['dataset_properties'],
                                   target_scaler=X['target_scaler'],
                                   output_type=net_output_type,
                                   feature_names=feature_names,
                                   feature_shapes=feature_shapes,
                                   known_future_features=known_future_features,
                                   time_feature_names=time_feature_names,
                                   static_features=X['dataset_properties']['static_features']
                                   )
        if net_output_type == 'distribution':
            dist_forecasting_strategy = X['dist_forecasting_strategy']  # type: DisForecastingStrategy

            network_init_kwargs.update(dict(forecast_strategy=dist_forecasting_strategy.forecast_strategy,
                                            num_samples=dist_forecasting_strategy.num_samples,
                                            aggregation=dist_forecasting_strategy.aggregation, ))

        if X['auto_regressive']:
            first_decoder = next(iter(network_decoder.items()))[1]
            if first_decoder.decoder_properties.recurrent:
                self.network = ForecastingSeq2SeqNet(**network_init_kwargs)
            else:
                self.network = ForecastingDeepARNet(**network_init_kwargs)
        else:
            first_decoder = next(iter(network_decoder.items()))[1]
            if first_decoder.decoder_properties.multi_blocks:
                self.network = NBEATSNet(**network_init_kwargs)
            else:
                self.network = ForecastingNet(**network_init_kwargs)

        # Properly set the network training device
        if self.device is None:
            self.device = get_device_from_fit_dictionary(X)

        self.to(self.device)

        if STRING_TO_TASK_TYPES[X['dataset_properties']['task_type']] in CLASSIFICATION_TASKS:
            self.final_activation = nn.Softmax(dim=1)

        self.is_fitted_ = True

        return self

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
            past_targets = X_batch['past_targets']
            past_features = X_batch['past_features']
            future_features = X_batch["future_features"]
            past_observed_targets = X_batch['past_observed_targets']

            if past_targets.ndim == 2:
                past_targets = past_targets.unsqueeze(-1)

            pred_kwargs = {"past_targets": past_targets,
                           "past_features": past_features,
                           "future_features": future_features}

            for key in pred_kwargs.keys():
                if pred_kwargs[key] is not None:
                    pred_kwargs[key] = pred_kwargs[key].float()

            pred_kwargs.update({'past_observed_targets': past_observed_targets})

            with torch.no_grad():
                Y_batch_pred = self.network.predict(**pred_kwargs)

            Y_batch_preds.append(Y_batch_pred.cpu())

        return torch.cat(Y_batch_preds, 0).cpu().numpy()
