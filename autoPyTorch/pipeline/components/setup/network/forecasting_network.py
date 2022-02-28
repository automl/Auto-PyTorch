from collections import OrderedDict
from typing import Any, Dict, Optional, Union, Tuple, List

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition

import numpy as np

import torch
from torch import nn
import warnings

from torch.distributions import (
    AffineTransform,
    TransformedDistribution,
)

from autoPyTorch.constants import CLASSIFICATION_TASKS, STRING_TO_TASK_TYPES
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.forecasting_target_scaling. \
    base_target_scaler import BaseTargetScaler
from autoPyTorch.pipeline.components.setup.network_backbone.\
    forecasting_backbone.forecasting_encoder.base_forecasting_encoder import (
    EncoderNetwork,
    NetworkStructure,
    EncoderBlockInfo,
    NetworkStructure,
    EncoderProperties
)
from autoPyTorch.pipeline.components.setup.network_backbone.\
    forecasting_backbone.forecasting_decoder.base_forecasting_decoder import (
    DecoderBlockInfo,
    DecoderProperties
)

from autoPyTorch.utils.common import FitRequirement, get_device_from_fit_dictionary
from autoPyTorch.pipeline.components.setup.network.base_network import NetworkComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, get_hyperparameter
from autoPyTorch.pipeline.components.training.base_training import autoPyTorchTrainingComponent
from autoPyTorch.pipeline.components.setup.network.forecasting_architecture import (
    ForecastingNet,
    ForecastingSeq2SeqNet,
    ForecastingDeepARNet,
    NBEATSNet,
)


class ForecastingNetworkComponent(NetworkComponent):
    def __init__(
            self,
            network: Optional[torch.nn.Module] = None,
            random_state: Optional[np.random.RandomState] = None,
            net_out_type: str = 'regression',
            forecast_strategy: Optional[str] = 'mean',
            num_samples: Optional[int] = None,
            aggregation: Optional[str] = None,

    ) -> None:
        super(ForecastingNetworkComponent, self).__init__(network=network, random_state=random_state)
        self.net_out_type = net_out_type
        self.forecast_strategy = forecast_strategy
        self.num_samples = num_samples
        self.aggregation = aggregation

    @property
    def _required_fit_requirements(self):
        return [
            FitRequirement('dataset_properties', (Dict,), user_defined=False, dataset_property=True),
            FitRequirement('window_size', (int,), user_defined=False, dataset_property=False),
            FitRequirement('network_structure', (Dict,), user_defined=False, dataset_property=False),
            FitRequirement("network_embedding", (torch.nn.Module,), user_defined=False, dataset_property=False),
            FitRequirement("network_encoder", (Dict[str, EncoderBlockInfo]), user_defined=False,
                           dataset_property=False),
            FitRequirement("network_decoder", (Dict[str, DecoderBlockInfo]), user_defined=False,
                           dataset_property=False),
            FitRequirement("network_head", (Optional[torch.nn.Module],), user_defined=False, dataset_property=False),
            FitRequirement("target_scaler", (BaseTargetScaler,), user_defined=False, dataset_property=False),
            FitRequirement("required_net_out_put_type", (str,), user_defined=False, dataset_property=False),
            FitRequirement("encoder_properties_1", (Dict,), user_defined=False, dataset_property=False),
        ]

    def fit(self, X: Dict[str, Any], y: Any = None) -> autoPyTorchTrainingComponent:
        # Make sure that input dictionary X has the required
        # information to fit this stage
        self.check_requirements(X, y)

        if self.net_out_type != X['required_net_out_put_type']:
            raise ValueError(f"network output type must be the same as required_net_out_put_type defiend by "
                             f"loss function. However, net_out_type is {self.net_out_type} and "
                             f"required_net_out_put_type is {X['required_net_out_put_type']}")

        network_structure = X['network_structure']
        network_encoder = X['network_encoder']
        network_decoder = X['network_decoder']

        network_init_kwargs = dict(network_structure=network_structure,
                                   network_embedding=X['network_embedding'],
                                   network_encoder=network_encoder,
                                   network_decoder=network_decoder,
                                   network_head=X['network_head'],
                                   window_size=X['window_size'],
                                   dataset_properties=X['dataset_properties'],
                                   target_scaler=X['target_scaler'],
                                   output_type=self.net_out_type,
                                   forecast_strategy=self.forecast_strategy,
                                   num_samples=self.num_samples,
                                   aggregation=self.aggregation, )

        if X['decoder_properties']['recurrent']:
            # decoder is RNN or Transformer
            self.network = ForecastingSeq2SeqNet(**network_init_kwargs)
        elif X['decoder_properties']['multi_blocks']:
            self.network = NBEATSNet(**network_init_kwargs)
        elif X['auto_regressive']:
            # decoder is MLP and auto_regressive, we have deep AR model
            self.network = ForecastingDeepARNet(**network_init_kwargs)
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
            past_targets = X_batch['past_targets']
            past_features = X_batch['past_features']
            future_features = X_batch["future_features"]
            static_features = X_batch["static_features"]

            if past_targets.ndim == 2:
                past_targets = past_targets.unsqueeze(-1)

            pred_kwargs = {"past_targets": past_targets,
                           "past_features": past_features,
                           "future_features": future_features,
                           "static_features": static_features}

            for key in pred_kwargs.keys():
                if pred_kwargs[key] is not None:
                    pred_kwargs[key] = pred_kwargs[key].float()

            with torch.no_grad():
                Y_batch_pred = self.network.predict(**pred_kwargs)

            Y_batch_preds.append(Y_batch_pred.cpu())

        return torch.cat(Y_batch_preds, 0).cpu().numpy()

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
            net_out_type: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='net_out_type',
                                                                                value_range=('regression',
                                                                                             'distribution'),
                                                                                default_value='distribution'
                                                                                ),
            forecast_strategy: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='forecast_strategy',
                                                                                     value_range=('sample', 'mean'),
                                                                                     default_value='sample'),
            num_samples: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='num_samples',
                                                                               value_range=(50, 200),
                                                                               default_value=100),
            aggregation: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='aggregation',
                                                                               value_range=('mean', 'median'),
                                                                               default_value='mean')
    ) -> ConfigurationSpace:
        """
        prediction steagy
        """
        cs = ConfigurationSpace()

        net_out_type = get_hyperparameter(net_out_type, CategoricalHyperparameter)

        forecast_strategy = get_hyperparameter(forecast_strategy, CategoricalHyperparameter)
        num_samples = get_hyperparameter(num_samples, UniformIntegerHyperparameter)
        aggregation = get_hyperparameter(aggregation, CategoricalHyperparameter)

        cond_net_out_type = EqualsCondition(forecast_strategy, net_out_type, 'distribution')

        cond_num_sample = EqualsCondition(num_samples, forecast_strategy, 'sample')
        cond_aggregation = EqualsCondition(aggregation, forecast_strategy, 'sample')

        cs.add_hyperparameters([net_out_type, forecast_strategy, num_samples, aggregation])
        cs.add_conditions([cond_net_out_type, cond_aggregation, cond_num_sample])

        return cs
