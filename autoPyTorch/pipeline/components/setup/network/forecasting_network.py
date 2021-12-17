from typing import Any, Dict, Optional, Union, Tuple, List

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition

import numpy as np

import torch
from torch import nn

from autoPyTorch.constants import CLASSIFICATION_TASKS, STRING_TO_TASK_TYPES
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.forecasting_target_scaling. \
    base_target_scaler import BaseTargetScaler
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_encoder.base_forecasting_encoder \
    import EncoderNetwork
from autoPyTorch.utils.common import FitRequirement, get_device_from_fit_dictionary
from autoPyTorch.pipeline.components.setup.network.base_network import NetworkComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter
from autoPyTorch.pipeline.components.training.base_training import autoPyTorchTrainingComponent


class ForecastingNet(nn.Module):
    future_target_required = False

    def __init__(self,
                 network_embedding: nn.Module,  # TODO consider  embedding for past, future and static features
                 network_encoder: EncoderNetwork,
                 network_decoder: nn.Module,
                 network_head: nn.Module,
                 n_prediction_steps: int,
                 encoder_properties: Dict,
                 decoder_properties: Dict,
                 output_type: str = 'regression',
                 forecast_strategy: Optional[str] = 'mean',
                 num_samples: Optional[int] = 100,
                 aggregation: Optional[str] = 'mean'
                 ):
        """
        This is a basic forecasting network. It is only composed of a embedding net, an encoder and a head (including
        MLP decoder and the final head).

        This structure is active when the decoder is a MLP with auto_regressive set as false

        Args:
            network_embedding (nn.Module): network embedding
            network_encoder (EncoderNetwork): Encoder network, could be selected to return a sequence or a
            network_decoder (nn.Module): network decoder
            network_head (nn.Module): network head, maps the output of decoder to the final output
            n_prediction_steps (int): how many steps the network want to predict
            encoder_properties (Dict): encoder properties
            decoder_properties: (Dict): decoder properties
            output_type (str): the form that the network outputs. It could be regression, distribution and
            (TODO) quantile
            forecast_strategy (str): only valid if output_type is distribution or quantile, how the network transforms
            its output to predicted values, could be mean or sample
            num_samples (int): only valid if output_type is not regression and forecast_strategy is sample. this
            indicates the number of the points to sample when doing prediction
            aggregation (str): how the samples are aggregated. We could take their mean or median values.
        """
        super(ForecastingNet, self).__init__()
        self.embedding = network_embedding
        self.encoder = network_encoder  # type:EncoderNetwork
        self.decoder = network_decoder
        self.head = network_head

        self.n_prediction_steps = n_prediction_steps
        self.output_type = output_type
        self.forecast_strategy = forecast_strategy
        self.num_samples = num_samples
        self.aggregation = aggregation

        if decoder_properties['has_hidden_states']:
            if not encoder_properties['has_hidden_states']:
                raise ValueError('when decoder contains hidden states, encoder must provide the hidden states '
                                 'for decoder!')
        self.encoder_has_hidden_states = encoder_properties['has_hidden_states']
        self.decoder_has_hidden_states = decoder_properties['has_hidden_states']

    def forward(self,
                targets_past: torch.Tensor,
                targets_future: Optional[torch.Tensor] = None,
                features_past: Optional[torch.Tensor] = None,
                features_future: Optional[torch.Tensor] = None,
                features_static: Optional[torch.Tensor] = None,
                hidden_states: Optional[Tuple[torch.Tensor]] = None):
        if features_past is not None:
            x_past = torch.cat([targets_past, features_past], dim=1)
        else:
            x_past = targets_past
        x_past = self.embedding(x_past)
        if self.encoder_has_hidden_states:
            x_past, _ = self.encoder(x_past)
        else:
            x_past = self.encoder(x_past)
        x_past = self.decoder(x_past)
        output = self.head(x_past)
        return output

    def pred_from_net_output(self, net_output):
        if self.output_type == 'regression':
            return net_output
        elif self.output_type == 'distribution':
            if self.forecast_strategy == 'mean':
                if isinstance(net_output, list):
                    return torch.cat([dist.mean for dist in net_output], dim=-2)
                else:
                    return net_output.mean
            elif self.forecast_strategy == 'sample':
                if isinstance(net_output, list):
                    samples = torch.cat([dist.sample((self.num_samples,)) for dist in net_output], dim=-2)
                else:
                    samples = net_output.sample((self.num_samples,))
                if self.aggregation == 'mean':
                    return torch.mean(samples, dim=0)
                elif self.aggregation == 'median':
                    return torch.median(samples, 0)[0]
                else:
                    raise ValueError(f'Unknown aggregation: {self.aggregation}')
            else:
                raise ValueError(f'Unknown forecast_strategy: {self.forecast_strategy}')
        else:
            raise ValueError(f'Unknown output_type: {self.output_type}')

    def predict(self,
                targets_past: torch.Tensor,
                features_past: Optional[torch.Tensor] = None,
                features_future: Optional[torch.Tensor] = None,
                features_static: Optional[torch.Tensor] = None
                ):
        net_output = self(targets_past, features_past)
        return self.pred_from_net_output(net_output)


class ForecastingSeq2SeqNet(ForecastingNet):
    future_target_required = True
    """
    Forecasting network with Seq2Seq structure.

    This structure is activate when the decoder is recurrent (RNN). We train the network with teacher forcing, thus
    targets_future is required for the network. To train the network, past targets and past features are fed to the
    encoder to obtain the hidden states whereas future targets and future features
    """

    def forward(self,
                targets_past: torch.Tensor,
                targets_future: Optional[torch.Tensor] = None,
                features_past: Optional[torch.Tensor] = None,
                features_future: Optional[torch.Tensor] = None,
                features_static: Optional[torch.Tensor] = None,
                hidden_states: Optional[Tuple[torch.Tensor]] = None):
        x_past = targets_past if features_past is None else torch.cat([targets_past, features_past], dim=-1)

        x_past = self.embedding(x_past)

        if self.training:
            # we do one step ahead forecasting
            targets_future = torch.cat([targets_past[:, [-1], :], targets_future[:, :-1, :]], dim=1)

            x_future = targets_future if features_future is None else torch.cat([targets_future, features_future],
                                                                                dim=-1)

            _, hidden_states = self.encoder(x_past)
            x_future, _ = self.decoder(x_future, hidden_states)
            net_output = self.head(x_future)

            return net_output
        else:
            all_predictions = []
            predicted_target = targets_past[:, [-1]]

            _, hidden_states = self.encoder(x_past)
            for idx_pred in range(self.n_prediction_steps):
                x_future = predicted_target if features_future is None else torch.cat(
                    [predicted_target, features_future[:, [idx_pred], :]],
                    dim=-1)

                x_future, hidden_states = self.decoder(x_future, hx=hidden_states)
                net_output = self.head(x_future[:, -1:, ])
                predicted_target = self.pred_from_net_output(net_output).to(targets_past.device)

                all_predictions.append(net_output)

            if self.output_type != 'distribution':
                all_predictions = torch.cat(all_predictions, dim=1)

            return all_predictions

    def predict(self,
                targets_past: torch.Tensor,
                features_past: Optional[torch.Tensor] = None,
                features_future: Optional[torch.Tensor] = None,
                features_static: Optional[torch.Tensor] = None
                ):
        net_output = self(targets_past, features_past, features_future)
        return self.pred_from_net_output(net_output)


class ForecastingDeepARNet(ForecastingNet):
    future_target_required = True

    def __init__(self,
                 **kwargs):
        """
        Forecasting network with DeepAR structure.

        This structure is activate when the decoder is not recurrent (MLP) and its hyperparameter "auto_regressive" is
        set  as True. We train the network to let it do a one-step prediction. This structure is compatible with any
         sorts of encoder (except MLP).
        """
        super(ForecastingDeepARNet, self).__init__(**kwargs)
        # this determines the training targets
        self.encoder_bijective_seq_output = kwargs['encoder_properties']['bijective_seq_output']

    def forward(self,
                targets_past: torch.Tensor,
                targets_future: Optional[torch.Tensor] = None,
                features_past: Optional[torch.Tensor] = None,
                features_future: Optional[torch.Tensor] = None,
                features_static: Optional[torch.Tensor] = None,
                hidden_states: Optional[Tuple[torch.Tensor]] = None):
        x_past = targets_past if features_past is None else torch.cat([targets_past, features_past], dim=-1)

        # TODO consider static features
        x_past = self.embedding(x_past)

        if self.training:
            x_future = targets_future if features_future is None else torch.cat([targets_future, features_future],
                                                                                dim=-1)
            x_future = self.embedding(x_future)

            x_input = torch.cat([x_past, x_future[:, :-1]], dim=1)

            if self.encoder_has_hidden_states:
                x_input, _ = self.encoder(x_input, output_seq=True)
            else:
                x_input = self.encoder(x_input, output_seq=True)

            net_output = self.head(self.decoder(x_input))
            return net_output
        else:
            all_predictions = []
            batch_size = targets_past.shape[0]

            if self.encoder_has_hidden_states:
                # For RNN, we only feed the hidden state and generated future input to the netwrok
                encoder_output, hidden_states = self.encoder(x_past)
                repeated_state = [
                    s.repeat_interleave(repeats=self.num_samples, dim=1)
                    for s in hidden_states
                ]

            else:
                # For other models, the full past targets are passed to the network.
                encoder_output = self.encoder(x_past)
                repeated_past_target = targets_past.repeat_interleave(repeats=self.num_samples, dim=0).squeeze(1)

            repeated_static_feat = features_static.repeat_interleave(
                repeats=self.num_samples, dim=0
            ).unsqueeze(dim=1) if features_static is not None else None

            repeated_time_feat = features_future.repeat_interleave(
                repeats=self.num_samples, dim=0
            ) if features_future is not None else None

            net_output = self.head(self.decoder(encoder_output))

            next_sample = net_output.sample(sample_shape=(self.num_samples,))

            next_sample = next_sample.transpose(0, 1).reshape(
                (next_sample.shape[0] * next_sample.shape[1], 1, -1)
            )

            all_predictions.append(next_sample)

            for k in range(1, self.n_prediction_steps):
                x_next = next_sample if repeated_time_feat is None else torch.cat([next_sample,
                                                                                   repeated_time_feat[:, k:k + 1]],
                                                                                  dim=-1)
                if self.encoder_has_hidden_states:
                    encoder_output, repeated_state = self.encoder(x_next, hx=repeated_state)
                else:
                    x_next = torch.cat([repeated_past_target, x_next], dim=1)
                    encoder_output = self.encoder(x_next)
                # for training, the encoder output a sequence. Thus for prediction, the network should have the same
                # format output
                encoder_output = torch.unsqueeze(encoder_output, 1)

                net_output = self.head(self.decoder(encoder_output))

                next_sample = net_output.sample()
                all_predictions.append(next_sample)

            all_predictions = torch.cat(all_predictions, dim=1).unflatten(0, (batch_size, self.num_samples))

            return all_predictions

    def pred_from_net_output(self, net_output: torch.Tensor):
        if not self.output_type == 'distribution' and self.forecast_strategy == 'sample':
            raise ValueError(f"A DeepAR network must have output type as Distribution and forecast_strategy as sample,"
                             f"but this network has {self.output_type} and {self.forecast_strategy}")
        if self.aggregation == 'mean':
            return torch.mean(net_output, dim=1)
        elif self.aggregation == 'median':
            return torch.median(net_output, dim=1)[0]
        else:
            raise ValueError(f'Unknown aggregation: {self.aggregation}')


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
            FitRequirement("network_embedding", (torch.nn.Module,), user_defined=False, dataset_property=False),
            FitRequirement("network_encoder", (torch.nn.Module,), user_defined=False, dataset_property=False),
            FitRequirement("network_decoder", (torch.nn.Module,), user_defined=False, dataset_property=False),
            FitRequirement("network_head", (torch.nn.Module,), user_defined=False, dataset_property=False),
            FitRequirement("required_net_out_put_type", (str,), user_defined=False, dataset_property=False),
            FitRequirement("encoder_properties", (Dict,), user_defined=False, dataset_property=False),
            FitRequirement("decoder_properties", (Dict,), user_defined=False, dataset_property=False),
        ]

    def fit(self, X: Dict[str, Any], y: Any = None) -> autoPyTorchTrainingComponent:
        # Make sure that input dictionary X has the required
        # information to fit this stage
        self.check_requirements(X, y)

        if self.net_out_type != X['required_net_out_put_type']:
            raise ValueError(f"network output type must be the same as required_net_out_put_type defiend by "
                             f"loss function. However, net_out_type is {self.net_out_type} and "
                             f"required_net_out_put_type is {X['required_net_out_put_type']}")

        network_init_kwargs = dict(network_embedding=X['network_embedding'],
                                   network_encoder=X['network_encoder'],
                                   network_decoder=X['network_decoder'],
                                   network_head=X['network_head'],
                                   n_prediction_steps=X['dataset_properties']['n_prediction_steps'],
                                   encoder_properties=X['encoder_properties'],
                                   decoder_properties=X['decoder_properties'],
                                   output_type=self.net_out_type,
                                   forecast_strategy=self.forecast_strategy,
                                   num_samples=self.num_samples,
                                   aggregation=self.aggregation, )

        if X['decoder_properties']['has_hidden_states']:
            # decoder is RNN
            self.network = ForecastingSeq2SeqNet(**network_init_kwargs)
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
            X = X_batch['past_target']

            X = X.float()

            if target_scaler is None:
                loc = 0.
                scale = 1.
            else:
                X, _, loc, scale = target_scaler(X)

            X = X.to(self.device)

            with torch.no_grad():
                Y_batch_pred = self.network.predict(X)
                if loc is None:
                    loc = 0.
                if scale is None:
                    scale = 1.
                Y_batch_pred = Y_batch_pred.cpu() * scale + loc

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
