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
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.base_forecasting_encoder \
    import EncoderNetwork
from autoPyTorch.utils.common import FitRequirement, get_device_from_fit_dictionary
from autoPyTorch.pipeline.components.setup.network.base_network import NetworkComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, get_hyperparameter
from autoPyTorch.pipeline.components.training.base_training import autoPyTorchTrainingComponent


class TransformedDistribution_(TransformedDistribution):
    """
    We implement the mean function such that we do not need to enquire base mean every time
    """

    @property
    def mean(self):
        mean = self.base_dist.mean
        for transform in self.transforms:
            mean = transform(mean)
        return mean


def get_lagged_subsequences(
        sequence: torch.Tensor,
        subsequences_length: int,
        lags_seq: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns lagged subsequences of a given sequence, this allows the model to receive the input from the past targets
    outside the sliding windows. This implementation is similar to gluonTS's implementation
     the only difference is that we pad the sequence that is not long enough

    Parameters
    ----------
    sequence : Tensor
        the sequence from which lagged subsequences should be extracted.
        Shape: (N, T, C).
    subsequences_length : int
        length of the subsequences to be extracted.
    lags_seq: Optional[List[int]]
        lags of the sequence, indicating the sequence that needs to be extracted
    lag_mask: Optional[torch.Tensor]
        a mask tensor indicating

    Returns
    --------
    lagged : Tensor
        a tensor of shape (N, S, I * C), where S = subsequences_length and
        I = len(indices), containing lagged subsequences.
    """
    batch_size = sequence.shape[0]
    num_features = sequence.shape[2]
    if mask is None:
        if lags_seq is None:
            warnings.warn('Neither lag_mask or lags_seq is given, we simply return the input value')
            return sequence, None
        # generate mask
        num_lags = len(lags_seq)

        # build a mask
        mask_length = max(lags_seq) + subsequences_length
        mask = torch.zeros((num_lags, mask_length), dtype=torch.bool)
        for i, lag_index in enumerate(lags_seq):
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            mask[i, begin_index: end_index] = True
    else:
        num_lags = mask.shape[0]
        mask_length = mask.shape[1]

    mask_extend = mask.clone()

    if mask_length > sequence.shape[1]:
        sequence = torch.cat([sequence.new_zeros([batch_size, mask_length - sequence.shape[1], num_features]),
                              sequence], dim=1)
    elif mask_length < sequence.shape[1]:
        mask_extend = torch.cat([mask.new_zeros([num_lags, sequence.shape[1] - mask_length]), mask_extend], dim=1)
    #  (N, 1, T, C)
    sequence = sequence.unsqueeze(1)

    # (I, T, 1)
    mask_extend = mask_extend.unsqueeze(-1)

    # (N, I, S, C)
    lagged_seq = torch.masked_select(sequence, mask_extend).reshape(batch_size, num_lags, subsequences_length, -1)

    lagged_seq = torch.transpose(lagged_seq, 1, 2).reshape(batch_size, subsequences_length, -1)

    return lagged_seq, mask


def get_lagged_subsequences_inference(
        sequence: torch.Tensor,
        subsequences_length: int,
        lags_seq: Optional[List[int]] = None, ):
    """
    this function works exactly the same as get_lagged_subsequences. However, this implementation is faster when no
    cached value is available, thus it more suitable during inference times.

    designed for doing inference for DeepAR, the core idea is to use
    """
    sequence_length = sequence.shape[1]
    batch_size = sequence.shape[0]
    lagged_values = []
    for lag_index in lags_seq:
        begin_index = -lag_index - subsequences_length
        end_index = -lag_index if lag_index > 0 else None
        if end_index is not None and end_index < -sequence_length:
            lagged_values.append(torch.zeros([batch_size, subsequences_length, *sequence.shape[2:]]))
            continue
        if begin_index < -sequence_length:
            if end_index is not None:
                pad_shape = [batch_size, subsequences_length - sequence_length - end_index, *sequence.shape[2:]]
                lagged_values.append(torch.cat([torch.zeros(pad_shape), sequence[:, :end_index, ...]], dim=1))
            else:
                pad_shape = [batch_size, subsequences_length - sequence_length, *sequence.shape[2:]]
                lagged_values.append(torch.cat([torch.zeros(pad_shape), sequence], dim=1))
            continue
        else:
            lagged_values.append(sequence[:, begin_index:end_index, ...])

    lagged_seq = torch.stack(lagged_values, -1).transpose(-1, -2).reshape(batch_size, subsequences_length, -1)
    return lagged_seq


class ForecastingNet(nn.Module):
    future_target_required = False

    def __init__(self,
                 network_embedding: nn.Module,  # TODO consider  embedding for past, future and static features
                 network_encoder: EncoderNetwork,
                 network_decoder: nn.Module,
                 network_head: Optional[nn.Module],
                 window_size: int,
                 target_scaler: BaseTargetScaler,
                 dataset_properties: Dict,
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
            dataset_properties (Dict): dataset properties
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

        self.target_scaler = target_scaler

        self.n_prediction_steps = dataset_properties['n_prediction_steps']  # type: int
        self.window_size = window_size

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
        # self.mask_futur_features = decoder_properties['mask_future_features']
        self._device = torch.device('cpu')

        self.encoder_lagged_input = encoder_properties['lagged_input']
        self.decoder_lagged_input = decoder_properties['lagged_input']

        if self.encoder_lagged_input:
            self.cached_lag_mask_encoder = None
        if self.decoder_lagged_input:
            self.cached_lag_mask_decoder = None

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self.to(device)
        self._device = device

    def rescale_output(self,
                       outputs: Union[torch.distributions.Distribution, torch.Tensor],
                       loc: Optional[torch.Tensor],
                       scale: Optional[torch.Tensor],
                       device: torch.device = torch.device('cpu')):
        if loc is not None or scale is not None:
            if isinstance(outputs, torch.distributions.Distribution):
                transform = AffineTransform(loc=0.0 if loc is None else loc.to(device),
                                            scale=1.0 if scale is None else scale.to(device),
                                            )
                outputs = TransformedDistribution_(outputs, [transform])
            else:
                if loc is None:
                    outputs = outputs * scale.to(device)
                elif scale is None:
                    outputs = outputs + loc.to(device)
                else:
                    outputs = outputs * scale.to(device) + loc.to(device)
        return outputs

    def scale_value(self,
                    outputs: Union[torch.distributions.Distribution, torch.Tensor],
                    loc: Optional[torch.Tensor],
                    scale: Optional[torch.Tensor],
                    device: torch.device = torch.device('cpu')):
        if loc is not None or scale is not None:
            if loc is None:
                outputs = outputs / scale.to(device)
            elif scale is None:
                outputs = outputs - loc.to(device)
            else:
                outputs = (outputs - loc.to(device)) / scale.to(device)
        return outputs

    def forward(self,
                targets_past: torch.Tensor,
                targets_future: Optional[torch.Tensor] = None,
                features_past: Optional[torch.Tensor] = None,
                features_future: Optional[torch.Tensor] = None,
                features_static: Optional[torch.Tensor] = None,
                hidden_states: Optional[Tuple[torch.Tensor]] = None):
        # TODO We need to replace thus None with empty tensors to avoid checking if they are None every time!
        if self.encoder_lagged_input:
            targets_past[:, -self.window_size:], _, loc, scale = self.target_scaler(targets_past[:, -self.window_size:])
            targets_past[:, :-self.window_size] = self.scale_value(targets_past[:, :-self.window_size], loc, scale)
            x_past, self.cached_lag_mask_encoder = get_lagged_subsequences(targets_past,
                                                                           self.window_size,
                                                                           self.encoder.lagged_value,
                                                                           self.cached_lag_mask_encoder)
        else:
            if self.window_size < targets_past.shape[1]:
                targets_past = targets_past[:, -self.window_size:]
            targets_past, _, loc, scale = self.target_scaler(targets_past)
            x_past = targets_past

        if features_past is not None:
            x_past = torch.cat([features_past, x_past], dim=1)

        x_past = x_past.to(device=self.device)
        x_past = self.embedding(x_past)

        if self.encoder_has_hidden_states:
            x_past, _ = self.encoder(x_past)
        else:
            x_past = self.encoder(x_past)
        x_past = self.decoder(x_past)
        output = self.head(x_past)
        return self.rescale_output(output, loc, scale, self.device)

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
    Forecasting network with Seq2Seq structure, Encoder/ Decoder need to be the same recurrent models while 

    This structure is activate when the decoder is recurrent (RNN or transformer). 
    We train the network with teacher forcing, thus
    targets_future is required for the network. To train the network, past targets and past features are fed to the
    encoder to obtain the hidden states whereas future targets and future features.
    When the output type is distribution and forecast_strategy is sampling, this model is equivalent to a deepAR model 
    during inference.
    """

    def __init__(self, **kwargs):
        super(ForecastingSeq2SeqNet, self).__init__(**kwargs)
        self.mask_on_future_target = kwargs['decoder_properties']['mask_on_future_target']
        if self.mask_on_future_target:
            self.tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.n_prediction_steps)

    def forward(self,
                targets_past: torch.Tensor,
                targets_future: Optional[torch.Tensor] = None,
                features_past: Optional[torch.Tensor] = None,
                features_future: Optional[torch.Tensor] = None,
                features_static: Optional[torch.Tensor] = None,
                hidden_states: Optional[Tuple[torch.Tensor]] = None):
        if self.encoder_lagged_input:
            targets_past[:, -self.window_size:], _, loc, scale = self.target_scaler(targets_past[:, -self.window_size:])
            targets_past[:, :-self.window_size] = self.scale_value(targets_past[:, :-self.window_size], loc, scale)
            x_past, self.cached_lag_mask_encoder = get_lagged_subsequences(targets_past,
                                                                           self.window_size,
                                                                           self.encoder.lagged_value,
                                                                           self.cached_lag_mask_encoder)
        else:
            if self.window_size < targets_past.shape[1]:
                targets_past = targets_past[:, -self.window_size:]
            targets_past, _, loc, scale = self.target_scaler(targets_past)
            x_past = targets_past

        x_past = x_past if features_past is None else torch.cat([features_past, x_past], dim=-1)

        x_past = x_past.to(self.device)
        x_past = self.embedding(x_past)

        if self.training:
            # we do one step ahead forecasting
            if self.decoder_lagged_input:
                targets_future = torch.cat([targets_past, targets_future[:, :-1, :]], dim=1)
                targets_future, self.cached_lag_mask_decoder = get_lagged_subsequences(targets_future,
                                                                                       self.n_prediction_steps,
                                                                                       self.decoder.lagged_value,
                                                                                       self.cached_lag_mask_decoder)
            else:
                targets_future = torch.cat([targets_past[:, [-1], :], targets_future[:, :-1, :]], dim=1)

            x_future = targets_future if features_future is None else torch.cat([features_future, targets_future],
                                                                                dim=-1)
            x_future = x_future.to(self.device)

            if self.encoder_has_hidden_states:
                # RNN
                _, features_latent = self.encoder(x_past, output_seq=True)
                x_future, _ = self.decoder(x_future, features_latent)
            elif self.mask_on_future_target:
                features_latent = self.encoder(x_past, output_seq=True)
                x_future = self.decoder(x_future, features_latent, tgt_mask=self.tgt_mask.to(self.device))
            else:
                raise NotImplementedError
            net_output = self.head(x_future)

            return self.rescale_output(net_output, loc, scale, self.device)
        else:
            if self.encoder_has_hidden_states:
                _, features_latent = self.encoder(x_past, output_seq=True)
            else:
                features_latent = self.encoder(x_past, output_seq=True)

            if features_future is not None:
                features_future = features_future

            if self.forecast_strategy != 'sample':
                all_predictions = []
                predicted_target = targets_past[:, [-1]]
                targets_past = targets_past[:, :-1]
                for idx_pred in range(self.n_prediction_steps):
                    if self.decoder_lagged_input:
                        x_future = torch.cat([targets_past, predicted_target.cpu()], dim=1)
                        if self.decoder_has_hidden_states:
                            x_future = get_lagged_subsequences_inference(x_future, 1, self.decoder.lagged_value)
                        else:
                            x_future = get_lagged_subsequences_inference(x_future, idx_pred + 1,
                                                                         self.decoder.lagged_value)
                    else:
                        if self.decoder_has_hidden_states:
                            x_future = predicted_target[:, [-1]]
                        else:
                            x_future = predicted_target

                    if self.decoder_has_hidden_states:
                        x_future = x_future if features_future is None else torch.cat(
                            [features_future[:, [idx_pred], :], x_future], dim=-1)
                    else:
                        x_future = x_future if features_future is None else torch.cat(
                            [features_future[:, idx_pred + 1, :], x_future], dim=-1)

                    x_future = x_future.to(self.device)
                    if self.decoder_has_hidden_states:
                        x_future, features_latent = self.decoder(x_future, features_latent=features_latent)
                    else:
                        x_future = self.decoder(x_future, features_latent)

                    net_output = self.head(x_future[:, -1:, ])
                    predicted_target = torch.cat([predicted_target, self.pred_from_net_output(net_output).cpu()],
                                                 dim=1)

                    all_predictions.append(net_output)

                if self.output_type != 'distribution':
                    all_predictions = torch.cat(all_predictions, dim=1)
                else:
                    all_predictions = self.pred_from_net_output(all_predictions)

                return self.rescale_output(all_predictions, loc, scale, self.device)

            else:
                # we follow the DeepAR implementation:
                all_samples = []
                batch_size = targets_past.shape[0]

                if self.encoder_has_hidden_states:

                    if isinstance(features_latent, tuple):
                        repeated_state = [
                            s.repeat_interleave(repeats=self.num_samples, dim=1)
                            for s in features_latent
                        ]
                    else:
                        repeated_state = features_latent.repeat_interleave(repeats=self.num_samples, dim=1)
                else:
                    # Transformer's hidden states is of shape
                    repeated_state = features_latent.repeat_interleave(repeats=self.num_samples, dim=0)
                if self.decoder_lagged_input:
                    max_lag_seq_length = max(self.decoder.lagged_value) + 1
                else:
                    max_lag_seq_length = 1 + self.window_size
                repeated_past_target = targets_past[:, -max_lag_seq_length:].repeat_interleave(repeats=self.num_samples,
                                                                                               dim=0).squeeze(1)
                repeated_predicted_target = repeated_past_target[:, [-1]]
                repeated_past_target = repeated_past_target[:, :-1, ]

                repeated_static_feat = features_static.repeat_interleave(
                    repeats=self.num_samples, dim=0
                ).unsqueeze(dim=1) if features_static is not None else None

                repeated_time_feat = features_future.repeat_interleave(
                    repeats=self.num_samples, dim=0
                ) if features_future is not None else None

                for idx_pred in range(self.n_prediction_steps):
                    if self.decoder_lagged_input:
                        x_future = torch.cat([repeated_past_target, repeated_predicted_target.cpu()], dim=1)
                        if self.decoder_has_hidden_states:
                            x_future = get_lagged_subsequences_inference(x_future, 1, self.decoder.lagged_value)
                        else:
                            x_future = get_lagged_subsequences_inference(x_future, idx_pred + 1,
                                                                         self.decoder.lagged_value)
                    else:
                        if self.decoder_has_hidden_states:
                            x_future = repeated_predicted_target[:, [-1]]
                        else:
                            x_future = repeated_predicted_target

                    if self.decoder_has_hidden_states:
                        x_future = x_future if repeated_time_feat is None else torch.cat(
                            [repeated_time_feat[:, [idx_pred], :], x_future], dim=-1)
                    else:
                        # decoder uses the entire future targets
                        x_future = x_future if repeated_time_feat is None else torch.cat(
                            [repeated_time_feat[:, :idx_pred + 1, :], x_future], dim=-1)

                    x_future = x_future.to(self.device)
                    if self.decoder_has_hidden_states:
                        x_future, repeated_state = self.decoder(x_future, features_latent=repeated_state)
                    else:
                        x_future = self.decoder(x_future, repeated_state)
                    net_output = self.head(x_future[:, -1:, ])
                    samples = self.pred_from_net_output(net_output).cpu()
                    repeated_predicted_target = torch.cat([repeated_predicted_target,
                                                           samples],
                                                          dim=1)
                    all_samples.append(samples)

                all_predictions = torch.cat(all_samples, dim=1).unflatten(0, (batch_size, self.num_samples))

                if self.aggregation == 'mean':
                    return self.rescale_output(torch.mean(all_predictions, dim=1), loc, scale)
                elif self.aggregation == 'median':
                    return self.rescale_output(torch.median(all_predictions, dim=1)[0], loc, scale)
                else:
                    raise ValueError(f'Unknown aggregation: {self.aggregation}')

    def predict(self,
                targets_past: torch.Tensor,
                features_past: Optional[torch.Tensor] = None,
                features_future: Optional[torch.Tensor] = None,
                features_static: Optional[torch.Tensor] = None
                ):
        net_output = self(targets_past, features_past, features_future)
        if self.output_type != 'distribution':
            return self.pred_from_net_output(net_output)
        else:
            return net_output


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

        self.cached_lag_mask_encoder_test = None
        self.only_generate_future_dist = False

    def train(self, mode: bool = True) -> nn.Module:
        self.only_generate_future_dist = False
        return super().train(mode=mode)

    def forward(self,
                targets_past: torch.Tensor,
                targets_future: Optional[torch.Tensor] = None,
                features_past: Optional[torch.Tensor] = None,
                features_future: Optional[torch.Tensor] = None,
                features_static: Optional[torch.Tensor] = None,
                hidden_states: Optional[Tuple[torch.Tensor]] = None):
        if self.training:
            if self.encoder_lagged_input:
                targets_past[:, -self.window_size:], _, loc, scale = self.target_scaler(
                    targets_past[:, -self.window_size:])
                targets_past[:, :-self.window_size] = self.scale_value(targets_past[:, :-self.window_size], loc, scale)
                targets_future = self.scale_value(targets_future, loc, scale)

                targets_all = torch.cat([targets_past, targets_future[:, :-1]], dim=1)
                seq_length = self.window_size + self.n_prediction_steps
                targets_all, self.cached_lag_mask_encoder = get_lagged_subsequences(targets_all,
                                                                                    seq_length - 1,
                                                                                    self.encoder.lagged_value,
                                                                                    self.cached_lag_mask_encoder)
            else:
                if self.window_size < targets_past.shape[1]:
                    targets_past = targets_past[:, -self.window_size:]
                targets_past, _, loc, scale = self.target_scaler(targets_past)
                targets_future = self.scale_value(targets_future, loc, scale)
                targets_all = torch.cat([targets_past, targets_future[:, :-1]], dim=1)

            x_input = targets_all
            if features_past is not None:
                features_all = torch.cat([features_past[:, 1:], features_future], dim=1)
                x_input = torch.cat([features_all, x_input], dim=-1)
            x_input = x_input.to(self.device)

            x_input = self.embedding(x_input)

            if self.encoder_has_hidden_states:
                x_input, _ = self.encoder(x_input, output_seq=True)
            else:
                x_input = self.encoder(x_input, output_seq=True)
            if self.only_generate_future_dist:
                x_input = x_input[:, -self.n_prediction_steps:]
            net_output = self.head(self.decoder(x_input))
            return self.rescale_output(net_output, loc, scale, self.device)
        else:
            if self.encoder_lagged_input:
                targets_past[:, -self.window_size:], _, loc, scale = self.target_scaler(
                    targets_past[:, -self.window_size:])
                targets_past[:, :-self.window_size] = self.scale_value(targets_past[:, :-self.window_size], loc, scale)
                x_past, self.cached_lag_mask_encoder_test = get_lagged_subsequences(targets_past,
                                                                                    self.window_size,
                                                                                    self.encoder.lagged_value,
                                                                                    self.cached_lag_mask_encoder_test)
            else:
                if self.window_size < targets_past.shape[1]:
                    targets_past = targets_past[:, -self.window_size:]

                targets_past, _, loc, scale = self.target_scaler(targets_past)
                x_past = targets_past
            if features_past is not None:
                # features is one step ahead of target
                if self.window_size > 1:
                    features_all = torch.cat([features_past[:, -self.window_size + 1:, ],
                                              features_future],
                                             dim=1)
                else:
                    features_all = features_future
            else:
                features_all = None
            x_past = x_past if features_all is None else torch.cat([features_all[:, :self.window_size], x_past],
                                                                   dim=-1)

            x_past = x_past.to(self.device)
            # TODO consider static features
            x_past = self.embedding(x_past)

            all_samples = []
            batch_size = targets_past.shape[0]

            if self.encoder_has_hidden_states:
                # For RNN, we only feed the hidden state and generated future input to the netwrok
                encoder_output, hidden_states = self.encoder(x_past)
                if isinstance(hidden_states, tuple):
                    repeated_state = [
                        s.repeat_interleave(repeats=self.num_samples, dim=1)
                        for s in hidden_states
                    ]
                else:
                    repeated_state = hidden_states.repeat_interleave(repeats=self.num_samples, dim=1)
            else:
                # For other models, the full past targets are passed to the network.
                encoder_output = self.encoder(x_past)
            if self.encoder_lagged_input:
                max_lag_seq_length = max(max(self.encoder.lagged_value), self.window_size)
            else:
                max_lag_seq_length = self.window_size
            # TODO considering padding targets here instead of inside get_lagged function
            repeated_past_target = targets_past[:, -max_lag_seq_length:, ].repeat_interleave(
                repeats=self.num_samples,
                dim=0).squeeze(1)

            repeated_static_feat = features_static.repeat_interleave(
                repeats=self.num_samples, dim=0
            ).unsqueeze(dim=1) if features_static is not None else None

            if features_all is not None:
                if not self.encoder_has_hidden_states:
                        # both feature_past and feature_future must exist or not, otherwise deepAR is disabled due to
                        # data properties!!!
                        time_feature = features_all
                else:
                    time_feature = features_future[:, 1:] if self.n_prediction_steps > 1 else None
            else:
                time_feature = None

            repeated_time_feat = time_feature.repeat_interleave(
                repeats=self.num_samples, dim=0
            ) if features_future is not None else None

            net_output = self.head(self.decoder(encoder_output))

            next_sample = net_output.sample(sample_shape=(self.num_samples,))

            next_sample = next_sample.transpose(0, 1).reshape(
                (next_sample.shape[0] * next_sample.shape[1], 1, -1)
            ).cpu()

            all_samples.append(next_sample)

            for k in range(1, self.n_prediction_steps):
                if self.encoder_has_hidden_states:
                    if self.encoder_lagged_input:
                        x_next = torch.cat([repeated_past_target, *all_samples], dim=1)
                        x_next = get_lagged_subsequences_inference(x_next, 1, self.encoder.lagged_value)
                    else:
                        x_next = next_sample
                    x_next = x_next if repeated_time_feat is None else torch.cat([repeated_time_feat[:, k - 1:k],
                                                                                  x_next], dim=-1)
                    x_next = x_next.to(self.device)
                    encoder_output, repeated_state = self.encoder(x_next, hx=repeated_state)
                else:
                    if self.encoder_lagged_input:
                        x_next = torch.cat([repeated_past_target, *all_samples], dim=1)
                        x_next = get_lagged_subsequences_inference(x_next,
                                                                   self.window_size + k,
                                                                   self.encoder.lagged_value)
                    else:
                        x_next = torch.cat([repeated_past_target[:, -self.window_size:], *all_samples], dim=1)
                    if repeated_time_feat is None:
                        x_next = x_next
                    else:
                        x_next = torch.cat([repeated_time_feat[:, :self.window_size + k],
                                            x_next], dim=-1)

                    encoder_output = self.encoder(x_next)

                if self.encoder_has_hidden_states:
                    x_next = x_next.to(self.device)
                    encoder_output, repeated_state = self.encoder(x_next, hx=repeated_state)
                else:
                    x_next = torch.cat([repeated_past_target, *all_samples], dim=1).to(self.device)
                    encoder_output = self.encoder(x_next)
                # During training, the encoder output a sequence. Thus for prediction, the network should have the same
                # output format
                encoder_output = torch.unsqueeze(encoder_output, 1)

                net_output = self.head(self.decoder(encoder_output))

                next_sample = net_output.sample().cpu()
                all_samples.append(next_sample)

            all_predictions = torch.cat(all_samples, dim=1).unflatten(0, (batch_size, self.num_samples))

            if not self.output_type == 'distribution' and self.forecast_strategy == 'sample':
                raise ValueError(
                    f"A DeepAR network must have output type as Distribution and forecast_strategy as sample,"
                    f"but this network has {self.output_type} and {self.forecast_strategy}")
            if self.aggregation == 'mean':
                return self.rescale_output(torch.mean(all_predictions, dim=1), loc, scale)
            elif self.aggregation == 'median':
                return self.rescale_output(torch.median(all_predictions, dim=1)[0], loc, scale)
            else:
                raise ValueError(f'Unknown aggregation: {self.aggregation}')

    def predict(self,
                targets_past: torch.Tensor,
                features_past: Optional[torch.Tensor] = None,
                features_future: Optional[torch.Tensor] = None,
                features_static: Optional[torch.Tensor] = None
                ):
        net_output = self(targets_past, features_past, features_future)
        return net_output


class NBEATSNet(ForecastingNet):
    future_target_required = False

    def forward(self,
                targets_past: torch.Tensor,
                targets_future: Optional[torch.Tensor] = None,
                features_past: Optional[torch.Tensor] = None,
                features_future: Optional[torch.Tensor] = None,
                features_static: Optional[torch.Tensor] = None,
                hidden_states: Optional[Tuple[torch.Tensor]] = None):
        if self.window_size < targets_past.shape[1]:
            targets_past = targets_past[:, -self.window_size:]
        targets_past, _, loc, scale = self.target_scaler(targets_past)
        targets_past = targets_past.to(self.device)

        batch_size = targets_past.shape[0]
        output_shape = targets_past.shape[2:]
        forcast_shape = [batch_size, self.n_prediction_steps, *output_shape]

        forecast = torch.zeros(forcast_shape).to(self.device).flatten(1)
        backcast = self.encoder(targets_past)
        for block in self.decoder:
            backcast_block, forecast_block = block(backcast)

            backcast = backcast - backcast_block
            forecast = forecast + forecast_block
        backcast = backcast.reshape(targets_past.shape)
        forecast = forecast.reshape(forcast_shape)

        forecast = self.rescale_output(forecast, loc, scale, self.device)
        if self.training:
            backcast = self.rescale_output(backcast, loc, scale, self.device)
            return backcast, forecast
        else:
            return forecast

    def pred_from_net_output(self, net_output: torch.Tensor):
        return net_output


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
            FitRequirement("network_embedding", (torch.nn.Module,), user_defined=False, dataset_property=False),
            FitRequirement("network_encoder", (torch.nn.Module,), user_defined=False, dataset_property=False),
            FitRequirement("network_decoder", (torch.nn.Module,), user_defined=False, dataset_property=False),
            FitRequirement("network_head", (Optional[torch.nn.Module],), user_defined=False, dataset_property=False),
            FitRequirement("target_scaler", (BaseTargetScaler,), user_defined=False, dataset_property=False),
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
                                   window_size=X['window_size'],
                                   dataset_properties=X['dataset_properties'],
                                   encoder_properties=X['encoder_properties'],
                                   decoder_properties=X['decoder_properties'],
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
            X = X_batch['past_target']

            X = X.float()

            with torch.no_grad():
                Y_batch_pred = self.network.predict(X)

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
