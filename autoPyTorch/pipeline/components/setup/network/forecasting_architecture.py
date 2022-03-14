from collections import OrderedDict
from typing import Any, Dict, Optional, Union, Tuple, List
from enum import Enum

from abc import abstractmethod

import torch
from torch import nn
import warnings

from torch.distributions import (
    AffineTransform,
    TransformedDistribution,
)

from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.forecasting_target_scaling. \
    base_target_scaler import BaseTargetScaler
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.components_util import NetworkStructure
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.components import (
    EncoderNetwork,
    EncoderBlockInfo,
)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.cells import (
    VariableSelector,
    StackedEncoder,
    StackedDecoder,
    TemporalFusionLayer
)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder.components import (
    DecoderBlockInfo
)

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.components_util import AddLayer
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import (
    TimeDistributed, TimeDistributedInterpolation, GatedLinearUnit, ResampleNorm, AddNorm, GateAddNorm,
    GatedResidualNetwork, VariableSelectionNetwork, InterpretableMultiHeadAttention
)


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


class AbstractForecastingNet(nn.Module):
    future_target_required = False
    dtype = torch.float

    def __init__(self,
                 network_structure: NetworkStructure,
                 network_embedding: nn.Module,  # TODO consider  embedding for past, future and static features
                 network_encoder: Dict[str, EncoderBlockInfo],
                 network_decoder: Dict[str, DecoderBlockInfo],
                 temporal_fusion: Optional[TemporalFusionLayer],
                 network_head: Optional[nn.Module],
                 window_size: int,
                 target_scaler: BaseTargetScaler,
                 dataset_properties: Dict,
                 auto_regressive: bool,
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
            auto_regressive (bool): if the overall model is auto-regressive model
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
        super().__init__()
        self.network_structure = network_structure
        self.embedding = network_embedding
        # modules that generate tensors while doing forward pass
        self.lazy_modules = []
        if network_structure.variable_selection:
            self.variable_selector = VariableSelector(network_structure=network_structure,
                                                      dataset_properties=dataset_properties,
                                                      network_encoder=network_encoder,
                                                      auto_regressive=auto_regressive)
            self.lazy_modules.append(self.variable_selector)
        has_temporal_fusion = network_structure.use_temporal_fusion
        self.encoder = StackedEncoder(network_structure=network_structure,
                                      has_temporal_fusion=has_temporal_fusion,
                                      encoder_info=network_encoder,
                                      decoder_info=network_decoder)
        self.decoder = StackedDecoder(network_structure=network_structure,
                                      encoder=self.encoder.encoder,
                                      encoder_info=network_encoder,
                                      decoder_info=network_decoder)
        if has_temporal_fusion:
            self.temporal_fusion = temporal_fusion  # type: TemporalFusionLayer
            self.lazy_modules.append(self.temporal_fusion)
        self.has_temporal_fusion = has_temporal_fusion
        self.head = network_head

        first_decoder = 0
        for i in range(1, network_structure.num_blocks + 1):
            block_number = f'block_{i}'
            if block_number in network_decoder:
                if first_decoder == 0:
                    first_decoder = block_number

        if first_decoder == 0:
            raise ValueError("At least one decoder must be specified!")

        self.target_scaler = target_scaler

        self.n_prediction_steps = dataset_properties['n_prediction_steps']  # type: int
        self.window_size = window_size

        self.output_type = output_type
        self.forecast_strategy = forecast_strategy
        self.num_samples = num_samples
        self.aggregation = aggregation

        self._device = torch.device('cpu')

        if not network_structure.variable_selection:
            self.encoder_lagged_input = network_encoder['block_1'].encoder_properties.lagged_input
            self.decoder_lagged_input = network_decoder[first_decoder].decoder_properties.lagged_input
        else:
            self.encoder_lagged_input = False
            self.decoder_lagged_input = False

        if self.encoder_lagged_input:
            self.cached_lag_mask_encoder = None
            self.encoder_lagged_value = network_encoder['block_1'].encoder.lagged_value
        if self.decoder_lagged_input:
            self.cached_lag_mask_decoder = None
            self.decoder_lagged_value = network_decoder[first_decoder].decoder.lagged_value

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self.to(device)
        self._device = device
        for model in self.lazy_modules:
            model.device = device

    def rescale_output(self,
                       outputs: Union[torch.distributions.Distribution, torch.Tensor, List[torch.Tensor]],
                       loc: Optional[torch.Tensor],
                       scale: Optional[torch.Tensor],
                       device: torch.device = torch.device('cpu')):
        if isinstance(outputs, List):
            return [self.rescale_output(output, loc, scale, device) for output in outputs]
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

    @abstractmethod
    def forward(self,
                past_targets: torch.Tensor,
                future_targets: Optional[torch.Tensor] = None,
                past_features: Optional[torch.Tensor] = None,
                future_features: Optional[torch.Tensor] = None,
                static_features: Optional[torch.Tensor] = None,
                encoder_lengths: Optional[torch.Tensor] = None,
                decoder_observed_values: Optional[torch.Tensor] = None,
                ):
        raise NotImplementedError

    @abstractmethod
    def pred_from_net_output(self, net_output):
        raise NotImplementedError

    @abstractmethod
    def predict(self,
                past_targets: torch.Tensor,
                past_features: Optional[torch.Tensor] = None,
                future_features: Optional[torch.Tensor] = None,
                static_features: Optional[torch.Tensor] = None
                ):
        raise NotImplementedError

    def repeat_intermediate_values(self,
                                   intermediate_values: List[Optional[Union[torch.Tensor, Tuple[torch.Tensor]]]],
                                   is_hidden_states: List[bool],
                                   repeats: int) -> List[Optional[Union[torch.Tensor, Tuple[torch.Tensor]]]]:
        for i, (is_hx, inter_value) in enumerate(zip(is_hidden_states, intermediate_values)):
            if isinstance(inter_value, torch.Tensor):
                repeated_value = inter_value.repeat_interleave(repeats=repeats, dim=1 if is_hx else 0)
                intermediate_values[i] = repeated_value
            elif isinstance(inter_value, Tuple):
                dim = 1 if is_hx else 0
                repeated_value = tuple(hx.repeat_interleave(repeats=repeats, dim=dim) for hx in inter_value)
                intermediate_values[i] = repeated_value
        return intermediate_values


class ForecastingNet(AbstractForecastingNet):
    def pre_processing(self,
                       past_targets: torch.Tensor,
                       past_features: Optional[torch.Tensor] = None,
                       future_features: Optional[torch.Tensor] = None,
                       static_features: Optional[torch.Tensor] = None,
                       length_past: int = 0,
                       length_future: int = 0,
                       variable_selector_kwargs: Dict = {},
                       ):
        if self.encoder_lagged_input:
            past_targets[:, -self.window_size:], _, loc, scale = self.target_scaler(past_targets[:, -self.window_size:])
            past_targets[:, :-self.window_size] = self.scale_value(past_targets[:, :-self.window_size], loc, scale)
            x_past, self.cached_lag_mask_encoder = get_lagged_subsequences(past_targets,
                                                                           self.window_size,
                                                                           self.encoder_lagged_value,
                                                                           self.cached_lag_mask_encoder)
        else:
            if self.window_size < past_targets.shape[1]:
                past_targets = past_targets[:, -self.window_size:]
            past_targets, _, loc, scale = self.target_scaler(past_targets)
            x_past = past_targets

        if self.network_structure.variable_selection:
            batch_size = x_past.shape[0]
            if length_past > 0:
                if past_features is None:
                    length_past = x_past.shape[1]
                    x_past = {'past_targets': x_past.to(device=self.device),
                              'features': torch.zeros((batch_size, length_past, 1),
                                                      dtype=past_targets.dtype, device=self.device)}
            else:
                x_past = None
            if length_future > 0:
                if future_features is None:
                    x_future = {'features': torch.zeros((batch_size, length_future, 1),
                                                        dtype=past_targets.dtype, device=self.device)}
            else:
                x_future = None
            x_past, x_future, x_static, static_context_initial_hidden = self.variable_selector(
                x_past=x_past,
                x_future=x_future,
                x_static=static_features,
                batch_size=batch_size,
                length_past=length_past,
                length_future=length_future,
                **variable_selector_kwargs
            )
            return x_past, x_future, x_static, loc, scale, static_context_initial_hidden
        else:
            if past_features is not None:
                x_past = torch.cat([past_features, x_past], dim=1)

            x_past = x_past.to(device=self.device)
            x_past = self.embedding(x_past)
            return x_past, future_features, static_features, loc, scale, None

    def forward(self,
                past_targets: torch.Tensor,
                future_targets: Optional[torch.Tensor] = None,
                past_features: Optional[torch.Tensor] = None,
                future_features: Optional[torch.Tensor] = None,
                static_features: Optional[torch.Tensor] = None,
                encoder_lengths: Optional[torch.LongTensor] = None,
                decoder_observed_values: Optional[torch.Tensor] = None,
                ):
        x_past, x_future, x_static, loc, scale, static_context_initial_hidden = self.pre_processing(
            past_targets=past_targets,
            past_features=past_features,
            future_features=future_features,
            static_features=static_features,
            length_past=self.window_size,
            length_future=self.n_prediction_steps
        )

        encoder_additional = [static_context_initial_hidden]
        encoder_additional.extend([None] * (self.network_structure.num_blocks - 1))
        encoder2decoder, encoder_output = self.encoder(encoder_input=x_past, additional_input=encoder_additional)
        decoder_output = self.decoder(x_future=x_future, encoder_output=encoder2decoder)

        if self.has_temporal_fusion:
            decoder_output = self.temporal_fusion(encoder_output=encoder_output,
                                                  decoder_output=decoder_output,
                                                  encoder_lengths=encoder_lengths,
                                                  decoder_length=self.n_prediction_steps,
                                                  static_embedding=x_static
                                                  )
        output = self.head(decoder_output)
        return self.rescale_output(output, loc, scale, self.device)

    def pred_from_net_output(self, net_output):
        if self.output_type == 'regression':
            return net_output
        elif self.output_type == 'quantile':
            return net_output[0]
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
                past_targets: torch.Tensor,
                past_features: Optional[torch.Tensor] = None,
                future_features: Optional[torch.Tensor] = None,
                static_features: Optional[torch.Tensor] = None,
                encoder_lengths: Optional[torch.LongTensor] = None,
                ):
        net_output = self(past_targets, past_features, encoder_lengths=encoder_lengths)
        return self.pred_from_net_output(net_output)


class ForecastingSeq2SeqNet(ForecastingNet):
    future_target_required = True
    """
    Forecasting network with Seq2Seq structure, Encoder/ Decoder need to be the same recurrent models while 

    This structure is activate when the decoder is recurrent (RNN or transformer). 
    We train the network with teacher forcing, thus
    future_targets is required for the network. To train the network, past targets and past features are fed to the
    encoder to obtain the hidden states whereas future targets and future features.
    When the output type is distribution and forecast_strategy is sampling, this model is equivalent to a deepAR model 
    during inference.
    """

    def __init__(self, **kwargs):
        super(ForecastingSeq2SeqNet, self).__init__(**kwargs)

    def decoder_select_variable(self, future_targets: torch.tensor, future_features: Optional[torch.Tensor]):
        batch_size = future_targets.shape[0]
        length_future = future_targets.shape[1]
        if future_features is None:
            x_future = {
                'future_prediction': future_targets.to(self.device),
                'features': torch.zeros((batch_size, length_future, 1),
                                        dtype=future_targets.dtype, device=self.device)}
        _, x_future, _, _ = self.variable_selector(x_past=None,
                                                   x_future=x_future,
                                                   x_static=None,
                                                   length_past=0,
                                                   length_future=length_future,
                                                   batch_size=batch_size,
                                                   use_cached_static_contex=True
                                                   )
        return x_future

    def forward(self,
                past_targets: torch.Tensor,
                future_targets: Optional[torch.Tensor] = None,
                past_features: Optional[torch.Tensor] = None,
                future_features: Optional[torch.Tensor] = None,
                static_features: Optional[torch.Tensor] = None,
                encoder_lengths: Optional[torch.Tensor] = None,
                decoder_observed_values: Optional[torch.Tensor] = None, ):
        x_past, x_future, x_static, loc, scale, static_context_initial_hidden = self.pre_processing(
            past_targets=past_targets,
            past_features=past_features,
            future_features=future_features,
            static_features=static_features,
            length_past=self.window_size,
            length_future=0,
            variable_selector_kwargs={'cache_static_contex': True}
        )
        encoder_additional = [static_context_initial_hidden]
        encoder_additional.extend([None] * (self.network_structure.num_blocks - 1))

        if self.training:
            # we do one step ahead forecasting
            if self.decoder_lagged_input:
                future_targets = torch.cat([past_targets, future_targets[:, :-1, :]], dim=1)
                future_targets, self.cached_lag_mask_decoder = get_lagged_subsequences(future_targets,
                                                                                       self.n_prediction_steps,
                                                                                       self.decoder_lagged_value,
                                                                                       self.cached_lag_mask_decoder)
            else:
                future_targets = torch.cat([past_targets[:, [-1], :], future_targets[:, :-1, :]], dim=1)

            if self.network_structure.variable_selection:
                x_future = self.decoder_select_variable(future_targets, future_features)
            else:
                x_future = future_targets if future_features is None else torch.cat([future_features, future_targets],
                                                                                    dim=-1)
            x_future = x_future.to(self.device)

            encoder2decoder, encoder_output = self.encoder(encoder_input=x_past,
                                                           additional_input=encoder_additional)

            decoder_output = self.decoder(x_future=x_future, encoder_output=encoder2decoder)

            if self.has_temporal_fusion:
                decoder_output = self.temporal_fusion(encoder_output=encoder_output,
                                                      decoder_output=decoder_output,
                                                      encoder_lengths=encoder_lengths,
                                                      decoder_length=self.n_prediction_steps,
                                                      static_embedding=x_static
                                                      )
            net_output = self.head(decoder_output)

            return self.rescale_output(net_output, loc, scale, self.device)
        else:
            encoder2decoder, encoder_output = self.encoder(encoder_input=x_past, additional_input=encoder_additional)

            if future_features is not None:
                future_features = future_features

            if self.has_temporal_fusion:
                decoder_output_all = None

            if self.forecast_strategy != 'sample':
                all_predictions = []
                predicted_target = past_targets[:, [-1]]
                past_targets = past_targets[:, :-1]
                for idx_pred in range(self.n_prediction_steps):
                    predicted_target = predicted_target.cpu()
                    if self.decoder_lagged_input:
                        x_future = torch.cat([past_targets, predicted_target], dim=1)
                        x_future = get_lagged_subsequences_inference(x_future, 1, self.decoder_lagged_value)
                    else:
                        x_future = predicted_target[:, [-1]]

                    x_future = x_future.to(self.device)

                    if self.network_structure.variable_selection:
                        x_future = self.decoder_select_variable(
                            future_targets=predicted_target[:, -1:].to(self.device),
                            future_features=future_features[:, [idx_pred]] if future_features is not None else None
                        )
                    else:
                        x_future = x_future if future_features is None else torch.cat([future_features, future_targets],
                                                                                      dim=-1)
                    decoder_output = self.decoder(x_future,
                                                  encoder_output=encoder2decoder,
                                                  cache_intermediate_state=True,
                                                  incremental_update=idx_pred > 0)

                    if self.has_temporal_fusion:
                        if decoder_output_all is not None:
                            decoder_output_all = torch.cat([decoder_output_all, decoder_output], dim=1)
                        else:
                            decoder_output_all = decoder_output
                        decoder_output = self.temporal_fusion(encoder_output=encoder_output,
                                                              decoder_output=decoder_output_all,
                                                              encoder_lengths=encoder_lengths,
                                                              decoder_length=idx_pred + 1,
                                                              static_embedding=x_static
                                                              )[:, -1:]

                    net_output = self.head(decoder_output)
                    predicted_target = torch.cat([predicted_target, self.pred_from_net_output(net_output).cpu()],
                                                 dim=1)

                    all_predictions.append(net_output)

                if self.output_type == 'regression':
                    all_predictions = torch.cat(all_predictions, dim=1)
                elif self.output_type == 'quantile':
                    all_predictions = torch.cat([self.pred_from_net_output(pred) for pred in all_predictions], dim=1)
                else:
                    all_predictions = self.pred_from_net_output(all_predictions)

                return self.rescale_output(all_predictions, loc, scale, self.device)

            else:
                # we follow the DeepAR implementation:
                all_samples = []
                batch_size = past_targets.shape[0]

                encoder2decoder = self.repeat_intermediate_values(
                    encoder2decoder,
                    is_hidden_states=self.encoder.encoder_has_hidden_states,
                    repeats=self.num_samples)

                intermediate_values = self.repeat_intermediate_values([encoder_output, encoder_lengths],
                                                                      is_hidden_states=[False, False],
                                                                      repeats=self.num_samples)

                encoder_output = intermediate_values[0]
                encoder_lengths = intermediate_values[1]

                if self.decoder_lagged_input:
                    max_lag_seq_length = max(self.decoder_lagged_value) + 1
                else:
                    max_lag_seq_length = 1 + self.window_size
                repeated_past_target = past_targets[:, -max_lag_seq_length:].repeat_interleave(repeats=self.num_samples,
                                                                                               dim=0).squeeze(1)
                repeated_predicted_target = repeated_past_target[:, [-1]]
                repeated_past_target = repeated_past_target[:, :-1, ]

                repeated_static_feat = static_features.repeat_interleave(
                    repeats=self.num_samples, dim=0
                ).unsqueeze(dim=1) if static_features is not None else None

                repeated_time_feat = future_features.repeat_interleave(
                    repeats=self.num_samples, dim=0
                ) if future_features is not None else None

                if self.network_structure.variable_selection:
                    self.variable_selector.cached_static_contex = self.repeat_intermediate_values(
                        [self.variable_selector.cached_static_contex],
                        is_hidden_states=[False],
                        repeats=self.num_samples)[0]

                for idx_pred in range(self.n_prediction_steps):
                    if self.decoder_lagged_input:
                        x_future = torch.cat([repeated_past_target, repeated_predicted_target.cpu()], dim=1)
                        x_future = get_lagged_subsequences_inference(x_future, 1, self.decoder_lagged_value)
                    else:
                        x_future = repeated_predicted_target[:, [-1]]

                    if self.network_structure.variable_selection:
                        x_future = self.decoder_select_variable(
                            future_targets=x_future[:, -1:],
                            future_features=None if repeated_time_feat is None else repeated_time_feat[:, [idx_pred]])
                    else:
                        x_future = x_future if repeated_time_feat is None else torch.cat(
                            [repeated_time_feat[:, [idx_pred], :], x_future], dim=-1)

                        x_future = x_future.to(self.device)

                    decoder_output = self.decoder(x_future,
                                                  encoder_output=encoder2decoder,
                                                  cache_intermediate_state=True,
                                                  incremental_update=idx_pred > 0)
                    if self.has_temporal_fusion:
                        if decoder_output_all is not None:
                            decoder_output_all = torch.cat([decoder_output_all, decoder_output], dim=1)
                        else:
                            decoder_output_all = decoder_output
                        decoder_output = self.temporal_fusion(encoder_output=encoder_output,
                                                              decoder_output=decoder_output_all,
                                                              encoder_lengths=encoder_lengths,
                                                              decoder_length=idx_pred + 1,
                                                              static_embedding=x_static,
                                                              )[:, -1:]

                    net_output = self.head(decoder_output)
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
                past_targets: torch.Tensor,
                past_features: Optional[torch.Tensor] = None,
                future_features: Optional[torch.Tensor] = None,
                static_features: Optional[torch.Tensor] = None,
                encoder_lengths: Optional[torch.LongTensor] = None,
                ):
        net_output = self(past_targets=past_targets,
                          past_features=past_features,
                          future_features=future_features,
                          static_features=static_features,
                          encoder_lengths=encoder_lengths)
        if self.output_type == 'regression':
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
        self.encoder_bijective_seq_output = kwargs['network_encoder']['block_1'].encoder_properties.bijective_seq_output

        self.cached_lag_mask_encoder_test = None
        self.only_generate_future_dist = False

    def train(self, mode: bool = True) -> nn.Module:
        self.only_generate_future_dist = False
        return super().train(mode=mode)

    def decoder_select_variable(self, future_targets: torch.tensor, future_features: Optional[torch.Tensor]):
        batch_size = future_targets.shape[0]
        length_future = future_targets.shape[1]
        if future_features is None:
            x_future = {
                'future_prediction': future_targets.to(self.device),
                'features': torch.zeros((batch_size, length_future, 1),
                                        dtype=future_targets.dtype, device=self.device)}
        _, x_future, _, _ = self.variable_selector(x_past=None,
                                                   x_future=x_future,
                                                   x_static=None,
                                                   length_past=0,
                                                   length_future=length_future,
                                                   batch_size=batch_size,
                                                   use_cached_static_contex=True
                                                   )
        return x_future

    def forward(self,
                past_targets: torch.Tensor,
                future_targets: Optional[torch.Tensor] = None,
                past_features: Optional[torch.Tensor] = None,
                future_features: Optional[torch.Tensor] = None,
                static_features: Optional[torch.Tensor] = None,
                encoder_lengths: Optional[torch.Tensor] = None,
                decoder_observed_values: Optional[torch.Tensor] = None, ):
        if self.training:
            if self.encoder_lagged_input:
                past_targets[:, -self.window_size:], _, loc, scale = self.target_scaler(
                    past_targets[:, -self.window_size:])
                past_targets[:, :-self.window_size] = self.scale_value(past_targets[:, :-self.window_size], loc, scale)
                future_targets = self.scale_value(future_targets, loc, scale)

                targets_all = torch.cat([past_targets, future_targets[:, :-1]], dim=1)
                seq_length = self.window_size + self.n_prediction_steps
                targets_all, self.cached_lag_mask_encoder = get_lagged_subsequences(targets_all,
                                                                                    seq_length - 1,
                                                                                    self.encoder_lagged_value,
                                                                                    self.cached_lag_mask_encoder)
            else:
                if self.window_size < past_targets.shape[1]:
                    past_targets = past_targets[:, -self.window_size:]
                past_targets, _, loc, scale = self.target_scaler(past_targets)
                future_targets = self.scale_value(future_targets, loc, scale)
                targets_all = torch.cat([past_targets, future_targets[:, :-1]], dim=1)

            if self.network_structure.variable_selection:
                batch_size = past_targets.shape[0]
                length_past = self.window_size + self.n_prediction_steps
                if past_features is None:
                    if past_features is None:
                        x_past = {'past_targets': targets_all.to(device=self.device),
                                  'features': torch.zeros((batch_size, length_past, 1),
                                                          dtype=targets_all.dtype, device=self.device)}

                x_input, _, _, static_context_initial_hidden = self.variable_selector(x_past=x_past,
                                                                                      x_future=None,
                                                                                      x_static=static_features,
                                                                                      length_past=length_past,
                                                                                      length_future=0,
                                                                                      batch_size=batch_size,
                                                                                      )
            else:
                x_input = targets_all
                if past_features is not None:
                    features_all = torch.cat([past_features[:, 1:], future_features], dim=1)
                    x_input = torch.cat([features_all, targets_all], dim=-1)
                x_input = x_input.to(self.device)

                x_input = self.embedding(x_input)
                static_context_initial_hidden = None

            encoder_additional = [static_context_initial_hidden]
            encoder_additional.extend([None] * (self.network_structure.num_blocks - 1))

            encoder2decoder, encoder_output = self.encoder(encoder_input=x_input,
                                                           additional_input=encoder_additional,
                                                           output_seq=True)

            if self.only_generate_future_dist:
                # DeepAR only receives the output of the last encoder
                encoder2decoder = encoder2decoder[-1][:, -self.n_prediction_steps:]
            net_output = self.head(self.decoder(x_future=None, encoder_output=encoder2decoder))
            # DeepAR does not allow tf layers
            return self.rescale_output(net_output, loc, scale, self.device)
        else:
            if self.encoder_lagged_input:
                past_targets[:, -self.window_size:], _, loc, scale = self.target_scaler(
                    past_targets[:, -self.window_size:])
                past_targets[:, :-self.window_size] = self.scale_value(past_targets[:, :-self.window_size], loc, scale)
                x_past, self.cached_lag_mask_encoder_test = get_lagged_subsequences(past_targets,
                                                                                    self.window_size,
                                                                                    self.encoder_lagged_value,
                                                                                    self.cached_lag_mask_encoder_test)
            else:
                if self.window_size < past_targets.shape[1]:
                    past_targets = past_targets[:, -self.window_size:]

                past_targets, _, loc, scale = self.target_scaler(past_targets)
                x_past = past_targets

            if self.network_structure.variable_selection:
                batch_size = past_targets.shape[0]
                length_past = self.window_size
                if past_features is None:
                    if past_features is None:
                        x_past = {'past_targets': past_targets.to(device=self.device),
                                  'features': torch.zeros((batch_size, length_past, 1),
                                                          dtype=past_targets.dtype, device=self.device)}

                x_past, _, _, static_context_initial_hidden = self.variable_selector(x_past=x_past,
                                                                                     x_future=None,
                                                                                     x_static=static_features,
                                                                                     length_past=length_past,
                                                                                     length_future=0,
                                                                                     batch_size=batch_size,
                                                                                     cache_static_contex=True
                                                                                     )
            else:
                if past_features is not None:
                    # features is one step ahead of target
                    if self.window_size > 1:
                        features_all = torch.cat([past_features[:, -self.window_size + 1:, ],
                                                  future_features],
                                                 dim=1)
                    else:
                        features_all = future_features
                else:
                    features_all = None
                x_past = x_past if features_all is None else torch.cat([features_all[:, :self.window_size], x_past],
                                                                       dim=-1)

                x_past = x_past.to(self.device)
                # TODO consider static features
                x_past = self.embedding(x_past)
                static_context_initial_hidden = None

            all_samples = []
            batch_size = past_targets.shape[0]

            encoder_additional = [static_context_initial_hidden]
            encoder_additional.extend([None] * (self.network_structure.num_blocks - 1))

            encoder2decoder, encoder_output = self.encoder(encoder_input=x_past,
                                                           additional_input=encoder_additional,
                                                           cache_intermediate_state=True,
                                                           )

            self.encoder.cached_intermediate_state = self.repeat_intermediate_values(
                self.encoder.cached_intermediate_state,
                is_hidden_states=self.encoder.encoder_has_hidden_states,
                repeats=self.num_samples)
            if self.network_structure.variable_selection:
                self.variable_selector.cached_static_contex = self.repeat_intermediate_values(
                    [self.variable_selector.cached_static_contex],
                    is_hidden_states=[False],
                    repeats=self.num_samples)[0]

            if self.encoder_lagged_input:
                max_lag_seq_length = max(max(self.encoder_lagged_value), self.window_size)
            else:
                max_lag_seq_length = self.window_size
            # TODO considering padding targets here instead of inside get_lagged function
            repeated_past_target = past_targets[:, -max_lag_seq_length:, ].repeat_interleave(
                repeats=self.num_samples,
                dim=0).squeeze(1)

            repeated_static_feat = static_features.repeat_interleave(
                repeats=self.num_samples, dim=0
            ).unsqueeze(dim=1) if static_features is not None else None

            if features_all is not None:
                if not self.encoder_has_hidden_states:
                    # both feature_past and feature_future must exist or not, otherwise deepAR is disabled due to
                    # data properties!!!
                    time_feature = features_all
                else:
                    time_feature = future_features[:, 1:] if self.n_prediction_steps > 1 else None
            else:
                time_feature = None

            repeated_time_feat = time_feature.repeat_interleave(
                repeats=self.num_samples, dim=0
            ) if future_features is not None else None

            net_output = self.head(self.decoder(x_future=None, encoder_output=encoder2decoder))

            next_sample = net_output.sample(sample_shape=(self.num_samples,))

            next_sample = next_sample.transpose(0, 1).reshape(
                (next_sample.shape[0] * next_sample.shape[1], 1, -1)
            ).cpu()

            all_samples.append(next_sample)

            for k in range(1, self.n_prediction_steps):
                if self.encoder_lagged_input:
                    repeated_past_target = torch.cat([repeated_past_target, all_samples[-1]], dim=1)
                    x_next = get_lagged_subsequences_inference(repeated_past_target, 1, self.encoder_lagged_value)
                else:
                    x_next = next_sample

                x_next = x_next.to(self.device)

                if self.network_structure.variable_selection:
                    batch_size = past_targets.shape[0]
                    if past_features is None:
                        if past_features is None:
                            x_next = {'past_targets': x_next,
                                      'features': torch.zeros((batch_size, 1, 1),
                                                              dtype=x_next.dtype, device=self.device)}

                    x_next, _, _, _ = self.variable_selector(x_past=x_next,
                                                             x_future=None,
                                                             x_static=static_features,
                                                             length_past=1,
                                                             length_future=0,
                                                             batch_size=batch_size,
                                                             cache_static_contex=False,
                                                             use_cached_static_contex=True,
                                                             )
                encoder2decoder, _ = self.encoder(encoder_input=x_next,
                                                  additional_input=[None] * self.network_structure.num_blocks,
                                                  output_seq=False, cache_intermediate_state=True,
                                                  incremental_update=True)

                net_output = self.head(self.decoder(x_future=None, encoder_output=encoder2decoder))

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
                past_targets: torch.Tensor,
                past_features: Optional[torch.Tensor] = None,
                future_features: Optional[torch.Tensor] = None,
                static_features: Optional[torch.Tensor] = None,
                encoder_lengths: Optional[torch.LongTensor] = None,
                ):
        net_output = self(past_targets=past_targets,
                          past_features=past_features,
                          future_features=future_features,
                          static_features=static_features,
                          encoder_lengths=encoder_lengths)
        return net_output


class NBEATSNet(ForecastingNet):
    future_target_required = False

    def forward(self,
                past_targets: torch.Tensor,
                future_targets: Optional[torch.Tensor] = None,
                past_features: Optional[torch.Tensor] = None,
                future_features: Optional[torch.Tensor] = None,
                static_features: Optional[torch.Tensor] = None,
                encoder_lengths: Optional[torch.Tensor] = None,
                decoder_observed_values: Optional[torch.Tensor] = None, ):
        if self.window_size < past_targets.shape[1]:
            past_targets = past_targets[:, -self.window_size:]
        past_targets, _, loc, scale = self.target_scaler(past_targets)
        past_targets = past_targets.to(self.device)

        batch_size = past_targets.shape[0]
        output_shape = past_targets.shape[2:]
        forcast_shape = [batch_size, self.n_prediction_steps, *output_shape]

        forecast = torch.zeros(forcast_shape).to(self.device).flatten(1)
        backcast, _ = self.encoder(past_targets, [None])
        backcast = backcast[0]
        for block in self.decoder.decoder['block_1']:
            backcast_block, forecast_block = block([None], backcast)

            backcast = backcast - backcast_block
            forecast = forecast + forecast_block
        backcast = backcast.reshape(past_targets.shape)
        forecast = forecast.reshape(forcast_shape)

        forecast = self.rescale_output(forecast, loc, scale, self.device)
        if self.training:
            backcast = self.rescale_output(backcast, loc, scale, self.device)
            return backcast, forecast
        else:
            return forecast

    def pred_from_net_output(self, net_output: torch.Tensor):
        return net_output
