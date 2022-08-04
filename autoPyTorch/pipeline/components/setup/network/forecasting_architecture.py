import warnings
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.distributions import AffineTransform, TransformedDistribution

from autoPyTorch.pipeline.components.setup.forecasting_target_scaling.base_target_scaler import BaseTargetScaler
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.cells import (
    StackedDecoder,
    StackedEncoder,
    TemporalFusionLayer,
    VariableSelector
)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.components_util import \
    NetworkStructure
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder.components import \
    DecoderBlockInfo
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.components import \
    EncoderBlockInfo
from autoPyTorch.pipeline.components.setup.network_embedding.NoEmbedding import \
    _NoEmbedding

ALL_NET_OUTPUT = Union[torch.Tensor, List[torch.Tensor], torch.distributions.Distribution]


class TransformedDistribution_(TransformedDistribution):
    """
    We implement the mean function such that we do not need to enquire base mean every time
    """

    @property
    def mean(self) -> torch.Tensor:
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

    Args:
        sequence (torch.Tensor):
            the sequence from which lagged subsequences should be extracted, Shape: (N, T, C).
        subsequences_length (int):
            length of the subsequences to be extracted.
        lags_seq (Optional[List[int]]):
            lags of the sequence, indicating the sequence that needs to be extracted
        mask (Optional[torch.Tensor]):
            a mask tensor indicating, it is a cached mask tensor that allows the model to quickly extract the desired
            lagged values

    Returns:
        lagged (Tensor)
            A tensor of shape (N, S, I * C), where S = subsequences_length and I = len(indices),
             containing lagged subsequences.
        mask (torch.Tensor):
            cached mask
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
        lags_seq: List[int]) -> torch.Tensor:
    """
    this function works exactly the same as get_lagged_subsequences. However, this implementation is faster when no
    cached value is available, thus it is applied during inference times.

    Args:
        sequence (torch.Tensor):
            the sequence from which lagged subsequences should be extracted, Shape: (N, T, C).
        subsequences_length (int):
            length of the subsequences to be extracted.
        lags_seq (Optional[List[int]]):
            lags of the sequence, indicating the sequence that needs to be extracted

    Returns:
        lagged (Tensor)
            A tensor of shape (N, S, I * C), where S = subsequences_length and I = len(indices),
             containing lagged subsequences.
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
    """
    This is a basic forecasting network. It is only composed of a embedding net, an encoder and a head (including
    MLP decoder and the final head).

    This structure is active when the decoder is a MLP with auto_regressive set as false

    Attributes:
        network_structure (NetworkStructure):
            network structure information
        network_embedding (nn.Module):
            network embedding
        network_encoder (Dict[str, EncoderBlockInfo]):
            Encoder network, could be selected to return a sequence or a 2D Matrix
        network_decoder (Dict[str, DecoderBlockInfo]):
            network decoder
        temporal_fusion Optional[TemporalFusionLayer]:
            Temporal Fusion Layer
        network_head (nn.Module):
            network head, maps the output of decoder to the final output
        dataset_properties (Dict):
            dataset properties
        auto_regressive (bool):
            if the model is auto-regressive model
        output_type (str):
            the form that the network outputs. It could be regression, distribution or quantile
        forecast_strategy (str):
            only valid if output_type is distribution or quantile, how the network transforms
            its output to predicted values, could be mean or sample
        num_samples (int):
            only valid if output_type is not regression and forecast_strategy is sample. This indicates the
            number of the points to sample when doing prediction
        aggregation (str):
            only valid if output_type is not regression and forecast_strategy is sample. The way that the samples
            are aggregated. We could take their mean or median values.
    """
    future_target_required = False
    dtype = torch.float

    def __init__(self,
                 network_structure: NetworkStructure,
                 network_embedding: nn.Module,
                 network_encoder: Dict[str, EncoderBlockInfo],
                 network_decoder: Dict[str, DecoderBlockInfo],
                 temporal_fusion: Optional[TemporalFusionLayer],
                 network_head: nn.Module,
                 window_size: int,
                 target_scaler: BaseTargetScaler,
                 dataset_properties: Dict,
                 auto_regressive: bool,
                 feature_names: Union[Tuple[str], Tuple[()]] = (),
                 known_future_features: Union[Tuple[str], Tuple[()]] = (),
                 feature_shapes: Dict[str, int] = {},
                 static_features: Union[Tuple[str], Tuple[()]] = (),
                 time_feature_names: Union[Tuple[str], Tuple[()]] = (),
                 output_type: str = 'regression',
                 forecast_strategy: Optional[str] = 'mean',
                 num_samples: int = 50,
                 aggregation: str = 'mean'
                 ):
        super().__init__()
        self.network_structure = network_structure
        self.embedding = network_embedding
        if len(known_future_features) > 0:
            known_future_features_idx = [feature_names.index(kff) for kff in known_future_features]
            self.decoder_embedding = self.embedding.get_partial_models(known_future_features_idx)
        else:
            self.decoder_embedding = _NoEmbedding()
        # modules that generate tensors while doing forward pass
        self.lazy_modules = []
        if network_structure.variable_selection:
            self.variable_selector = VariableSelector(network_structure=network_structure,
                                                      dataset_properties=dataset_properties,
                                                      network_encoder=network_encoder,
                                                      auto_regressive=auto_regressive,
                                                      feature_names=feature_names,
                                                      known_future_features=known_future_features,
                                                      feature_shapes=feature_shapes,
                                                      static_features=static_features,
                                                      time_feature_names=time_feature_names,
                                                      )
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
            if temporal_fusion is None:
                raise ValueError("When the network structure uses temporal fusion layer, "
                                 "temporal_fusion must be given!")
            self.temporal_fusion = temporal_fusion  # type: TemporalFusionLayer
            self.lazy_modules.append(self.temporal_fusion)
        self.has_temporal_fusion = has_temporal_fusion
        self.head = network_head

        first_decoder = 'block_0'
        for i in range(1, network_structure.num_blocks + 1):
            block_number = f'block_{i}'
            if block_number in network_decoder:
                if first_decoder == 'block_0':
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
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self.to(device)
        self._device = device
        for model in self.lazy_modules:
            model.device = device

    def rescale_output(self,
                       outputs: ALL_NET_OUTPUT,
                       loc: Optional[torch.Tensor],
                       scale: Optional[torch.Tensor],
                       device: torch.device = torch.device('cpu')) -> ALL_NET_OUTPUT:
        """
        rescale the network output to its raw scale

        Args:
            outputs (ALL_NET_OUTPUT):
                network head output
            loc (Optional[torch.Tensor]):
                scaling location value
            scale (Optional[torch.Tensor]):
                scaling scale value
            device (torch.device):
                which device the output is stored

        Return:
            ALL_NET_OUTPUT:
                rescaleed network output
        """
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
                    outputs = outputs * scale.to(device)  # type: ignore[union-attr]
                elif scale is None:
                    outputs = outputs + loc.to(device)
                else:
                    outputs = outputs * scale.to(device) + loc.to(device)
        return outputs

    def scale_value(self,
                    raw_value: torch.Tensor,
                    loc: Optional[torch.Tensor],
                    scale: Optional[torch.Tensor],
                    device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        scale the outputs

        Args:
            raw_value (torch.Tensor):
                network head output
            loc (Optional[torch.Tensor]):
                scaling location value
            scale (Optional[torch.Tensor]):
                scaling scale value
            device (torch.device):
                which device the output is stored

        Return:
            torch.Tensor:
                scaled input value
        """
        if loc is not None or scale is not None:
            if loc is None:
                outputs = raw_value / scale.to(device)  # type: ignore[union-attr]
            elif scale is None:
                outputs = raw_value - loc.to(device)
            else:
                outputs = (raw_value - loc.to(device)) / scale.to(device)
            return outputs
        else:
            return raw_value

    @abstractmethod
    def forward(self,
                past_targets: torch.Tensor,
                future_targets: Optional[torch.Tensor] = None,
                past_features: Optional[torch.Tensor] = None,
                future_features: Optional[torch.Tensor] = None,
                past_observed_targets: Optional[torch.BoolTensor] = None,
                decoder_observed_values: Optional[torch.Tensor] = None,
                ) -> ALL_NET_OUTPUT:
        raise NotImplementedError

    @abstractmethod
    def pred_from_net_output(self, net_output: ALL_NET_OUTPUT) -> torch.Tensor:
        """
        This function is applied to transform the network head output to torch tensor to create the point prediction

        Args:
            net_output (ALL_NET_OUTPUT):
                network head output

        Return:
            torch.Tensor:
                point prediction
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self,
                past_targets: torch.Tensor,
                past_features: Optional[torch.Tensor] = None,
                future_features: Optional[torch.Tensor] = None,
                past_observed_targets: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        raise NotImplementedError

    def repeat_intermediate_values(self,
                                   intermediate_values: List[Optional[Union[torch.Tensor, Tuple[torch.Tensor]]]],
                                   is_hidden_states: List[bool],
                                   repeats: int) -> List[Optional[Union[torch.Tensor, Tuple[torch.Tensor]]]]:
        """
        This function is often applied for auto-regressive model where we sample multiple points to form several
        trajectories and we need to repeat the intermediate values to ensure that the batch sizes match

        Args:
             intermediate_values (List[Optional[Union[torch.Tensor, Tuple[torch.Tensor]]]])
                a list of intermediate values to be repeated
             is_hidden_states  (List[bool]):
                if the intermediate_value is hidden states in RNN-form network, we need to consider the
                hidden states differently
            repeats (int):
                number of repeats

        Return:
            List[Optional[Union[torch.Tensor, Tuple[torch.Tensor]]]]:
                repeated values
        """
        for i, (is_hx, inter_value) in enumerate(zip(is_hidden_states, intermediate_values)):
            if isinstance(inter_value, torch.Tensor):
                repeated_value = inter_value.repeat_interleave(repeats=repeats, dim=1 if is_hx else 0)
                intermediate_values[i] = repeated_value
            elif isinstance(inter_value, tuple):
                dim = 1 if is_hx else 0
                repeated_value = tuple(hx.repeat_interleave(repeats=repeats, dim=dim) for hx in inter_value)
                intermediate_values[i] = repeated_value
        return intermediate_values

    def pad_tensor(self, tensor_to_be_padded: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        pad tensor to meet the required length

        Args:
             tensor_to_be_padded (torch.Tensor)
                tensor to be padded
             target_length  (int):
                target length

        Return:
            torch.Tensor:
                padded tensors
        """
        tensor_shape = tensor_to_be_padded.shape
        padding_size = [tensor_shape[0], target_length - tensor_shape[1], tensor_shape[-1]]
        tensor_to_be_padded = torch.cat([tensor_to_be_padded.new_zeros(padding_size), tensor_to_be_padded], dim=1)
        return tensor_to_be_padded


class ForecastingNet(AbstractForecastingNet):
    def pre_processing(self,
                       past_targets: torch.Tensor,
                       past_observed_targets: torch.BoolTensor,
                       past_features: Optional[torch.Tensor] = None,
                       future_features: Optional[torch.Tensor] = None,
                       length_past: int = 0,
                       length_future: int = 0,
                       variable_selector_kwargs: Dict = {},
                       ) -> Tuple[torch.Tensor, ...]:
        if self.encoder_lagged_input:
            if self.window_size < past_targets.shape[1]:
                past_targets[:, -self.window_size:], _, loc, scale = self.target_scaler(
                    past_targets[:, -self.window_size:],
                    past_observed_targets[:, -self.window_size:]
                )
                past_targets[:, :-self.window_size] = torch.where(
                    past_observed_targets[:, :-self.window_size],
                    self.scale_value(past_targets[:, :-self.window_size], loc, scale),
                    past_targets[:, :-self.window_size])
            else:
                past_targets, _, loc, scale = self.target_scaler(
                    past_targets,
                    past_observed_targets
                )
            truncated_past_targets, self.cached_lag_mask_encoder = get_lagged_subsequences(past_targets,
                                                                                           self.window_size,
                                                                                           self.encoder_lagged_value,
                                                                                           self.cached_lag_mask_encoder)
        else:
            if self.window_size < past_targets.shape[1]:
                past_targets = past_targets[:, -self.window_size:]
                past_observed_targets = past_observed_targets[:, -self.window_size:]
            past_targets, _, loc, scale = self.target_scaler(past_targets, past_observed_targets)
            truncated_past_targets = past_targets
        if past_features is not None:
            if self.window_size <= past_features.shape[1]:
                past_features = past_features[:, -self.window_size:]
            elif self.encoder_lagged_input:
                past_features = self.pad_tensor(past_features, self.window_size)

        if self.network_structure.variable_selection:
            batch_size = truncated_past_targets.shape[0]
            feat_dict_static = {}
            if length_past > 0:
                if past_features is not None:
                    past_features = self.embedding(past_features.to(self.device))
                feat_dict_past = {'past_targets': truncated_past_targets.to(device=self.device)}

                if past_features is not None:
                    for feature_name in self.variable_selector.feature_names:
                        tensor_idx = self.variable_selector.feature_names2tensor_idx[feature_name]
                        if feature_name not in self.variable_selector.static_features:
                            feat_dict_past[feature_name] = past_features[:, :, tensor_idx[0]: tensor_idx[1]]
                        else:
                            static_feature = past_features[:, 0, tensor_idx[0]: tensor_idx[1]]
                            feat_dict_static[feature_name] = static_feature

                if hasattr(self.variable_selector, 'placeholder_features'):
                    for placehold in self.variable_selector.placeholder_features:
                        feat_dict_past[placehold] = torch.zeros((batch_size, length_past, 1),
                                                                dtype=past_targets.dtype,
                                                                device=self.device)
            else:
                feat_dict_past = None  # type: ignore[assignment]
            if length_future > 0:
                if future_features is not None:
                    future_features = self.decoder_embedding(future_features.to(self.device))
                feat_dict_future = {}
                if hasattr(self.variable_selector, 'placeholder_features'):
                    for placehold in self.variable_selector.placeholder_features:
                        feat_dict_future[placehold] = torch.zeros((batch_size,
                                                                   length_future, 1),
                                                                  dtype=past_targets.dtype,
                                                                  device=self.device)
                if future_features is not None:
                    for feature_name in self.variable_selector.known_future_features:
                        tensor_idx = self.variable_selector.future_feature_name2tensor_idx[feature_name]
                        if feature_name not in self.variable_selector.static_features:
                            feat_dict_future[feature_name] = future_features[:, :, tensor_idx[0]: tensor_idx[1]]
                        else:
                            if length_past == 0:
                                # Otherwise static_feature is acquired when processing with encoder network
                                static_feature = future_features[:, 0, tensor_idx[0]: tensor_idx[1]]
                                feat_dict_static[feature_name] = static_feature

            else:
                feat_dict_future = None  # type: ignore[assignment]

            x_past, x_future, x_static, static_context_initial_hidden = self.variable_selector(
                x_past=feat_dict_past,
                x_future=feat_dict_future,
                x_static=feat_dict_static,
                batch_size=batch_size,
                length_past=length_past,
                length_future=length_future,
                **variable_selector_kwargs
            )

            return x_past, x_future, x_static, loc, scale, static_context_initial_hidden, past_targets
        else:
            if past_features is not None:
                x_past = torch.cat([truncated_past_targets, past_features], dim=-1).to(device=self.device)
                x_past = self.embedding(x_past.to(device=self.device))
            else:
                x_past = self.embedding(truncated_past_targets.to(device=self.device))
            if future_features is not None and length_future > 0:
                future_features = self.decoder_embedding(future_features.to(self.device))
            return x_past, future_features, None, loc, scale, None, past_targets

    def forward(self,
                past_targets: torch.Tensor,
                future_targets: Optional[torch.Tensor] = None,
                past_features: Optional[torch.Tensor] = None,
                future_features: Optional[torch.Tensor] = None,
                past_observed_targets: Optional[torch.BoolTensor] = None,
                decoder_observed_values: Optional[torch.Tensor] = None,
                ) -> ALL_NET_OUTPUT:
        x_past, x_future, x_static, loc, scale, static_context_initial_hidden, _ = self.pre_processing(
            past_targets=past_targets,
            past_observed_targets=past_observed_targets,
            past_features=past_features,
            future_features=future_features,
            length_past=min(self.window_size, past_targets.shape[1]),
            length_future=self.n_prediction_steps
        )

        encoder_additional = [static_context_initial_hidden]
        encoder_additional.extend([None] * (self.network_structure.num_blocks - 1))

        encoder2decoder, encoder_output = self.encoder(encoder_input=x_past, additional_input=encoder_additional)

        decoder_output = self.decoder(x_future=x_future, encoder_output=encoder2decoder,
                                      pos_idx=(x_past.shape[1], x_past.shape[1] + self.n_prediction_steps))

        if self.has_temporal_fusion:
            decoder_output = self.temporal_fusion(encoder_output=encoder_output,
                                                  decoder_output=decoder_output,
                                                  past_observed_targets=past_observed_targets,
                                                  decoder_length=self.n_prediction_steps,
                                                  static_embedding=x_static
                                                  )

        output = self.head(decoder_output)

        return self.rescale_output(output, loc, scale, self.device)

    def pred_from_net_output(self, net_output: ALL_NET_OUTPUT) -> torch.Tensor:
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
                    raise NotImplementedError(f'Unknown aggregation: {self.aggregation}')
            else:
                raise NotImplementedError(f'Unknown forecast_strategy: {self.forecast_strategy}')
        else:
            raise NotImplementedError(f'Unknown output_type: {self.output_type}')

    def predict(self,
                past_targets: torch.Tensor,
                past_features: Optional[torch.Tensor] = None,
                future_features: Optional[torch.Tensor] = None,
                past_observed_targets: Optional[torch.BoolTensor] = None,
                ) -> torch.Tensor:
        net_output = self(past_targets=past_targets,
                          past_features=past_features,
                          future_features=future_features,
                          past_observed_targets=past_observed_targets)
        return self.pred_from_net_output(net_output)


class ForecastingSeq2SeqNet(ForecastingNet):
    future_target_required = True
    """
    Forecasting network with Seq2Seq structure, Encoder/ Decoder need to be the same recurrent models while

    This structure is activate when the decoder is recurrent (RNN or transformer).
    We train the network with teacher forcing, thus
    future_targets is required for the network. To train the network, past targets and past features are fed to the
    encoder to obtain the hidden states whereas future targets and future features.
    When the output type is distribution and forecast_strategy is sampling,
    this model is equivalent to a deepAR model during inference.
    """

    def decoder_select_variable(self, future_targets: torch.tensor,
                                future_features: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size = future_targets.shape[0]
        length_future = future_targets.shape[1]
        future_targets = future_targets.to(self.device)
        if future_features is not None:
            future_features = self.decoder_embedding(future_features.to(self.device))
        feat_dict_future = {}
        if hasattr(self.variable_selector, 'placeholder_features'):
            for placeholder in self.variable_selector.placeholder_features:
                feat_dict_future[placeholder] = torch.zeros((batch_size,
                                                             length_future, 1),
                                                            dtype=future_targets.dtype,
                                                            device=self.device)

        for feature_name in self.variable_selector.known_future_features:
            tensor_idx = self.variable_selector.future_feature_name2tensor_idx[feature_name]
            if feature_name not in self.variable_selector.static_features:
                feat_dict_future[feature_name] = future_features[:, :, tensor_idx[0]: tensor_idx[1]]

        feat_dict_future['future_prediction'] = future_targets
        _, x_future, _, _ = self.variable_selector(x_past=None,
                                                   x_future=feat_dict_future,
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
                past_observed_targets: Optional[torch.BoolTensor] = None,
                decoder_observed_values: Optional[torch.Tensor] = None, ) -> ALL_NET_OUTPUT:
        x_past, _, x_static, loc, scale, static_context_initial_hidden, past_targets = self.pre_processing(
            past_targets=past_targets,
            past_observed_targets=past_observed_targets,
            past_features=past_features,
            future_features=future_features,
            length_past=min(self.window_size, past_targets.shape[1]),
            length_future=0,
            variable_selector_kwargs={'cache_static_contex': True}
        )
        encoder_additional = [static_context_initial_hidden]
        encoder_additional.extend([None] * (self.network_structure.num_blocks - 1))

        if self.training:
            future_targets = self.scale_value(future_targets, loc, scale)
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
                decoder_input = self.decoder_select_variable(future_targets, future_features)
            else:
                decoder_input = future_targets if future_features is None else torch.cat([future_features,
                                                                                          future_targets], dim=-1)
                decoder_input = decoder_input.to(self.device)
                decoder_input = self.decoder_embedding(decoder_input)

            encoder2decoder, encoder_output = self.encoder(encoder_input=x_past,
                                                           additional_input=encoder_additional)

            decoder_output = self.decoder(x_future=decoder_input, encoder_output=encoder2decoder,
                                          pos_idx=(x_past.shape[1], x_past.shape[1] + self.n_prediction_steps))

            if self.has_temporal_fusion:
                decoder_output = self.temporal_fusion(encoder_output=encoder_output,
                                                      decoder_output=decoder_output,
                                                      past_observed_targets=past_observed_targets,
                                                      decoder_length=self.n_prediction_steps,
                                                      static_embedding=x_static
                                                      )
            net_output = self.head(decoder_output)

            return self.rescale_output(net_output, loc, scale, self.device)
        else:
            encoder2decoder, encoder_output = self.encoder(encoder_input=x_past, additional_input=encoder_additional)

            if self.has_temporal_fusion:
                decoder_output_all: Optional[torch.Tensor] = None

            if self.forecast_strategy != 'sample':
                all_predictions = []
                predicted_target = past_targets[:, [-1]]
                past_targets = past_targets[:, :-1]
                for idx_pred in range(self.n_prediction_steps):
                    predicted_target = predicted_target.cpu()
                    if self.decoder_lagged_input:
                        past_targets = torch.cat([past_targets, predicted_target], dim=1)
                        ar_future_target = get_lagged_subsequences_inference(past_targets, 1,
                                                                             self.decoder_lagged_value)
                    else:
                        ar_future_target = predicted_target[:, [-1]]

                    if self.network_structure.variable_selection:
                        decoder_input = self.decoder_select_variable(
                            future_targets=predicted_target[:, -1:].to(self.device),
                            future_features=future_features[:, [idx_pred]] if future_features is not None else None
                        )
                    else:
                        decoder_input = ar_future_target if future_features is None else torch.cat(
                            [future_features[:, [idx_pred]],
                             ar_future_target,
                             ],
                            dim=-1)
                        decoder_input = decoder_input.to(self.device)
                        decoder_input = self.decoder_embedding(decoder_input)

                    decoder_output = self.decoder(decoder_input,
                                                  encoder_output=encoder2decoder,
                                                  pos_idx=(x_past.shape[1] + idx_pred, x_past.shape[1] + idx_pred + 1),
                                                  cache_intermediate_state=True,
                                                  incremental_update=idx_pred > 0)

                    if self.has_temporal_fusion:
                        if decoder_output_all is not None:
                            decoder_output_all = torch.cat([decoder_output_all, decoder_output], dim=1)
                        else:
                            decoder_output_all = decoder_output
                        decoder_output = self.temporal_fusion(encoder_output=encoder_output,
                                                              decoder_output=decoder_output_all,
                                                              past_observed_targets=past_observed_targets,
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
                batch_size = past_targets.shape[0]

                encoder2decoder = self.repeat_intermediate_values(
                    encoder2decoder,
                    is_hidden_states=self.encoder.encoder_has_hidden_states,
                    repeats=self.num_samples)

                if self.has_temporal_fusion:
                    intermediate_values = self.repeat_intermediate_values([encoder_output, past_observed_targets],
                                                                          is_hidden_states=[False, False],
                                                                          repeats=self.num_samples)

                    encoder_output = intermediate_values[0]
                    past_observed_targets = intermediate_values[1]

                if self.decoder_lagged_input:
                    max_lag_seq_length = max(self.decoder_lagged_value) + 1
                else:
                    max_lag_seq_length = 1 + self.window_size
                repeated_past_target = past_targets[:, -max_lag_seq_length:].repeat_interleave(repeats=self.num_samples,
                                                                                               dim=0).squeeze(1)
                repeated_predicted_target = repeated_past_target[:, [-1]]
                repeated_past_target = repeated_past_target[:, :-1, ]

                repeated_x_static = x_static.repeat_interleave(
                    repeats=self.num_samples, dim=0
                ) if x_static is not None else None

                repeated_future_features = future_features.repeat_interleave(
                    repeats=self.num_samples, dim=0
                ) if future_features is not None else None

                if self.network_structure.variable_selection:
                    self.variable_selector.cached_static_contex = self.repeat_intermediate_values(
                        [self.variable_selector.cached_static_contex],
                        is_hidden_states=[False],
                        repeats=self.num_samples
                    )[0]

                for idx_pred in range(self.n_prediction_steps):
                    if self.decoder_lagged_input:
                        ar_future_target = torch.cat([repeated_past_target, repeated_predicted_target.cpu()], dim=1)
                        ar_future_target = get_lagged_subsequences_inference(ar_future_target, 1,
                                                                             self.decoder_lagged_value)
                    else:
                        ar_future_target = repeated_predicted_target[:, [-1]]

                    if self.network_structure.variable_selection:
                        decoder_input = self.decoder_select_variable(
                            future_targets=ar_future_target,
                            future_features=None if repeated_future_features is None else
                            repeated_future_features[:, [idx_pred]])
                    else:
                        decoder_input = ar_future_target if repeated_future_features is None else torch.cat(
                            [repeated_future_features[:, [idx_pred], :], ar_future_target], dim=-1)

                        decoder_input = decoder_input.to(self.device)
                        decoder_input = self.decoder_embedding(decoder_input)

                    decoder_output = self.decoder(decoder_input,
                                                  encoder_output=encoder2decoder,
                                                  pos_idx=(x_past.shape[1] + idx_pred, x_past.shape[1] + idx_pred + 1),
                                                  cache_intermediate_state=True,
                                                  incremental_update=idx_pred > 0)

                    if self.has_temporal_fusion:
                        if decoder_output_all is not None:
                            decoder_output_all = torch.cat([decoder_output_all, decoder_output], dim=1)
                        else:
                            decoder_output_all = decoder_output
                        decoder_output = self.temporal_fusion(encoder_output=encoder_output,
                                                              decoder_output=decoder_output_all,
                                                              past_observed_targets=past_observed_targets,
                                                              decoder_length=idx_pred + 1,
                                                              static_embedding=repeated_x_static,
                                                              )[:, -1:, ]

                    net_output = self.head(decoder_output)
                    samples = net_output.sample().cpu()

                    repeated_predicted_target = torch.cat([repeated_predicted_target,
                                                           samples],
                                                          dim=1)

                all_predictions = repeated_predicted_target[:, 1:].unflatten(0, (batch_size, self.num_samples))

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
                past_observed_targets: Optional[torch.BoolTensor] = None,
                ) -> torch.Tensor:
        net_output = self(past_targets=past_targets,
                          past_features=past_features,
                          future_features=future_features,
                          past_observed_targets=past_observed_targets)
        if self.output_type == 'regression':
            return self.pred_from_net_output(net_output)
        else:
            return net_output


class ForecastingDeepARNet(ForecastingSeq2SeqNet):
    future_target_required = True

    def __init__(self,
                 **kwargs: Any):
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

    def encoder_select_variable(self, past_targets: torch.tensor, past_features: Optional[torch.Tensor],
                                length_past: int,
                                **variable_selector_kwargs: Any) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        batch_size = past_targets.shape[0]
        past_targets = past_targets.to(self.device)
        if past_features is not None:
            past_features = past_features.to(self.device)
            past_features = self.embedding(past_features)
        feat_dict_past = {'past_targets': past_targets.to(device=self.device)}
        feat_dict_static = {}
        if hasattr(self.variable_selector, 'placeholder_features'):
            for placehold in self.variable_selector.placeholder_features:
                feat_dict_past[placehold] = torch.zeros((batch_size, length_past, 1),
                                                        dtype=past_targets.dtype,
                                                        device=self.device)

        for feature_name in self.variable_selector.feature_names:
            tensor_idx = self.variable_selector.feature_names2tensor_idx[feature_name]
            if feature_name not in self.variable_selector.static_features:
                feat_dict_past[feature_name] = past_features[:, :, tensor_idx[0]: tensor_idx[1]]
            else:
                static_feature = past_features[:, 0, tensor_idx[0]: tensor_idx[1]]
                feat_dict_static[feature_name] = static_feature

        x_past, _, _, static_context_initial_hidden = self.variable_selector(x_past=feat_dict_past,
                                                                             x_future=None,
                                                                             x_static=feat_dict_static,
                                                                             length_past=length_past,
                                                                             length_future=0,
                                                                             batch_size=batch_size,
                                                                             **variable_selector_kwargs,
                                                                             )
        return x_past, static_context_initial_hidden

    def forward(self,
                past_targets: torch.Tensor,
                future_targets: Optional[torch.Tensor] = None,
                past_features: Optional[torch.Tensor] = None,
                future_features: Optional[torch.Tensor] = None,
                past_observed_targets: Optional[torch.BoolTensor] = None,
                decoder_observed_values: Optional[torch.Tensor] = None, ) -> ALL_NET_OUTPUT:
        encode_length = min(self.window_size, past_targets.shape[1])

        if past_observed_targets is None:
            past_observed_targets = torch.ones_like(past_targets, dtype=torch.bool)

        if self.training:
            if self.encoder_lagged_input:
                if self.window_size < past_targets.shape[1]:
                    past_targets[:, -self.window_size:], _, loc, scale = self.target_scaler(
                        past_targets[:, -self.window_size:],
                        past_observed_targets[:, -self.window_size:]
                    )

                    past_targets[:, :-self.window_size] = torch.where(
                        past_observed_targets[:, :-self.window_size],
                        self.scale_value(past_targets[:, :-self.window_size], loc, scale),
                        past_targets[:, :-self.window_size])
                else:
                    past_targets, _, loc, scale = self.target_scaler(
                        past_targets,
                        past_observed_targets
                    )

                future_targets = self.scale_value(future_targets, loc, scale)

                targets_all = torch.cat([past_targets, future_targets[:, :-1]], dim=1)
                seq_length = self.window_size + self.n_prediction_steps
                targets_all, self.cached_lag_mask_encoder = get_lagged_subsequences(targets_all,
                                                                                    seq_length - 1,
                                                                                    self.encoder_lagged_value,
                                                                                    self.cached_lag_mask_encoder)
                targets_all = targets_all[:, -(encode_length + self.n_prediction_steps - 1):]
            else:
                if self.window_size < past_targets.shape[1]:
                    past_targets = past_targets[:, -self.window_size:]
                    past_observed_targets = past_observed_targets[:, -self.window_size:]
                past_targets, _, loc, scale = self.target_scaler(past_targets, past_observed_targets)
                future_targets = self.scale_value(future_targets, loc, scale)
                targets_all = torch.cat([past_targets, future_targets[:, :-1]], dim=1)

            if self.network_structure.variable_selection:
                if past_features is not None:
                    assert future_features is not None
                    past_features = past_features[:, -self.window_size:]
                    features_all = torch.cat([past_features, future_features[:, :-1]], dim=1)
                else:
                    features_all = None
                length_past = min(self.window_size, past_targets.shape[1]) + self.n_prediction_steps - 1
                encoder_input, static_context_initial_hidden = self.encoder_select_variable(targets_all,
                                                                                            past_features=features_all,
                                                                                            length_past=length_past)
            else:
                if past_features is not None:
                    assert future_features is not None
                    if self.window_size <= past_features.shape[1]:
                        past_features = past_features[:, -self.window_size:]

                    features_all = torch.cat([past_features, future_features[:, :-1]], dim=1)
                    encoder_input = torch.cat([features_all, targets_all], dim=-1)
                else:
                    encoder_input = targets_all

                encoder_input = encoder_input.to(self.device)

                encoder_input = self.embedding(encoder_input)
                static_context_initial_hidden = None  # type: ignore[assignment]

            encoder_additional: List[Optional[torch.Tensor]] = [static_context_initial_hidden]
            encoder_additional.extend([None] * (self.network_structure.num_blocks - 1))

            encoder2decoder, encoder_output = self.encoder(encoder_input=encoder_input,
                                                           additional_input=encoder_additional,
                                                           output_seq=True)

            if self.only_generate_future_dist:
                # DeepAR only receives the output of the last encoder
                encoder2decoder = [encoder2decoder[-1][:, -self.n_prediction_steps:]]
            net_output = self.head(self.decoder(x_future=None, encoder_output=encoder2decoder))
            # DeepAR does not allow tf layers
            return self.rescale_output(net_output, loc, scale, self.device)
        else:
            if self.encoder_lagged_input:
                if self.window_size < past_targets.shape[1]:
                    past_targets[:, -self.window_size:], _, loc, scale = self.target_scaler(
                        past_targets[:, -self.window_size:],
                        past_observed_targets[:, -self.window_size:],
                    )

                    past_targets[:, :-self.window_size] = torch.where(
                        past_observed_targets[:, :-self.window_size],
                        self.scale_value(past_targets[:, :-self.window_size], loc, scale),
                        past_targets[:, :-self.window_size])
                else:
                    past_targets, _, loc, scale = self.target_scaler(
                        past_targets,
                        past_observed_targets,
                    )

                truncated_past_targets, self.cached_lag_mask_encoder_test = get_lagged_subsequences(
                    past_targets,
                    self.window_size,
                    self.encoder_lagged_value,
                    self.cached_lag_mask_encoder_test
                )
                truncated_past_targets = truncated_past_targets[:, -encode_length:]
            else:
                if self.window_size < past_targets.shape[1]:
                    past_targets = past_targets[:, -self.window_size:]
                    past_observed_targets = past_observed_targets[:, -self.window_size]
                past_targets, _, loc, scale = self.target_scaler(past_targets, past_observed_targets)
                truncated_past_targets = past_targets

            if self.network_structure.variable_selection:
                if past_features is not None:
                    features_all = past_features[:, -self.window_size:]
                else:
                    features_all = None
                variable_selector_kwargs = dict(cache_static_contex=True,
                                                use_cached_static_contex=False)

                encoder_input, static_context_initial_hidden = self.encoder_select_variable(truncated_past_targets,
                                                                                            past_features=features_all,
                                                                                            length_past=encode_length,
                                                                                            **variable_selector_kwargs)

            else:
                if past_features is not None:
                    assert future_features is not None
                    features_all = torch.cat([past_features[:, -encode_length:], future_features[:, :-1]], dim=1)
                else:
                    features_all = None

                encoder_input = truncated_past_targets if features_all is None else torch.cat(
                    [features_all[:, :encode_length], truncated_past_targets], dim=-1
                )

                encoder_input = encoder_input.to(self.device)
                encoder_input = self.embedding(encoder_input)
                static_context_initial_hidden = None  # type: ignore[assignment]

            all_samples = []
            batch_size: int = past_targets.shape[0]

            encoder_additional: List[Optional[torch.Tensor]] = [static_context_initial_hidden]  # type: ignore[no-redef]
            encoder_additional.extend([None] * (self.network_structure.num_blocks - 1))

            encoder2decoder, encoder_output = self.encoder(encoder_input=encoder_input,
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
                max_lag_seq_length = max(max(self.encoder_lagged_value), encode_length)
            else:
                max_lag_seq_length = encode_length

            net_output = self.head(self.decoder(x_future=None, encoder_output=encoder2decoder))

            next_sample = net_output.sample(sample_shape=(self.num_samples,))

            next_sample = next_sample.transpose(0, 1).reshape(
                (next_sample.shape[0] * next_sample.shape[1], 1, -1)
            ).cpu()

            all_samples.append(next_sample)

            # TODO considering padding targets here instead of inside get_lagged function
            if self.n_prediction_steps > 1:
                repeated_past_target = past_targets[:, -max_lag_seq_length:, ].repeat_interleave(
                    repeats=self.num_samples,
                    dim=0).squeeze(1)

                if future_features is not None:
                    future_features = future_features[:, 1:]
                else:
                    future_features = None

                repeated_future_features = future_features.repeat_interleave(
                    repeats=self.num_samples, dim=0
                ) if future_features is not None else None

            for k in range(1, self.n_prediction_steps):
                if self.encoder_lagged_input:
                    repeated_past_target = torch.cat([repeated_past_target, all_samples[-1]], dim=1)
                    ar_future_target = get_lagged_subsequences_inference(repeated_past_target, 1,
                                                                         self.encoder_lagged_value)
                else:
                    ar_future_target = next_sample

                if self.network_structure.variable_selection:
                    length_past = 1
                    variable_selector_kwargs = dict(use_cached_static_contex=True)
                    if repeated_future_features is not None:
                        feature_next = repeated_future_features[:, [k - 1]]
                    else:
                        feature_next = None
                    encoder_input, _ = self.encoder_select_variable(ar_future_target, past_features=feature_next,
                                                                    length_past=1,
                                                                    **variable_selector_kwargs)

                else:
                    if repeated_future_features is not None:
                        encoder_input = torch.cat([repeated_future_features[:, [k - 1]], ar_future_target], dim=-1)
                    else:
                        encoder_input = ar_future_target
                    encoder_input = encoder_input.to(self.device)
                    encoder_input = self.embedding(encoder_input)

                encoder2decoder, _ = self.encoder(encoder_input=encoder_input,
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
                past_observed_targets: Optional[torch.BoolTensor] = None,
                ) -> torch.Tensor:
        net_output = self(past_targets=past_targets,
                          past_features=past_features,
                          future_features=future_features,
                          past_observed_targets=past_observed_targets)
        return net_output


class NBEATSNet(ForecastingNet):
    future_target_required = False

    def forward(self,  # type: ignore[override]
                past_targets: torch.Tensor,
                future_targets: Optional[torch.Tensor] = None,
                past_features: Optional[torch.Tensor] = None,
                future_features: Optional[torch.Tensor] = None,
                past_observed_targets: Optional[torch.BoolTensor] = None,
                decoder_observed_values: Optional[torch.Tensor] = None, ) -> Union[torch.Tensor,
                                                                                   Tuple[torch.Tensor, torch.Tensor]]:

        # Unlike other networks, NBEATS network is required to predict both past and future targets.
        # Thereby, we return two tensors for backcast and forecast
        if past_observed_targets is None:
            past_observed_targets = torch.ones_like(past_targets, dtype=torch.bool)

        if self.window_size <= past_targets.shape[1]:
            past_targets = past_targets[:, -self.window_size:]
            past_observed_targets = past_observed_targets[:, -self.window_size:]
        else:
            past_targets = self.pad_tensor(past_targets, self.window_size)

        past_targets, _, loc, scale = self.target_scaler(past_targets, past_observed_targets)

        past_targets = past_targets.to(self.device)

        batch_size = past_targets.shape[0]
        output_shape = past_targets.shape[2:]
        forcast_shape = [batch_size, self.n_prediction_steps, *output_shape]

        forecast = torch.zeros(forcast_shape).to(self.device).flatten(1)
        backcast, _ = self.encoder(past_targets, [None])
        backcast = backcast[0]
        # nbeats network only has one decoder block (flat decoder)
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

    def pred_from_net_output(self, net_output: torch.Tensor) -> torch.Tensor:
        return net_output
