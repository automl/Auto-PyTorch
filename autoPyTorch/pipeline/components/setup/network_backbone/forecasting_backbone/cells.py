from pytorch_forecasting.utils import create_mask

from typing import Any, Dict, Optional, List, Tuple

import torch
from torch import nn

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.components import (
    EncoderBlockInfo, EncoderOutputForm
)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder.components import (
    DecoderBlockInfo
)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.components_util import (
    NetworkStructure,
    AddLayer
)

from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import (
    GateAddNorm, GatedResidualNetwork, VariableSelectionNetwork, InterpretableMultiHeadAttention
)


class TemporalFusionLayer(nn.Module):
    """
    (Lim et al.
    Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting,
    https://arxiv.org/abs/1912.09363)
    we follow the implementation from pytorch forecasting:
    https://github.com/jdb78/pytorch-forecasting/blob/master/pytorch_forecasting/models/temporal_fusion_transformer/__init__.py
    """

    def __init__(self,
                 window_size: int,
                 n_prediction_steps: int,
                 network_structure: NetworkStructure,
                 network_encoder: Dict[str, EncoderBlockInfo],
                 n_decoder_output_features: int,
                 d_model: int,
                 n_head: int,
                 dropout: Optional[float] = None):
        super().__init__()
        num_blocks = network_structure.num_blocks
        last_block = f'block_{num_blocks}'
        n_encoder_output = network_encoder[last_block].encoder_output_shape[-1]
        self.window_size = window_size
        self.n_prediction_steps = n_prediction_steps
        self.timestep = window_size + n_prediction_steps

        if n_decoder_output_features != n_encoder_output:
            self.decoder_proj_layer = nn.Linear(n_decoder_output_features, n_encoder_output, bias=False)
        else:
            self.decoder_proj_layer = None
        if network_structure.variable_selection:
            if network_structure.skip_connection:
                # static feature selector needs to generate the same number of features as the output of the encoder
                n_encoder_output_first = network_encoder['block_1'].encoder_output_shape[-1]
                self.static_context_enrichment = GatedResidualNetwork(
                    n_encoder_output_first, n_encoder_output_first, n_encoder_output_first, dropout
                )
                self.enrichment = GatedResidualNetwork(
                    input_size=n_encoder_output,
                    hidden_size=n_encoder_output,
                    output_size=d_model,
                    dropout=dropout,
                    context_size=n_encoder_output_first,
                    residual=True,
                )
                self.enrich_with_static = True
        if not hasattr(self, 'enrichment'):
            self.enrichment = GatedResidualNetwork(
                input_size=n_encoder_output,
                hidden_size=n_encoder_output,
                output_size=d_model,
                dropout=self.dropout_rate if self.use_dropout else None,
                residual=True,
            )
            self.enrich_with_static = False

        self.attention_fusion = InterpretableMultiHeadAttention(
            d_model=d_model,
            n_head=n_head,
            dropout=dropout
        )
        self.post_attn_gate_norm = GateAddNorm(d_model, dropout=dropout, trainable_add=False)
        self.pos_wise_ff = GatedResidualNetwork(input_size=d_model, hidden_size=d_model,
                                                output_size=d_model, dropout=self.hparams.dropout)

        self.network_structure = network_structure
        if network_structure.skip_connection:
            if network_structure.skip_connection_type == 'add':
                self.residual_connection = AddLayer(d_model, n_encoder_output)
            elif network_structure.skip_connection_type == 'gate_add_norm':
                self.residual_connection = GateAddNorm(d_model, skip_size=n_encoder_output,
                                                       dropout=None, trainable_add=False)

    def forward(self, encoder_output: torch.Tensor, decoder_output: torch.Tensor, encoder_lengths: torch.LongTensor,
                static_embedding: Optional[torch.Tensor] = None):
        """
        Args:
            encoder_output: the output of the last layer of encoder network
            decoder_output: the output of the last layer of decoder network
            encoder_lengths: length of encoder network
            static_embedding: output of static variable selection network (if applible)
        """
        if self.decoder_proj_layer is not None:
            decoder_output = self.decoder_proj_layer(decoder_output)
        network_output = torch.cat([encoder_output, decoder_output], dim=1)

        if self.enrich_with_static:
            static_context_enrichment = self.static_context_enrichment(static_embedding)
            attn_input = self.enrichment(
                network_output, static_context_enrichment[:, None].expand(-1, self.timesteps, -1)
            )
        else:
            attn_input = self.enrichment(network_output)

        # Attention
        attn_output, attn_output_weights = self.attention_fusion(
            q=attn_input[:, self.window_size:],  # query only for predictions
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(
                encoder_lengths=encoder_lengths, decoder_length=self.n_prediction_steps
            ),
        )
        # skip connection over attention
        attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, self.window_size:])
        output = self.pos_wise_ff(attn_output)

        if self.network_structure.skip_connection:
            return self.residual_connection(output, decoder_output)
        else:
            return output

    def get_attention_mask(self, encoder_lengths: torch.LongTensor, decoder_length: int):
        """
        https://github.com/jdb78/pytorch-forecasting/blob/master/pytorch_forecasting/models/
        temporal_fusion_transformer/__init__.py
        """
        # indices to which is attended
        attend_step = torch.arange(decoder_length, device=self.device)
        # indices for which is predicted
        predict_step = torch.arange(0, decoder_length, device=self.device)[:, None]
        # do not attend to steps to self or after prediction
        # todo: there is potential value in attending to future forecasts if they are made with knowledge currently
        #   available
        #   one possibility is here to use a second attention layer for future attention (assuming different effects
        #   matter in the future than the past)
        #   or alternatively using the same layer but allowing forward attention - i.e. only masking out non-available
        #   data and self
        decoder_mask = attend_step >= predict_step
        # do not attend to steps where data is padded
        encoder_mask = create_mask(encoder_lengths.max(), encoder_lengths)
        # combine masks along attended time - first encoder and then decoder
        mask = torch.cat(
            (
                encoder_mask.unsqueeze(1).expand(-1, decoder_length, -1),
                decoder_mask.unsqueeze(0).expand(encoder_lengths.size(0), -1, -1),
            ),
            dim=2,
        )
        return mask


class VariableSelector(nn.Module):
    def __init__(self,
                 network_structure: NetworkStructure,
                 dataset_properties: Dict,
                 network_encoder: Dict[str, EncoderBlockInfo],
                 auto_regressive: bool = False
                 ):
        super().__init__()
        first_encoder_output_shape = network_encoder['block_1'].encoder_output_shape[-1]
        static_input_sizes = dataset_properties['static_features_shape']
        self.hidden_size = first_encoder_output_shape
        variable_selector = nn.ModuleDict()
        self.static_variable_selection = VariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={},
            dropout=network_structure.grn_dropout_rate,
        )
        self.static_input_sizes = static_input_sizes
        if dataset_properties['uni_variant']:
            # variable selection for encoder and decoder
            encoder_input_sizes = {
                'past_targets': dataset_properties['input_shape'][-1],
                'past_features': 0
            }
            decoder_input_sizes = {
                'future_features': 0
            }
            if auto_regressive:
                decoder_input_sizes.update({'future_prediction': dataset_properties['output_shape'][-1]})
        else:
            # TODO
            pass

        self.auto_regressive = auto_regressive

        # create single variable grns that are shared across decoder and encoder
        if network_structure.share_single_variable_networks:
            self.shared_single_variable_grns = nn.ModuleDict()
            for name, input_size in encoder_input_sizes.items():
                self.shared_single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, self.hidden_size),
                    self.hidden_size,
                    network_structure.grn_dropout_rate,
                )
            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[name] = GatedResidualNetwork(
                        input_size,
                        min(input_size, self.hidden_size),
                        self.hidden_size,
                        network_structure.grn_dropout_rate,
                    )

        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={},
            dropout=network_structure.grn_dropout_rate,
            context_size=self.hidden_size,
            single_variable_grns={}
            if not network_structure.share_single_variable_networks
            else variable_selector['shared_single_variable_grns'],
        )

        self.decoder_variable_selection = VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={},
            dropout=network_structure.grn_dropout_rate,
            context_size=self.hidden_size,
            single_variable_grns={}
            if not network_structure.share_single_variable_networks
            else variable_selector['shared_single_variable_grns'],
        )

        self.static_context_variable_selection = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=network_structure.grn_dropout_rate,
        )

        n_hidden_states = 0
        if network_encoder['block_1'].encoder_properties.has_hidden_states:
            n_hidden_states = network_encoder['block_1'].n_hidden_states

        static_context_initial_hidden = [GatedResidualNetwork(input_size=self.hidden_size,
                                                              hidden_size=self.hidden_size,
                                                              output_size=self.hidden_size,
                                                              dropout=network_structure.grn_dropout_rate,
                                                              ) for _ in range(n_hidden_states)]

        self.static_context_initial_hidden = nn.ModuleList(static_context_initial_hidden)
        self.cached_static_contex = None
        self.cached_static_embedding = None

    def forward(self,
                x_past: Optional[Dict[str,torch.Tensor]],
                x_future: Optional[Dict[str, torch.Tensor]],
                x_static: Optional[Dict[str, torch.Tensor]] = None,
                length_past: int = 0,
                length_future: int = 0,
                batch_size: int = 0,
                cache_static_contex: bool = False,
                use_cached_static_contex: bool = False,
                ):
        if x_past is None and x_future is None:
            raise ValueError('Either past input or future inputs need to be given!')
        if length_past == 0 and length_future == 0:
            raise ValueError("Either length_past or length_future must be given!")
        timesteps = length_past + length_future
        if not use_cached_static_contex:
            if self.static_input_sizes > 0:
                static_embedding, _ = self.static_variable_selection(x_static)
            else:
                static_embedding = torch.zeros(
                    (batch_size, self.hidden_size), dtype=self.dtype, device=self.device
                )
                static_variable_selection = torch.zeros((batch_size, 0), dtype=self.dtype, device=self.device)

            static_context_variable_selection = self.static_context_variable_selection(static_embedding)[:, None]
            static_context_initial_hidden = (init_hidden(static_embedding) for init_hidden in
                                             self.static_context_initial_hidden)
            if cache_static_contex:
                self.cached_static_contex = static_context_variable_selection
                self.cached_static_embedding = static_embedding
        else:
            static_embedding = self.cached_static_embedding
            static_context_initial_hidden = None
            static_context_variable_selection = self.cached_static_contex
        static_context_variable_selection = static_context_variable_selection[:, None].expand(-1, timesteps, -1)
        if x_past is not None:
            embeddings_varying_encoder, _ = self.encoder_variable_selection(
                x_past,
                static_context_variable_selection[:, :length_past],
            )
        else:
            embeddings_varying_encoder = None
        if x_future is not None:
            embeddings_varying_decoder, _ = self.decoder_variable_selection(
                x_future,
                static_context_variable_selection[:, length_past:],
            )
        else:
            embeddings_varying_decoder = None
        return embeddings_varying_encoder, embeddings_varying_decoder, static_embedding, static_context_initial_hidden


class StackedEncoder(nn.Module):
    def __init__(self,
                 network_structure: NetworkStructure,
                 has_temporal_fusion: bool,
                 encoder_info: Dict[str, EncoderBlockInfo],
                 decoder_info: Dict[str, DecoderBlockInfo],
                 ):
        super().__init__()
        self.num_blocks = network_structure.num_blocks
        self.skip_connection = network_structure.skip_connection
        self.has_temporal_fusion = has_temporal_fusion

        self.encoder_output_type = [EncoderOutputForm.NoOutput] * self.num_blocks
        self.encoder_has_hidden_states = [False] * self.num_blocks
        len_cached_intermediate_states = self.num_blocks + 1 if self.has_temporal_fusion else self.num_blocks
        self.cached_intermediate_state = [torch.empty(0) for _ in range(len_cached_intermediate_states)]

        self.encoder_num_hidden_states = []
        encoder = nn.ModuleDict()
        for i, block_idx in enumerate(range(1, self.num_blocks + 1)):
            block_id = f'block_{block_idx}'
            encoder[block_id] = encoder_info[block_id].encoder
            if self.skip_connection:
                input_size = encoder_info[block_id].encoder_output_shape[-1]
                skip_size = encoder_info[block_id].encoder_input_shape[-1]
                if network_structure.skip_connection_type == 'add':
                    encoder[f'skip_connection_{block_idx}'] = AddLayer(input_size, skip_size)
                elif network_structure.skip_connection_type == 'gate_add_norm':
                    encoder[f'skip_connection_{block_idx}'] = GateAddNorm(input_size,
                                                                  hidden_size=input_size,
                                                                  skip_size=skip_size,
                                                                  dropout=network_structure.grn_dropout_rate)
            if block_id in decoder_info:
                if decoder_info[block_id].decoder_properties.recurrent:
                    if decoder_info[block_id].decoder_properties.has_hidden_states:
                        # RNN
                        self.encoder_output_type[i] = EncoderOutputForm.HiddenStates
                    else:
                        # Transformer
                        self.encoder_output_type[i] = EncoderOutputForm.Sequence
                else:
                    self.encoder_output_type[i] = EncoderOutputForm.SequenceLast
            if encoder_info[block_id].encoder_properties.has_hidden_states:
                self.encoder_has_hidden_states[i] = True
                self.encoder_num_hidden_states[i] = encoder_info[block_id].n_hidden_states
            else:
                self.encoder_has_hidden_states[i] = False
        self.encoder = encoder

    def forward(self,
                encoder_input: torch.Tensor,
                additional_input: List[Optional[torch.Tensor]],
                output_seq: bool = False,
                cache_intermediate_state: bool = False,
                incremental_update: bool = False) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """
        A forward pass through the encoder
        Args:
             encoder_input (torch.Tensor): encoder input
             additional_input (List[Optional[torch.Tensor]]) additional input to the encoder, e.g., inital hidden states
             output_seq (bool) if a sequence output is generated
             incremental_update (bool) if an incremental update is applied, this is normally applied for auto-regressive
                model, however, ony deepAR requires encoder to do incremental update, thus the decoder only need to
                receive the last output of the encoder
        """
        encoder2decoder = []
        x = encoder_input
        for i, block_id in enumerate(range(1, self.num_blocks + 1)):
            output_seq_i = (output_seq or self.has_temporal_fusion or block_id < self.num_blocks)
            encoder_i = self.encoder[f'block_{block_id}']  # type: EncoderNetwork
            if self.encoder_has_hidden_states[i]:
                if incremental_update:
                    hx = self.cached_intermediate_state[i]
                    fx, hx = encoder_i(x, output_seq=False, hx=hx)
                else:
                    rnn_num_layers = encoder_i.config['num_layers']
                    hx = additional_input[i]
                    if rnn_num_layers == 1 or hx is None:
                        fx, hx = encoder_i(x, output_seq=output_seq_i, hx=hx)
                    else:
                        if self.encoder_num_hidden_states[i] == 1:
                            fx, hx = encoder_i(x, output_seq=output_seq_i, hx=hx.expand((rnn_num_layers, -1, -1)))
                        else:
                            hx = (hx_i.expand(rnn_num_layers, -1, -1) for hx_i in hx)
                            fx, hx = encoder_i(x, output_seq=output_seq_i, hx=hx)
            else:
                if incremental_update:
                    x_all = torch.cat([self.cached_intermediate_state[i], x], dim=1)
                    fx = encoder_i(x_all, output_seq=False)
                else:
                    fx = encoder_i(x, output_seq=output_seq_i)
            if self.skip_connection:
                fx = self.encoder[f'skip_connection_{block_id}'](fx, x)

            if self.encoder_output_type == EncoderOutputForm.HiddenStates:
                encoder2decoder.append(hx)
            elif self.encoder_output_type[i] == EncoderOutputForm.Sequence:
                encoder2decoder.append(fx)
            elif self.encoder_output_type[i] == EncoderOutputForm.SequenceLast:
                if output_seq or incremental_update:
                    encoder2decoder.append(fx)
                else:
                    encoder2decoder.append(encoder_i.get_last_seq_value(fx))
            if cache_intermediate_state:
                if self.encoder_has_hidden_states[i]:
                    self.cached_intermediate_state[i] = hx
                else:
                    if incremental_update:
                        self.cached_intermediate_state[i] = x_all
                    else:
                        self.cached_intermediate_state[i] = x
                    # otherwise the decoder does not exist for this layer
            x = fx
        if self.has_temporal_fusion:
            if incremental_update:
                self.cached_intermediate_state[i + 1] = torch.cat([self.cached_intermediate_state[i+1], x], dim=1)
            else:
                self.cached_intermediate_state[i + 1] = x
            return encoder2decoder, None
        else:
            return encoder2decoder, x


class StackedDecoder(nn.Module):
    def __init__(self,
                 network_structure: NetworkStructure,
                 encoder: nn.ModuleDict,
                 encoder_info: Dict[str, EncoderBlockInfo],
                 decoder_info: Dict[str, DecoderBlockInfo],
                 ):
        super().__init__()
        self.num_blocks = network_structure.num_blocks
        self.first_block = None
        self.skip_connection = network_structure.skip_connection

        self.decoder_has_hidden_states = []
        decoder = nn.ModuleDict()
        for i in range(1, self.num_blocks + 1):
            block_id = f'block_{i}'
            if block_id in decoder_info:
                self.first_block = i if self.first_block is None else self.first_block
                decoder[block_id] = decoder_info[block_id].decoder
                if decoder_info[block_id].decoder_properties.has_hidden_states:
                    self.decoder_has_hidden_states.append(True)
                else:
                    self.decoder_has_hidden_states.append(False)
                if self.skip_connection:
                    input_size_encoder = encoder_info[block_id].encoder_output_shape[-1]
                    skip_size_encoder = encoder_info[block_id].encoder_input_shape[-1]

                    input_size_decoder = decoder_info[block_id].decoder_output_shape[-1]
                    skip_size_decoder = decoder_info[block_id].decoder_input_shape[-1]
                    if input_size_encoder == input_size_decoder and skip_size_encoder == skip_size_decoder:
                        decoder[f'skip_connection_{i}'] = encoder[f'skip_connection_{i}']
                    else:
                        if network_structure.skip_connection_type == 'add':
                            decoder[f'skip_connection_{i}'] = AddLayer(input_size_decoder, skip_size_decoder)
                        elif network_structure.skip_connection_type == 'gate_add_norm':
                            decoder[f'skip_connection_{i}'] = GateAddNorm(input_size_decoder,
                                                                          hidden_size=input_size_decoder,
                                                                          skip_size=skip_size_decoder,
                                                                          dropout=network_structure.grn_dropout_rate)
        self.cached_intermediate_state = [torch.empty(0) for _ in range(self.num_blocks + 1 - self.first_block)]
        self.decoder = decoder

    def forward(self,
                x_future: Optional[torch.Tensor],
                encoder_output: List[torch.Tensor],
                cache_intermediate_state: bool = False,
                incremental_update: bool = False
                ) -> torch.Tensor:
        x = x_future
        for i, block_id in enumerate(range(self.first_block, self.num_blocks + 1)):
            decoder_i = self.decoder[f'block_{block_id}']  # type: DecoderNetwork
            if self.decoder_has_hidden_states[i]:
                if incremental_update:
                    hx = self.cached_intermediate_state[i]
                    fx, hx = decoder_i(x_future=x, encoder_output=hx)
                else:
                    fx, hx = decoder_i(x_future=x, encoder_output=encoder_output[i])
            else:
                if incremental_update:
                    x_all = torch.cat([self.cached_intermediate_state[i], x], dim=1)
                    fx = decoder_i(x_all, encoder_output=encoder_output[i])[:, -1:]
                else:
                    fx = decoder_i(x, encoder_output=encoder_output[i])
            if self.skip_connection:
                fx = self.decoder[f'skip_connection_{block_id}'](fx, x)
            if cache_intermediate_state:
                if self.encoder_has_hidden_states[i]:
                    self.cached_intermediate_state[i] = hx
                else:
                    if incremental_update:
                        self.cached_intermediate_state[i] = x_all
                    else:
                        self.cached_intermediate_state[i] = x
            x = fx
        return x