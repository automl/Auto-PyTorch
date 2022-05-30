import copy
import unittest
from itertools import product
from test.test_pipeline.components.setup.forecasting.forecasting_networks.test_base_components import \
    generate_fit_dict_and_dataset_property

from sklearn.pipeline import Pipeline

import torch

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.cells import (
    StackedDecoder,
    StackedEncoder,
    TemporalFusionLayer
)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.components_util import NetworkStructure
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.\
    seq_encoder import SeqForecastingEncoderChoice
from autoPyTorch.pipeline.components.setup.network_embedding.NoEmbedding import _NoEmbedding
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.forecasting_head import ForecastingHead
from autoPyTorch.utils.common import HyperparameterSearchSpace
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdate


class TestSeqEncoder(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_properties, self.fit_dictionary = generate_fit_dict_and_dataset_property()
        self.fit_dictionary['net_output_type'] = 'regression'
        self.fit_dictionary['network_embedding'] = _NoEmbedding()

    def test_config_space(self):
        seq_encoder_choice = SeqForecastingEncoderChoice(dataset_properties=self.dataset_properties)
        cs_seq = seq_encoder_choice.get_hyperparameter_search_space(
            dataset_properties=self.dataset_properties,
            num_blocks=HyperparameterSearchSpace(hyperparameter="num_blocks",
                                                 value_range=(2, 3),
                                                 default_value=2), )
        sample = cs_seq.sample_configuration()

        num_blocks = sample['num_blocks']
        seq_encoder_choice.set_hyperparameters(sample)

        fit_dict = copy.copy(self.fit_dictionary)
        fit_dict['dataset_properties'] = self.dataset_properties
        self.assertIsInstance(seq_encoder_choice.pipeline, Pipeline)
        encoder_choices = seq_encoder_choice.fit(fit_dict)
        fit_dict = encoder_choices.transform(fit_dict)

        self.assertTrue('network_structure' in fit_dict)
        network_structure = fit_dict['network_structure']
        self.assertIsInstance(network_structure, NetworkStructure)
        self.assertTrue(network_structure.num_blocks, num_blocks)

        self.assertTrue('network_encoder' in fit_dict)
        self.assertEqual(len(fit_dict['network_encoder']), num_blocks)

        self.assertTrue('network_decoder' in fit_dict)
        self.assertEqual(len(fit_dict['network_decoder']), num_blocks)

        # test error:
        dataset_properties = copy.copy(self.dataset_properties)
        dataset_properties.update({'feature_shapes': {},
                                   'feature_names': tuple(),
                                   'known_future_features': tuple(),
                                   'uni_variant': True,
                                   'input_shape': (100, 0),
                                   'static_features': tuple(),
                                   'future_feature_shapes': (dataset_properties['n_prediction_steps'], 0),
                                   })

    def test_deepar(self):
        for i, valid_encoder in enumerate(['RNNEncoder', 'TransformerEncoder', 'TCNEncoder', 'InceptionTimeEncoder']):
            with self.subTest(valid_encoder=valid_encoder):
                seq_encoder_choice = SeqForecastingEncoderChoice(dataset_properties=self.dataset_properties)
                update_ar = HyperparameterSearchSpaceUpdate(node_name="network_backbone",
                                                            hyperparameter='auto_regressive',
                                                            value_range=(True,),
                                                            default_value=True, )
                update_rnn_mlp = HyperparameterSearchSpaceUpdate(node_name="network_backbone",
                                                                 hyperparameter='decoder_type',
                                                                 value_range=('MLPDecoder',),
                                                                 default_value='MLPDecoder', )
                update_transformer_mlp = HyperparameterSearchSpaceUpdate(node_name="network_backbone",
                                                                         hyperparameter='decoder_type',
                                                                         value_range=('MLPDecoder',),
                                                                         default_value='MLPDecoder', )
                seq_encoder_choice._cs_updates = {"block_1:RNNEncoder:decoder_type": update_rnn_mlp,
                                                  "block_1:TransformerEncoder:decoder_type": update_transformer_mlp,
                                                  "block_1:MLPDecoder:auto_regressive": update_ar}

                cs_seq = seq_encoder_choice.get_hyperparameter_search_space(dataset_properties=self.dataset_properties,
                                                                            include=[valid_encoder])
                sample = cs_seq.get_default_configuration()

                seq_encoder_choice.set_hyperparameters(sample)

                fit_dict = copy.copy(self.fit_dictionary)
                fit_dict['dataset_properties'] = self.dataset_properties

                encoder_choices = seq_encoder_choice.fit(fit_dict)
                fit_dict = encoder_choices.transform(fit_dict)

                head = ForecastingHead()
                head = head.fit(fit_dict)
                fit_dict = head.transform(fit_dict)

                net_encoder = StackedEncoder(fit_dict['network_structure'], False,
                                             fit_dict['network_encoder'], fit_dict['network_decoder'])
                net_decoder = StackedDecoder(fit_dict['network_structure'], net_encoder.encoder,
                                             fit_dict['network_encoder'],
                                             fit_dict['network_decoder'])

                head = fit_dict['network_head']
                if i < 2:
                    input_tensor = torch.randn([10, 20, 59])  # 53 + 6(lag values)
                    input_tensor_future = torch.randn([10, 1, 59])
                else:
                    input_tensor = torch.randn([10, 20, 53])  # no lag
                    input_tensor_future = torch.randn([10, 1, 53])

                encoder2decoder, encoder_output = net_encoder(encoder_input=input_tensor,
                                                              additional_input=[None],
                                                              cache_intermediate_state=True,
                                                              )
                output = head(net_decoder(x_future=None, encoder_output=encoder2decoder))
                self.assertListEqual(list(output.shape), [10, 1, 1])

                encoder2decoder, encoder_output = net_encoder(encoder_input=input_tensor_future,
                                                              additional_input=[None],
                                                              output_seq=False, cache_intermediate_state=True,
                                                              incremental_update=True
                                                              )
                output = head(net_decoder(x_future=None, encoder_output=encoder2decoder))
                self.assertListEqual(list(output.shape), [10, 1, 1])

    def test_seq2seq(self):
        n_prediction_steps = self.dataset_properties['n_prediction_steps']

        for i, valid_encoder in enumerate(['RNNEncoder', 'TransformerEncoder']):
            with self.subTest(valid_encoder=valid_encoder):
                seq_encoder_choice = SeqForecastingEncoderChoice(dataset_properties=self.dataset_properties)

                update_rnn_rnn = HyperparameterSearchSpaceUpdate(node_name="network_backbone",
                                                                 hyperparameter='decoder_type',
                                                                 value_range=('RNNDecoder',),
                                                                 default_value='RNNDecoder', )
                update_trans_trans = HyperparameterSearchSpaceUpdate(node_name="network_backbone",
                                                                     hyperparameter='decoder_type',
                                                                     value_range=('TransformerDecoder',),
                                                                     default_value='TransformerDecoder', )

                seq_encoder_choice._cs_updates = {"block_1:RNNEncoder:decoder_type": update_rnn_rnn,
                                                  "block_1:TransformerEncoder:decoder_type": update_trans_trans}
                decoder_auto_regressive = HyperparameterSearchSpace(
                    hyperparameter="decoder_auto_regressive",
                    value_range=(True,),
                    default_value=True,
                )

                cs_seq = seq_encoder_choice.get_hyperparameter_search_space(
                    dataset_properties=self.dataset_properties,
                    decoder_auto_regressive=decoder_auto_regressive,
                    include=[valid_encoder]
                )
                sample = cs_seq.get_default_configuration()

                seq_encoder_choice.set_hyperparameters(sample)

                fit_dict = copy.copy(self.fit_dictionary)
                fit_dict['dataset_properties'] = self.dataset_properties

                encoder_choices = seq_encoder_choice.fit(fit_dict)
                fit_dict = encoder_choices.transform(fit_dict)

                head = ForecastingHead()
                head = head.fit(fit_dict)
                fit_dict = head.transform(fit_dict)

                net_encoder = StackedEncoder(fit_dict['network_structure'], False,
                                             fit_dict['network_encoder'], fit_dict['network_decoder'])
                net_decoder = StackedDecoder(fit_dict['network_structure'], net_encoder.encoder,
                                             fit_dict['network_encoder'],
                                             fit_dict['network_decoder'])

                head = fit_dict['network_head']

                input_tensor = torch.randn([10, 20, 59])  # 53 + 6(lag values)
                input_tensor_future = torch.randn([10, n_prediction_steps, 59])

                encoder2decoder, encoder_output = net_encoder(encoder_input=input_tensor,
                                                              additional_input=[None],
                                                              cache_intermediate_state=True,
                                                              )
                output = head(net_decoder(x_future=input_tensor_future, encoder_output=encoder2decoder))
                self.assertListEqual(list(output.shape), [10, n_prediction_steps, 1])

                net_encoder.eval()
                net_decoder.eval()
                input_tensor_future = torch.randn([10, 1, 59])

                encoder2decoder, encoder_output = net_encoder(encoder_input=input_tensor_future,
                                                              additional_input=[None],
                                                              output_seq=False, cache_intermediate_state=True,
                                                              incremental_update=True
                                                              )
                output = head(net_decoder(x_future=input_tensor_future, encoder_output=encoder2decoder))
                self.assertListEqual(list(output.shape), [10, 1, 1])

    def test_seq_models(self):
        update = HyperparameterSearchSpaceUpdate(node_name="network_backbone",
                                                 hyperparameter='auto_regressive',
                                                 value_range=(False,),
                                                 default_value=False, )
        # To avoid that default setting raises conflict for forbidden clauses
        update_rnn_default = HyperparameterSearchSpaceUpdate(node_name="network_backbone",
                                                             hyperparameter='decoder_type',
                                                             value_range=('MLPDecoder', 'RNNDecoder'),
                                                             default_value='RNNDecoder', )
        num_blocks = HyperparameterSearchSpace(hyperparameter="num_blocks",
                                               value_range=(2, 2),
                                               default_value=2)
        window_size: int = self.fit_dictionary['window_size']
        n_prediction_steps = self.dataset_properties['n_prediction_steps']
        n_features = self.dataset_properties['input_shape'][-1]
        n_targets = self.dataset_properties['output_shape'][-1]
        n_time_features = len(self.dataset_properties['time_feature_transform'])
        all_settings = [(True, False), (True, False), (True, False), (True, False), ('gate_add_norm', 'add')]
        for hp_values in product(*all_settings):
            hp_variable_selection = hp_values[0]
            hp_use_temporal_fusion = hp_values[1]
            hp_decoder_auto_regressive = hp_values[2]
            hp_skip_connection = hp_values[3]
            hp_skip_connection_type = hp_values[4]
            with self.subTest(hp_variable_selection=hp_variable_selection,
                              hp_use_temporal_fusion=hp_use_temporal_fusion,
                              hp_decoder_auto_regressive=hp_decoder_auto_regressive,
                              hp_skip_connection=hp_skip_connection,
                              hp_skip_connection_type=hp_skip_connection_type):
                variable_selection = HyperparameterSearchSpace('variable_selection',
                                                               (hp_variable_selection,), hp_variable_selection)
                use_temporal_fusion = HyperparameterSearchSpace('use_temporal_fusion',
                                                                (hp_use_temporal_fusion,), hp_use_temporal_fusion)
                decoder_auto_regressive = HyperparameterSearchSpace('decoder_auto_regressive',
                                                                    (hp_decoder_auto_regressive,),
                                                                    hp_decoder_auto_regressive)
                skip_connection = HyperparameterSearchSpace('skip_connection',
                                                            (hp_skip_connection,),
                                                            hp_skip_connection)
                skip_connection_type = HyperparameterSearchSpace('skip_connection_type',
                                                                 (hp_skip_connection_type,),
                                                                 hp_skip_connection_type)

                seq_encoder_choice = SeqForecastingEncoderChoice(dataset_properties=self.dataset_properties)
                seq_encoder_choice._cs_updates = {"block_1:MLPDecoder:auto_regressive": update,
                                                  "block_1:RNNEncoder:decoder_type": update_rnn_default,
                                                  "block_2:RNNEncoder:decoder_type": update_rnn_default,
                                                  }
                cs_seq_encoder = seq_encoder_choice.get_hyperparameter_search_space(
                    dataset_properties=self.dataset_properties,
                    num_blocks=num_blocks,
                    variable_selection=variable_selection,
                    use_temporal_fusion=use_temporal_fusion,
                    decoder_auto_regressive=decoder_auto_regressive,
                    skip_connection=skip_connection,
                    skip_connection_type=skip_connection_type
                )
                sample = cs_seq_encoder.sample_configuration()
                seq_encoder_choice.set_hyperparameters(sample)

                fit_dict = copy.copy(self.fit_dictionary)
                fit_dict['dataset_properties'] = self.dataset_properties

                encoder_choices = seq_encoder_choice.fit(fit_dict)
                fit_dict = encoder_choices.transform(fit_dict)

                head = ForecastingHead()
                head = head.fit(fit_dict)
                fit_dict = head.transform(fit_dict)

                network_structure = fit_dict['network_structure']
                net_encoder = StackedEncoder(fit_dict['network_structure'],
                                             network_structure.use_temporal_fusion,
                                             fit_dict['network_encoder'], fit_dict['network_decoder'])
                net_decoder = StackedDecoder(fit_dict['network_structure'], net_encoder.encoder,
                                             fit_dict['network_encoder'],
                                             fit_dict['network_decoder'])
                if hp_use_temporal_fusion:
                    temporal_fusion: TemporalFusionLayer = fit_dict['temporal_fusion']

                head = fit_dict['network_head']

                if hp_variable_selection:
                    n_feature_encoder = fit_dict['network_encoder']['block_1'].encoder_output_shape[-1]
                    if decoder_auto_regressive:
                        n_feature_decoder = n_feature_encoder
                    else:
                        n_feature_decoder = n_feature_encoder - 1
                else:
                    if hasattr(net_encoder.encoder['block_1'], 'lagged_value'):
                        n_feature_encoder = n_features + n_time_features
                        n_feature_encoder += n_targets * len(net_encoder.encoder['block_1'].lagged_value)
                    else:
                        n_feature_encoder = n_features + n_time_features + n_targets
                    if hp_decoder_auto_regressive:
                        if hasattr(net_decoder.decoder['block_1'], 'lagged_value'):
                            n_feature_decoder = n_features + n_time_features
                            n_feature_decoder += n_targets * len(
                                net_decoder.decoder['block_1'].lagged_value)
                        else:
                            n_feature_decoder = n_features + n_time_features + n_targets
                    else:
                        n_feature_decoder = n_features + n_time_features

                input_tensor = torch.ones([10, window_size, n_feature_encoder])
                input_tensor_future = torch.randn([10, n_prediction_steps, n_feature_decoder])
                input_tensor_future_ar = torch.randn([10, 1, n_feature_decoder])
                past_observed_values = torch.ones([10, window_size, 1]).bool()

                encoder2decoder, encoder_output = net_encoder(encoder_input=input_tensor,
                                                              additional_input=[None] * 2,
                                                              )

                decoder_output = net_decoder(x_future=input_tensor_future,
                                             encoder_output=encoder2decoder,
                                             pos_idx=(window_size, window_size + n_prediction_steps))

                if hp_use_temporal_fusion:
                    decoder_output = temporal_fusion(encoder_output=encoder_output,
                                                     decoder_output=decoder_output,
                                                     past_observed_targets=past_observed_values,
                                                     decoder_length=n_prediction_steps,
                                                     )

                output = head(decoder_output)
                self.assertListEqual(list(output.shape), [10, n_prediction_steps, 1])

                if hp_decoder_auto_regressive:
                    net_encoder.eval()
                    net_decoder.eval()
                    head.eval()

                    encoder2decoder, encoder_output = net_encoder(encoder_input=input_tensor,
                                                                  additional_input=[None] * 2,
                                                                  cache_intermediate_state=False,
                                                                  )

                    decoder_output = net_decoder(x_future=input_tensor_future_ar,
                                                 encoder_output=encoder2decoder,
                                                 pos_idx=(window_size, window_size + 1),
                                                 cache_intermediate_state=True,
                                                 )
                    if hp_use_temporal_fusion:
                        temporal_fusion.eval()
                        decoder_output = temporal_fusion(encoder_output=encoder_output,
                                                         decoder_output=decoder_output,
                                                         past_observed_targets=past_observed_values,
                                                         decoder_length=1,
                                                         )
                        output = head(decoder_output)
                        self.assertListEqual(list(output.shape), [10, 1, 1])

                    decoder_output = net_decoder.forward(x_future=input_tensor_future_ar,
                                                         encoder_output=encoder2decoder,
                                                         pos_idx=(window_size, window_size + 1),
                                                         cache_intermediate_state=True,
                                                         incremental_update=True,
                                                         )
                    if hp_use_temporal_fusion:
                        decoder_output = temporal_fusion(encoder_output=encoder_output,
                                                         decoder_output=decoder_output,
                                                         past_observed_targets=past_observed_values,
                                                         decoder_length=1,
                                                         )
                        output = head(decoder_output)
                        self.assertListEqual(list(output.shape), [10, 1, 1])
