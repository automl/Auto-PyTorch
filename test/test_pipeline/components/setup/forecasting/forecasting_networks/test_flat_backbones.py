import copy
import unittest
from test.test_pipeline.components.setup.forecasting.forecasting_networks.test_base_components import \
    generate_fit_dict_and_dataset_property

from ConfigSpace import Configuration

from sklearn.pipeline import Pipeline

import torch

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.cells import (
    StackedDecoder,
    StackedEncoder
)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.components_util import NetworkStructure
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder.MLPDecoder import (
    ForecastingMLPDecoder
)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder.\
    NBEATSDecoder import NBEATSDecoder
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.\
    flat_encoder import FlatForecastingEncoderChoice
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.flat_encoder.\
    MLPEncoder import MLPEncoder
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.flat_encoder.\
    NBEATSEncoder import NBEATSEncoder
from autoPyTorch.pipeline.components.setup.network_embedding.NoEmbedding import _NoEmbedding
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.forecasting_head import ForecastingHead


class TestFlatEncoder(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_properties, self.fit_dictionary = generate_fit_dict_and_dataset_property()
        self.fit_dictionary['net_output_type'] = 'regression'
        self.fit_dictionary['network_embedding'] = _NoEmbedding()

    def test_flat_encoder_choice(self):
        encoder_choices = FlatForecastingEncoderChoice(dataset_properties=self.dataset_properties)
        cs_flat = encoder_choices.get_hyperparameter_search_space(self.dataset_properties)
        available_encoder = cs_flat.get_hyperparameter("__choice__")

        self.assertTrue('MLPEncoder' in available_encoder.choices)
        self.assertTrue('NBEATSEncoder' in available_encoder.choices)

        sample = cs_flat.sample_configuration()
        encoder_choices.set_hyperparameters(sample)

        fit_dict = copy.copy(self.fit_dictionary)
        fit_dict['dataset_properties'] = self.dataset_properties
        self.assertIsInstance(encoder_choices.pipeline, Pipeline)
        encoder_choices = encoder_choices.fit(fit_dict)
        fit_dict = encoder_choices.transform(fit_dict)

        self.assertTrue('network_structure' in fit_dict)
        network_structure = fit_dict['network_structure']
        self.assertIsInstance(network_structure, NetworkStructure)
        self.assertTrue(network_structure.num_blocks, 1)

        self.assertTrue('network_encoder' in fit_dict)
        self.assertEqual(len(fit_dict['network_encoder']), 1)

        self.assertTrue('network_decoder' in fit_dict)
        self.assertEqual(len(fit_dict['network_decoder']), 1)

    def test_mlp_network(self):
        n_prediction_steps = self.dataset_properties['n_prediction_steps']
        network_structure = NetworkStructure()

        encoder_cfg = MLPEncoder().get_hyperparameter_search_space().get_default_configuration()
        encoder = MLPEncoder(**encoder_cfg)

        mlp_cs = ForecastingMLPDecoder.get_hyperparameter_search_space(self.dataset_properties,
                                                                       can_be_auto_regressive=True)
        mlp_cfg_non_ar_w_local = mlp_cs.get_default_configuration()
        mlp_cfg_non_ar_wo_local = copy.copy(mlp_cfg_non_ar_w_local.get_dictionary())

        mlp_cfg_non_ar_wo_local['has_local_layer'] = False
        mlp_cfg_non_ar_wo_local.pop('units_local_layer')

        mlp_cfg_non_ar_wo_local = Configuration(mlp_cs, values=mlp_cfg_non_ar_wo_local)

        decoder_w_local = ForecastingMLPDecoder(**mlp_cfg_non_ar_w_local)
        decoder_wo_local = ForecastingMLPDecoder(**mlp_cfg_non_ar_wo_local)

        decoders = {"non_ar_w_local": decoder_w_local,
                    "non_ar_wo_local": decoder_wo_local}

        fit_dict = copy.copy(self.fit_dictionary)
        fit_dict['dataset_properties'] = self.dataset_properties
        fit_dict['network_structure'] = network_structure

        encoder = encoder.fit(fit_dict)
        fit_dict = encoder.transform(fit_dict)

        for name, decoder in decoders.items():
            with self.subTest(decoder_name=name):
                fit_dict_ = copy.copy(fit_dict)

                decoder = decoder.fit(fit_dict_)
                fit_dict_ = decoder.transform(fit_dict_)

                input_tensor = torch.randn([10, 20, 3 + fit_dict_['X_train'].shape[-1]])
                input_tensor_future = torch.randn([10, n_prediction_steps, 2 + fit_dict_['X_train'].shape[-1]])

                head = ForecastingHead()
                head = head.fit(fit_dict_)
                fit_dict_ = head.transform(fit_dict_)

                net_encoder = StackedEncoder(network_structure, False,
                                             fit_dict_['network_encoder'], fit_dict_['network_decoder'])
                net_decoder = StackedDecoder(network_structure, net_encoder.encoder, fit_dict_['network_encoder'],
                                             fit_dict_['network_decoder'])

                head = fit_dict_['network_head']

                encoder2decoder, _ = net_encoder(input_tensor, [None])
                output = head(net_decoder(input_tensor_future, encoder2decoder))

                self.assertListEqual(list(output.shape), [10, n_prediction_steps, 1])

    def test_nbeats_network(self):
        n_prediction_steps = self.dataset_properties['n_prediction_steps']
        window_size = self.fit_dictionary['window_size']
        network_structure = NetworkStructure()

        encoder_cfg = NBEATSEncoder().get_hyperparameter_search_space().get_default_configuration()
        encoder = NBEATSEncoder(**encoder_cfg)

        nbeats_cs = NBEATSDecoder.get_hyperparameter_search_space(self.dataset_properties)

        nbeatsI_cfg = {
            "backcast_loss_ration": 0.0,
            "normalization": "LN",
            "activation": "relu",

            "n_beats_type": "I",

            "use_dropout_i": True,
            "num_stacks_i": 2,

            "num_blocks_i_1": 2,
            "num_layers_i_1": 2,
            "width_i_1": 16,
            "weight_sharing_i_1": True,
            "stack_type_i_1": 'trend',
            "expansion_coefficient_length_i_trend_1": 3,
            "dropout_i_1": 0.1,

            "num_blocks_i_2": 3,
            "num_layers_i_2": 2,
            "width_i_2": 16,
            "weight_sharing_i_2": False,
            "stack_type_i_2": 'seasonality',
            "expansion_coefficient_length_i_seasonality_2": 7,
            "dropout_i_2": 0.1,
        }

        nbeatsG_cfg = {
            "backcast_loss_ration": 0.0,
            "normalization": "NoNorm",
            "activation": "relu",

            "n_beats_type": "G",

            "use_dropout_g": True,
            "num_stacks_g": 2,

            "num_blocks_g": 1,
            "num_layers_g": 4,
            "width_g": 512,
            "weight_sharing_g": False,
            "expansion_coefficient_length_g": 32,
            "dropout_g": 0.1,
        }

        nbeatsI_cfg = Configuration(nbeats_cs, values=nbeatsI_cfg)
        nbeatsG_cfg = Configuration(nbeats_cs, values=nbeatsG_cfg)

        nbeats_i = NBEATSDecoder(**nbeatsI_cfg)
        nbeats_g = NBEATSDecoder(**nbeatsG_cfg)

        fit_dict = copy.copy(self.fit_dictionary)
        fit_dict['dataset_properties'] = self.dataset_properties
        fit_dict['network_structure'] = network_structure

        encoder = encoder.fit(fit_dict)
        fit_dict = encoder.transform(fit_dict)

        for decoder_idx, decoder in enumerate([nbeats_i, nbeats_g]):
            with self.subTest(decoder_idx=decoder_idx):
                fit_dict = copy.copy(fit_dict)
                fit_dict_ = copy.copy(fit_dict)

                decoder = decoder.fit(fit_dict_)
                fit_dict_ = decoder.transform(fit_dict_)

                input_tensor = torch.randn([10, 20, 1])

                head = ForecastingHead()
                head = head.fit(fit_dict_)
                fit_dict_ = head.transform(fit_dict_)

                encoder_net = fit_dict_['network_encoder']['block_1'].encoder
                decoder_net = fit_dict_['network_decoder']['block_1'].decoder
                idx_tracker = 0
                if decoder_idx == 0:
                    # only check nbeats_i
                    for i_stack in range(1, 1 + nbeatsI_cfg['num_stacks_i']):
                        num_blocks = nbeatsI_cfg[f'num_blocks_i_{i_stack}']
                        idx_end = idx_tracker + num_blocks
                        num_individual_models = len(set(decoder_net[idx_tracker:idx_end]))
                        if nbeatsI_cfg[f'weight_sharing_i_{i_stack}']:
                            self.assertEqual(num_individual_models, 1)
                        else:
                            self.assertEqual(num_individual_models, num_blocks)
                        idx_tracker = idx_end

                input_tensor = encoder_net(input_tensor, output_seq=False)

                for block in decoder_net:
                    backcast_block, forecast_block = block([None], input_tensor)
                    self.assertListEqual(list(backcast_block.shape), [10, window_size * 1])
                    self.assertListEqual(list(forecast_block.shape), [10, n_prediction_steps * 1])
