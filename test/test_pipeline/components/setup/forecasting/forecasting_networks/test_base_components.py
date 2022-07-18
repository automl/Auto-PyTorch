import copy
import itertools
import unittest

from ConfigSpace import Configuration

import numpy as np

import pandas as pd

import torch

from autoPyTorch.constants import TASK_TYPES_TO_STRING, TIMESERIES_FORECASTING
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone import ForecastingNetworkChoice
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.components_util import NetworkStructure
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder.MLPDecoder import (
    ForecastingMLPDecoder
)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder.components import (
    DecoderBlockInfo
)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.\
    base_forecasting_encoder import BaseForecastingEncoder
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.components import (
    EncoderBlockInfo,
    EncoderNetwork
)
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.distribution import (
    ALL_DISTRIBUTIONS,
    DisForecastingStrategy
)
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.forecasting_head import ForecastingHead
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdate


class DummyEmbedding(torch.nn.Module):
    def forward(self, x):
        if x.shape[-1] > 10:
            return x[..., :-10]
        return x


class DummyEncoderNetwork(EncoderNetwork):
    def forward(self, x, output_seq=False):
        if output_seq:
            return torch.ones((*x.shape[:-1], 10))
        return torch.ones((*x.shape[:-2], 1, 10))


class DummyForecastingEncoder(BaseForecastingEncoder):
    def n_encoder_output_feature(self):
        return 10

    def build_encoder(self, input_shape):
        return DummyEncoderNetwork()


class DummyTranformers():
    def __call__(self, x):
        return x[..., :(x.shape[-1] // 2)]


def generate_fit_dict_and_dataset_property():
    embedding = DummyEmbedding()

    transformation = [DummyTranformers()]
    n_prediction_steps = 3
    input_shape = (100, 50)
    output_shape = (n_prediction_steps, 1)
    time_feature_transform = [1, 2]

    feature_names = ('f1', 'f2', 'f3', 'f4', 'f5')
    feature_shapes = {'f1': 10, 'f2': 10, 'f3': 10, 'f4': 10, 'f5': 10}
    known_future_features = ('f1', 'f2', 'f3', 'f4', 'f5')

    dataset_properties = dict(input_shape=input_shape,
                              output_shape=output_shape,
                              transform_time_features=True,
                              time_feature_transform=time_feature_transform,
                              feature_shapes=feature_shapes,
                              known_future_features=known_future_features,
                              n_prediction_steps=n_prediction_steps,
                              encoder_can_be_auto_regressive=True,
                              feature_names=feature_names,
                              is_small_preprocess=True,
                              task_type=TASK_TYPES_TO_STRING[TIMESERIES_FORECASTING],
                              uni_variant=False,
                              future_feature_shapes=(n_prediction_steps, 50),
                              )

    fit_dictionary = dict(X_train=pd.DataFrame(np.random.randn(*input_shape)),
                          y_train=pd.DataFrame(np.random.randn(*output_shape)),
                          network_embedding=embedding,
                          preprocess_transforms=transformation,
                          transform_time_features=True,
                          window_size=5
                          )

    return dataset_properties, fit_dictionary


class TestForecastingNetworkBases(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_properties, self.fit_dictionary = generate_fit_dict_and_dataset_property()

        self.encoder = DummyForecastingEncoder()

        mlp_cs = ForecastingMLPDecoder.get_hyperparameter_search_space(self.dataset_properties,
                                                                       can_be_auto_regressive=True)
        mlp_cfg_non_ar_w_local = mlp_cs.get_default_configuration()
        mlp_cfg_non_ar_wo_local = copy.copy(mlp_cfg_non_ar_w_local.get_dictionary())

        mlp_cfg_non_ar_wo_local['has_local_layer'] = False
        mlp_cfg_non_ar_wo_local.pop('units_local_layer')

        mlp_cfg_ar = copy.copy(mlp_cfg_non_ar_wo_local)
        mlp_cfg_ar.pop('has_local_layer')
        mlp_cfg_ar['auto_regressive'] = True

        mlp_cfg_non_ar_wo_local = Configuration(mlp_cs, values=mlp_cfg_non_ar_wo_local)
        mlp_cfg_ar = Configuration(mlp_cs, values=mlp_cfg_ar)

        self.decoder_ar = ForecastingMLPDecoder(**mlp_cfg_ar)
        self.decoder_w_local = ForecastingMLPDecoder(**mlp_cfg_non_ar_w_local)
        self.decoder_wo_local = ForecastingMLPDecoder(**mlp_cfg_non_ar_wo_local)

        self.decoders = {"non_ar_w_local": self.decoder_w_local,
                         "non_ar_wo_local": self.decoder_wo_local,
                         "ar": self.decoder_ar}

    def test_encoder_choices(self):
        dataset_properties = {'task_type': TASK_TYPES_TO_STRING[TIMESERIES_FORECASTING]}
        encoder_choices = ForecastingNetworkChoice(dataset_properties)
        cs = encoder_choices.get_hyperparameter_search_space(dataset_properties)
        self.assertListEqual(list(cs.get_hyperparameter('__choice__').choices), ['flat_encoder', 'seq_encoder'])

        cs_only_flat = encoder_choices.get_hyperparameter_search_space(dataset_properties, include=['flat_encoder'])
        for hp_name in cs_only_flat.get_hyperparameter_names():
            self.assertFalse(hp_name.startswith('seq_encoder'))

        cs_only_flat = encoder_choices.get_hyperparameter_search_space(dataset_properties, include=['flat_encoder'])
        for hp_name in cs_only_flat.get_hyperparameter_names():
            self.assertFalse(hp_name.startswith('seq_encoder'))

        cs_only_rnn = encoder_choices.get_hyperparameter_search_space(dataset_properties,
                                                                      include=['seq_encoder:RNNEncoder'])

        self.assertListEqual(list(cs_only_rnn.get_hyperparameter('__choice__').choices), ['seq_encoder'])
        self.assertListEqual(list(cs_only_rnn.get_hyperparameter('seq_encoder:block_1:__choice__').choices),
                             ['RNNEncoder'])

        cs_no_rnn = encoder_choices.get_hyperparameter_search_space(dataset_properties,
                                                                    exclude=['seq_encoder:RNNEncoder'])
        for hp_name in cs_no_rnn.get_hyperparameter_names():
            self.assertFalse('RNNEncoder' in hp_name)

        sample = cs.sample_configuration()

        encoder_choices = encoder_choices.set_hyperparameters(sample)
        self.assertIsInstance(encoder_choices.choice.choice, BaseForecastingEncoder)

        encoder_choices = ForecastingNetworkChoice(dataset_properties)

        update_seq = HyperparameterSearchSpaceUpdate(node_name="network_backbone",
                                                     hyperparameter='__choice__',
                                                     value_range=('seq_encoder',),
                                                     default_value='seq_encoder', )

        encoder_choices._apply_search_space_update(update_seq)
        cs_seq = encoder_choices.get_hyperparameter_search_space(dataset_properties)
        self.assertListEqual(list(cs_seq.get_hyperparameter('__choice__').choices), ['seq_encoder'])

        encoder_choices = ForecastingNetworkChoice(dataset_properties)
        update_rnn_decoder_type = HyperparameterSearchSpaceUpdate(
            node_name="network_backbone",
            hyperparameter='seq_encoder:block_1:RNNEncoder:decoder_type',
            value_range=('MLPDecoder',),
            default_value='MLPDecoder', )
        encoder_choices._apply_search_space_update(update_rnn_decoder_type)
        cs_seq = encoder_choices.get_hyperparameter_search_space(dataset_properties)
        hp_rnn_decoder_type = cs_seq.get_hyperparameter(update_rnn_decoder_type.hyperparameter)
        self.assertListEqual(list(hp_rnn_decoder_type.choices), ['MLPDecoder'])

    def test_base_encoder(self):
        window_size = self.fit_dictionary['window_size']
        all_settings = [(True, False)] * 4
        for hp_values in itertools.product(*all_settings):
            uni_variant = hp_values[0]
            variable_selection = hp_values[1]
            transform_time_features = hp_values[2]
            is_small_preprocess = hp_values[3]
            with self.subTest(uni_variant=uni_variant,
                              variable_selection=variable_selection,
                              transform_time_features=transform_time_features,
                              is_small_preprocess=is_small_preprocess):
                network_structure = NetworkStructure(variable_selection=variable_selection)

                dataset_properties = copy.copy(self.dataset_properties)
                fit_dictionary = copy.copy(self.fit_dictionary)

                dataset_properties['is_small_preprocess'] = is_small_preprocess
                dataset_properties['uni_variant'] = uni_variant

                fit_dictionary['dataset_properties'] = self.dataset_properties
                fit_dictionary['network_structure'] = network_structure
                fit_dictionary['transform_time_features'] = transform_time_features
                fit_dictionary['dataset_properties'] = dataset_properties

                encoder_block_1 = copy.deepcopy(self.encoder)

                encoder_block_2 = copy.deepcopy(self.encoder)
                encoder_block_2.block_number = 2

                encoder_block_1 = encoder_block_1.fit(fit_dictionary)
                fit_dictionary = encoder_block_1.transform(fit_dictionary)
                network_encoder = fit_dictionary['network_encoder']
                self.assertIsInstance(network_encoder['block_1'], EncoderBlockInfo)
                self.assertEqual(network_encoder['block_1'].encoder_output_shape, (1, 10))

                if variable_selection:
                    self.assertEqual(network_encoder['block_1'].encoder_input_shape, (window_size, 10))
                else:
                    if uni_variant:
                        n_input_features = 0
                    else:
                        if is_small_preprocess:
                            n_input_features = 40
                        else:
                            n_input_features = 15

                    if transform_time_features:
                        n_input_features += len(dataset_properties['time_feature_transform'])

                    n_input_features += dataset_properties['output_shape'][-1]
                    self.assertEqual(network_encoder['block_1'].encoder_input_shape, (window_size,
                                                                                      n_input_features))

                encoder_block_2 = encoder_block_2.fit(fit_dictionary)
                fit_dictionary = encoder_block_2.transform(fit_dictionary)

                network_encoder = fit_dictionary['network_encoder']
                self.assertIsInstance(network_encoder['block_2'], EncoderBlockInfo)
                self.assertEqual(network_encoder['block_2'].encoder_output_shape, (1, 10))
                self.assertEqual(network_encoder['block_2'].encoder_input_shape, (1, 10))

    def test_base_decoder(self):
        n_prediction_steps = self.dataset_properties['n_prediction_steps']
        for variable_selection in (True, False):
            with self.subTest(variable_selection=variable_selection):
                network_structure = NetworkStructure(variable_selection=variable_selection, num_blocks=2)
                dataset_properties = copy.copy(self.dataset_properties)
                fit_dictionary = copy.copy(self.fit_dictionary)

                fit_dictionary['network_structure'] = network_structure
                fit_dictionary['dataset_properties'] = dataset_properties

                encoder_block_1 = copy.deepcopy(self.encoder)
                encoder_block_2 = copy.deepcopy(self.encoder)
                encoder_block_2.block_number = 2

                encoder_block_1 = encoder_block_1.fit(fit_dictionary)
                fit_dictionary = encoder_block_1.transform(fit_dictionary)
                encoder_block_2 = encoder_block_2.fit(fit_dictionary)
                fit_dictionary = encoder_block_2.transform(fit_dictionary)

                decoder1 = copy.deepcopy(self.decoder_w_local)
                decoder1 = decoder1.fit(fit_dictionary)
                self.assertEqual(decoder1.n_prediction_heads, n_prediction_steps)
                fit_dictionary = decoder1.transform(fit_dictionary)

                network_decoder = fit_dictionary['network_decoder']
                self.assertIsInstance(network_decoder['block_1'], DecoderBlockInfo)
                if variable_selection:
                    self.assertEqual(network_decoder['block_1'].decoder_input_shape,
                                     (n_prediction_steps, 10))  # Pure variable selection
                    self.assertEqual(network_decoder['block_1'].decoder_output_shape,
                                     (n_prediction_steps, 26))  # 10 (input features) + 16 (n_output_dims)
                else:
                    self.assertEqual(network_decoder['block_1'].decoder_input_shape,
                                     (n_prediction_steps, 52))  # 50 (input features) + 2 (time_transforms)
                    self.assertEqual(network_decoder['block_1'].decoder_output_shape,
                                     (n_prediction_steps, 68))  # 52 (input features) + 16 (n_out_dims)

                for name, decoder in self.decoders.items():
                    with self.subTest(decoder_name=name):
                        fit_dictionary_ = copy.deepcopy(fit_dictionary)
                        decoder2 = copy.deepcopy(decoder)
                        decoder2.block_number = 2
                        decoder2 = decoder2.fit(fit_dictionary_)
                        fit_dictionary_ = decoder2.transform(fit_dictionary_)
                        self.assertTrue(decoder2.is_last_decoder)
                        if name == 'ar':
                            self.assertEqual(fit_dictionary_['n_prediction_heads'], 1)
                        else:
                            self.assertEqual(fit_dictionary_['n_prediction_heads'], n_prediction_steps)
                        n_prediction_heads = fit_dictionary_['n_prediction_heads']

                        network_decoder = fit_dictionary_['network_decoder']['block_2']
                        self.assertIsInstance(network_decoder, DecoderBlockInfo)
                        if variable_selection:
                            self.assertEqual(network_decoder.decoder_input_shape, (n_prediction_heads, 26))

                            if name == 'non_ar_w_local':
                                # 26+16
                                self.assertEqual(network_decoder.decoder_output_shape, (n_prediction_heads, 42))
                            elif name == 'non_ar_wo_local':
                                # num_global
                                self.assertEqual(network_decoder.decoder_output_shape, (n_prediction_heads, 32))
                            elif name == 'ar':
                                self.assertEqual(network_decoder.decoder_output_shape, (n_prediction_heads, 32))  # 32
                        else:
                            self.assertEqual(network_decoder.decoder_input_shape, (n_prediction_heads, 68))

                            if name == 'non_ar_w_local':
                                # 26+16
                                self.assertEqual(network_decoder.decoder_output_shape, (n_prediction_heads, 84))
                            elif name == 'non_ar_wo_local':
                                # num_global
                                self.assertEqual(network_decoder.decoder_output_shape, (n_prediction_heads, 32))
                            elif name == 'ar':
                                self.assertEqual(network_decoder.decoder_output_shape, (n_prediction_heads, 32))  # 32

    def test_forecasting_heads(self):
        variable_selection = False
        n_prediction_steps = self.dataset_properties["n_prediction_steps"]

        network_structure = NetworkStructure(variable_selection=variable_selection, num_blocks=1)

        dataset_properties = copy.copy(self.dataset_properties)
        fit_dictionary = copy.copy(self.fit_dictionary)

        input_tensor = torch.randn([10, 20, 3 + fit_dictionary['X_train'].shape[-1]])
        input_tensor_future = torch.randn([10, n_prediction_steps, 2 + fit_dictionary['X_train'].shape[-1]])

        network_embedding = self.fit_dictionary['network_embedding']
        input_tensor = network_embedding(input_tensor)

        fit_dictionary['dataset_properties'] = self.dataset_properties
        fit_dictionary['network_structure'] = network_structure
        fit_dictionary['transform_time_features'] = True
        fit_dictionary['dataset_properties'] = dataset_properties
        encoder = copy.deepcopy(self.encoder)
        encoder = encoder.fit(fit_dictionary)
        fit_dictionary = encoder.transform(fit_dictionary)

        quantiles = [0.5, 0.1, 0.9]
        for name, decoder in self.decoders.items():
            with self.subTest(decoder_name=name):
                fit_dictionary_ = copy.deepcopy(fit_dictionary)
                decoder = decoder.fit(fit_dictionary_)
                fit_dictionary_ = decoder.transform(fit_dictionary_)

                for net_output_type in ['regression', 'distribution', 'quantile']:
                    def eval_heads_output(fit_dict):
                        head = ForecastingHead()
                        head = head.fit(fit_dict)
                        fit_dictionary_copy = head.transform(fit_dict)

                        encoder = fit_dictionary_copy['network_encoder']['block_1'].encoder
                        decoder = fit_dictionary_copy['network_decoder']['block_1'].decoder

                        head = fit_dictionary_copy['network_head']
                        output = head(decoder(input_tensor_future, encoder(input_tensor, output_seq=False)))
                        if name != "ar":
                            if net_output_type == 'regression':
                                self.assertListEqual(list(output.shape), [10, n_prediction_steps, 1])
                            elif net_output_type == 'distribution':
                                self.assertListEqual(list(output.sample().shape), [10, n_prediction_steps, 1])
                            elif net_output_type == 'quantile':
                                self.assertEqual(len(output), len(quantiles))
                                for output_quantile in output:
                                    self.assertListEqual(list(output_quantile.shape), [10, n_prediction_steps, 1])
                        else:
                            if net_output_type == 'regression':
                                self.assertListEqual(list(output.shape), [10, 1, 1])
                            elif net_output_type == 'distribution':
                                self.assertListEqual(list(output.sample().shape), [10, 1, 1])
                            elif net_output_type == 'quantile':
                                self.assertEqual(len(output), len(quantiles))
                                for output_quantile in output:
                                    self.assertListEqual(list(output_quantile.shape), [10, 1, 1])
                    with self.subTest(net_output_type=net_output_type):
                        fit_dictionary_copy = copy.deepcopy(fit_dictionary_)
                        fit_dictionary_copy['net_output_type'] = net_output_type

                        if net_output_type == 'distribution':
                            for dist in ALL_DISTRIBUTIONS.keys():
                                fit_dictionary_copy['dist_forecasting_strategy'] = DisForecastingStrategy(dist_cls=dist)
                                eval_heads_output(fit_dictionary_copy)
                        elif net_output_type == 'quantile':
                            fit_dictionary_copy['quantile_values'] = quantiles
                            eval_heads_output(fit_dictionary_copy)
                        else:
                            eval_heads_output(fit_dictionary_copy)
