import copy
import unittest
from unittest import mock

import pandas as pd
import numpy as np
import torch
from autoPyTorch.constants import (
    TASK_TYPES_TO_STRING,
    TIMESERIES_FORECASTING,
)

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone import ForecastingNetworkChoice
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder. \
    base_forecasting_encoder import BaseForecastingEncoder
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder. \
    base_forecasting_decoder import BaseForecastingDecoder
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder. \
    MLPDecoder import ForecastingMLPDecoder
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder.components import (
    DecoderBlockInfo, DecoderProperties
)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.components import (
    EncoderProperties, EncoderBlockInfo, EncoderNetwork
)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.components_util import NetworkStructure
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.forecasting_head import ForecastingHead


class DummyEmbedding(torch.nn.Module):
    def forward(self, x):
        if x.shape[-1] > 10:
            return x[..., :-10]
        return x

class DummyEncoderNetwork(EncoderNetwork):
    def forward(self, x, output_seq=False):
        if output_seq:
            return torch.ones(x.shape[:-1])
        return torch.ones((*x.shape[:-1], 10))


class DummyForecastingEncoder(BaseForecastingEncoder):
    def n_encoder_output_feature(self):
        return 10

    def build_encoder(self, input_shape):
        return DummyEncoderNetwork()


class DummyTranformers():
    def __call__(self, x):
        return x[..., :(x.shape[-1] // 2)]


class TestForecastingNetworkBases(unittest.TestCase):
    def setUp(self) -> None:
        embedding = DummyEmbedding()

        transformation = [DummyTranformers()]

        input_shape = (100, 50)
        output_shape = (100, 1)
        time_feature_transform = [1, 2]

        feature_shapes = {'f1': 10, 'f2': 10, 'f3': 10, 'f4': 10, 'f5': 10}
        known_future_features = ('f1', 'f2', 'f5')

        self.encoder = DummyForecastingEncoder()

        with mock.patch('autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.'
                        'forecasting_decoder.base_forecasting_decoder.BaseForecastingDecoder') as MockDecoder:
            mockdecoder = MockDecoder.return_value
            mockdecoder._build_decoder.return_values = (None, 10)
        self.decoder = mockdecoder

        self.dataset_properties = dict(input_shape=input_shape,
                                       output_shape=output_shape,
                                       transform_time_features=True,
                                       time_feature_transform=time_feature_transform,
                                       feature_shapes=feature_shapes,
                                       known_future_features=known_future_features,
                                       )

        self.fit_dictionary = dict(X_train=pd.DataFrame(np.random.randn(*input_shape)),
                                   y_train=pd.DataFrame(np.random.randn(*output_shape)),
                                   network_embedding=embedding,
                                   preprocess_transforms=transformation,
                                   window_size=3
                                   )

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
        self.assertListEqual(list(cs_only_rnn.get_hyperparameter('seq_encoder:block_1:__choice__').choices), ['RNNEncoder'])

        cs_no_rnn = encoder_choices.get_hyperparameter_search_space(dataset_properties,
                                                                    exclude=['seq_encoder:RNNEncoder'])
        for hp_name in cs_no_rnn.get_hyperparameter_names():
            self.assertFalse('RNNEncoder' in hp_name)

    def test_base_encoder(self):
        window_size = self.fit_dictionary['window_size']
        for uni_variant in (True, False):
            for variable_selection in (True, False):
                for transform_time_features in (True, False):
                    for is_small_preprocess in (True, False):
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
                        self.assertEqual(network_encoder['block_1'].encoder_output_shape, (window_size, 10))

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
                        self.assertEqual(network_encoder['block_2'].encoder_output_shape, (window_size, 10))
                        self.assertEqual(network_encoder['block_2'].encoder_input_shape, (window_size,
                                                                                          10))


