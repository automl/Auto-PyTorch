import copy
import unittest
import torch

from test.test_pipeline.components.setup.forecasting.forecasting_networks.test_base_components import (
    generate_fit_dict_and_dataset_property
)

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.seq_encoder \
    import SeqForecastingEncoderChoice
from autoPyTorch.pipeline.components.setup.network_embedding.NoEmbedding import _NoEmbedding
from autoPyTorch.utils.common import HyperparameterSearchSpace, get_hyperparameter
from sklearn.pipeline import Pipeline
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.components_util import NetworkStructure
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.forecasting_head import ForecastingHead

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.seq_encoder. \
    RNNEncoder import RNNEncoder
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.seq_encoder. \
    TCNEncoder import TCNEncoder
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.seq_encoder. \
    TransformerEncoder import TransformerEncoder
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.seq_encoder. \
    RNNEncoder import RNNEncoder

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder. \
    RNNDecoder import ForecastingRNNDecoder
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder. \
    MLPDecoder import ForecastingMLPDecoder
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder. \
    TransformerDecoder import ForecastingTransformerDecoder
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdate

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.cells import (
    StackedEncoder,
    StackedDecoder
)


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

    def test_deepar(self):
        for i, valid_encoder in enumerate(['RNNEncoder', 'TCNEncoder', 'TransformerEncoder']):
            seq_encoder_choice = SeqForecastingEncoderChoice(dataset_properties=self.dataset_properties)
            update = HyperparameterSearchSpaceUpdate(node_name="network_backbone",
                                                     hyperparameter='auto_regressive',
                                                     value_range=(True,),
                                                     default_value=True, )
            seq_encoder_choice._cs_updates = {"block_1:MLPDecoder:auto_regressive": update}
            cs_seq = seq_encoder_choice.get_hyperparameter_search_space(dataset_properties=self.dataset_properties,
                                                                        include=[valid_encoder])
            sample = cs_seq.get_default_configuration()
            seq_encoder_choice.set_hyperparameters(copy.copy(sample))

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
            if i != 1:
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
            self.assertListEqual(list(output.shape), [10, 1])

            encoder2decoder, encoder_output = net_encoder(encoder_input=input_tensor_future,
                                                          additional_input=[None],
                                                          output_seq=False, cache_intermediate_state=True,
                                                          incremental_update=True
                                                          )
            output = head(net_decoder(x_future=None, encoder_output=encoder2decoder))
            self.assertListEqual(list(output.shape), [10, 1, 1])

