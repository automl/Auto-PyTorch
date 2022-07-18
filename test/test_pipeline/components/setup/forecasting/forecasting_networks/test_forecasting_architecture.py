import copy
import unittest
from test.test_pipeline.components.setup.forecasting.forecasting_networks.test_base_components import \
    generate_fit_dict_and_dataset_property

import pytest

import torch

from autoPyTorch.pipeline.components.setup.forecasting_target_scaling.base_target_scaler import BaseTargetScaler
from autoPyTorch.pipeline.components.setup.network.forecasting_architecture import (
    AbstractForecastingNet,
    get_lagged_subsequences,
    get_lagged_subsequences_inference
)
from autoPyTorch.pipeline.components.setup.network.forecasting_network import ForecastingNetworkComponent
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone import ForecastingNetworkChoice
from autoPyTorch.pipeline.components.setup.network_embedding.NoEmbedding import _NoEmbedding
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.distribution import (
    ALL_DISTRIBUTIONS,
    DisForecastingStrategy
)
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.forecasting_head import ForecastingHead
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdate


class ReducedEmbedding(torch.nn.Module):
    # a dummy reduced embedding, it simply cut row for each categorical features
    def __init__(self, num_input_features, num_numerical_features: int):
        super(ReducedEmbedding, self).__init__()
        self.num_input_features = num_input_features
        self.num_numerical_features = num_numerical_features
        self.n_cat_features = len(num_input_features) - num_numerical_features

    def forward(self, x):
        x = x[..., :-self.n_cat_features]
        return x

    def get_partial_models(self, subset_features):
        num_numerical_features = sum([sf < self.num_numerical_features for sf in subset_features])
        num_input_features = [self.num_input_features[sf] for sf in subset_features]
        return ReducedEmbedding(num_input_features, num_numerical_features)


@pytest.fixture(params=['ForecastingNet', 'ForecastingSeq2SeqNet', 'ForecastingDeepARNet', 'NBEATSNet'])
def network_type(request):
    return request.param


@pytest.fixture(params=['RNNEncoder', 'TCNEncoder'])
def network_encoder(request):
    return request.param


@pytest.fixture(params=['ReducedEmbedding', 'NoEmbedding'])
def embedding(request):
    return request.param


@pytest.fixture(params=['distribution_mean', 'distribution_sample', 'regression', 'quantile'])
def net_output_type(request):
    return request.param


@pytest.fixture(params=[True, False])
def variable_selection(request):
    return request.param


@pytest.fixture(params=[True, False])
def with_static_features(request):
    return request.param


@pytest.fixture(params=[True, False])
def uni_variant_data(request):
    return request.param


class TestForecastingNetworks:
    dataset_properties, fit_dictionary = generate_fit_dict_and_dataset_property()

    def test_network_forward(self,
                             embedding,
                             net_output_type,
                             variable_selection,
                             with_static_features,
                             network_encoder,
                             network_type,
                             uni_variant_data):
        if network_type == 'ForecastingDeepARNet' and net_output_type != 'distribution_sample':
            return
        if network_type == 'ForecastingSeq2SeqNet' and network_encoder == 'TCNEncoder':
            return
        if network_type == 'NBEATSNet':
            # NBEATS only needs one pass
            if not (embedding == 'NoEmbedding' and net_output_type == 'regression'
                    and not variable_selection and not with_static_features and network_encoder == 'RNNEncoder'
                    and not uni_variant_data):
                return
        if uni_variant_data:
            if not (embedding == 'NoEmbedding' and not with_static_features):
                return

        dataset_properties = copy.copy(self.dataset_properties)
        time_feature_names = ('t1', 't2')
        dataset_properties['time_feature_names'] = time_feature_names

        if network_type != 'ForecastingDeepARNet':
            dataset_properties['known_future_features'] = ('f1', 'f3', 'f5')

        if with_static_features:
            dataset_properties['static_features'] = (0, 4)
        else:
            dataset_properties['static_features'] = tuple()

        fit_dictionary = copy.copy(self.fit_dictionary)
        fit_dictionary['dataset_properties'] = dataset_properties
        fit_dictionary['target_scaler'] = BaseTargetScaler(scaling_mode='standard').fit(fit_dictionary)

        if net_output_type.startswith("distribution"):
            fit_dictionary['dist_forecasting_strategy'] = DisForecastingStrategy(
                list(ALL_DISTRIBUTIONS.keys())[0],
                forecast_strategy=net_output_type.split("_")[1]
            )
            net_output_type = net_output_type.split("_")[0]
        elif net_output_type == 'quantile':
            fit_dictionary['quantile_values'] = [0.5, 0.1, 0.9]

        fit_dictionary['net_output_type'] = net_output_type

        if embedding == 'NoEmbedding':
            fit_dictionary['network_embedding'] = _NoEmbedding()
        else:
            fit_dictionary['network_embedding'] = ReducedEmbedding([10] * 5, 2)
            dataset_properties['feature_shapes'] = {'f1': 10, 'f2': 10, 'f3': 9, 'f4': 9, 'f5': 9}

        if uni_variant_data:
            fit_dictionary['X_train'] = None
            fit_dictionary['transform_time_features'] = False
            dataset_properties.update({'feature_shapes': {},
                                       'feature_names': tuple(),
                                       'known_future_features': tuple(),
                                       'uni_variant': True,
                                       'input_shape': (100, 0),
                                       'static_features': tuple(),
                                       'future_feature_shapes': (dataset_properties['n_prediction_steps'], 0),
                                       })

        n_prediction_steps = dataset_properties['n_prediction_steps']
        window_size = fit_dictionary['window_size']
        n_features_past = 10 * len(dataset_properties['feature_names']) + len(time_feature_names)
        n_features_future = 10 * len(dataset_properties['known_future_features']) + len(time_feature_names)
        n_targets = 1

        backbone = ForecastingNetworkChoice(dataset_properties)
        head = ForecastingHead()
        network = ForecastingNetworkComponent()

        if network_type == 'NBEATSNet':
            updates = [HyperparameterSearchSpaceUpdate(node_name="network_backbone",
                                                       hyperparameter='__choice__',
                                                       value_range=('flat_encoder',),
                                                       default_value='flat_encoder', )]
            include = ['flat_encoder:NBEATSEncoder']

        else:
            updates = [HyperparameterSearchSpaceUpdate(node_name="network_backbone",
                                                       hyperparameter='__choice__',
                                                       value_range=('seq_encoder',),
                                                       default_value='seq_encoder', ),
                       HyperparameterSearchSpaceUpdate(node_name="network_backbone",
                                                       hyperparameter='seq_encoder:num_blocks',
                                                       value_range=(1, 1),
                                                       default_value=1, ),
                       ]
            include = [f'seq_encoder:{network_encoder}']

            if network_type == 'ForecastingNet':
                updates.append(HyperparameterSearchSpaceUpdate(
                    node_name="network_backbone",
                    hyperparameter='seq_encoder:block_1:MLPDecoder:auto_regressive',
                    value_range=(False,),
                    default_value=False, ))

                updates.append(HyperparameterSearchSpaceUpdate(
                    node_name="network_backbone",
                    hyperparameter='seq_encoder:decoder_auto_regressive',
                    value_range=(False,),
                    default_value=False, ))
                if uni_variant_data and network_encoder == 'RNNEncoder':
                    updates.append(HyperparameterSearchSpaceUpdate(
                        node_name="network_backbone",
                        hyperparameter='seq_encoder:block_1:RNNEncoder:decoder_type',
                        value_range=('MLPDecoder',),
                        default_value='MLPDecoder', ))

            elif network_type == 'ForecastingSeq2SeqNet':
                updates.append(HyperparameterSearchSpaceUpdate(
                    node_name="network_backbone",
                    hyperparameter='seq_encoder:block_1:RNNEncoder:decoder_type',
                    value_range=("RNNDecoder",),
                    default_value="RNNDecoder", ))
                updates.append(HyperparameterSearchSpaceUpdate(
                    node_name="network_backbone",
                    hyperparameter='seq_encoder:decoder_auto_regressive',
                    value_range=(True,),
                    default_value=True, ))

            elif network_type == 'ForecastingDeepARNet':
                updates.append(HyperparameterSearchSpaceUpdate(
                    node_name="network_backbone",
                    hyperparameter='seq_encoder:block_1:RNNEncoder:decoder_type',
                    value_range=('MLPDecoder',),
                    default_value='MLPDecoder', ))

                updates.append(HyperparameterSearchSpaceUpdate(
                    node_name="network_backbone",
                    hyperparameter='seq_encoder:block_1:MLPDecoder:auto_regressive',
                    value_range=(True,),
                    default_value=True, ))

            if variable_selection:
                updates.append(HyperparameterSearchSpaceUpdate(
                    node_name="network_backbone",
                    hyperparameter='seq_encoder:variable_selection',
                    value_range=(True,),
                    default_value=True, ))
            else:
                updates.append(HyperparameterSearchSpaceUpdate(
                    node_name="network_backbone",
                    hyperparameter='seq_encoder:variable_selection',
                    value_range=(False,),
                    default_value=False, ))

        for update in updates:
            backbone._apply_search_space_update(update)

        cs = backbone.get_hyperparameter_search_space(dataset_properties=dataset_properties, include=include)

        sample = cs.sample_configuration()
        backbone.set_hyperparameters(sample)

        backbone = backbone.fit(fit_dictionary)
        fit_dictionary = backbone.transform(fit_dictionary)

        head = head.fit(fit_dictionary)
        fit_dictionary = head.transform(fit_dictionary)

        network = network.fit(fit_dictionary)
        fit_dictionary = network.transform(fit_dictionary)

        neu_arch = fit_dictionary['network']

        assert isinstance(neu_arch, AbstractForecastingNet)
        batch_size = 2

        past_targets = torch.ones([batch_size, 50, n_targets])
        future_targets = torch.ones([batch_size, n_prediction_steps, n_targets])
        past_observed_targets = torch.ones([batch_size, 50, n_targets]).bool()
        if uni_variant_data:
            past_features = None
            future_features = None
        else:
            past_features = torch.ones([batch_size, 50, n_features_past])
            future_features = torch.ones([batch_size, n_prediction_steps, n_features_future])

        output = neu_arch(past_targets=past_targets,
                          future_targets=future_targets,
                          past_features=past_features,
                          future_features=future_features,
                          past_observed_targets=past_observed_targets)

        if net_output_type.startswith('distribution'):
            assert isinstance(output, torch.distributions.Distribution)
            output = output.mean
        elif net_output_type == 'quantile':
            assert len(output) == 3
            output = output[0]
        if network_type in ["ForecastingNet", "ForecastingSeq2SeqNet"]:
            assert list(output.shape) == [batch_size, n_prediction_steps, n_targets]

        elif network_type == "ForecastingDeepARNet":
            assert list(output.shape) == [batch_size, n_prediction_steps + min(50, neu_arch.window_size) - 1, n_targets]
        else:
            backcast = output[0]
            forecast = output[1]
            assert list(backcast.shape) == [batch_size, window_size, n_targets]
            assert list(forecast.shape) == [batch_size, n_prediction_steps, n_targets]

        neu_arch.eval()
        output = neu_arch.predict(past_targets=past_targets,
                                  past_features=past_features,
                                  future_features=future_features,
                                  past_observed_targets=past_observed_targets)

        assert list(output.shape) == [batch_size, n_prediction_steps, n_targets]

        neu_arch.train()

        past_targets = torch.ones([batch_size, 3, n_targets])
        future_targets = torch.ones([batch_size, n_prediction_steps, n_targets])
        past_observed_targets = torch.ones([batch_size, 3, n_targets]).bool()
        if uni_variant_data:
            past_features = None
            future_features = None
        else:
            past_features = torch.ones([batch_size, 3, n_features_past])
            future_features = torch.ones([batch_size, n_prediction_steps, n_features_future])

        output = neu_arch(past_targets=past_targets,
                          future_targets=future_targets,
                          past_features=past_features,
                          future_features=future_features,
                          past_observed_targets=past_observed_targets)
        if net_output_type.startswith('distribution'):
            assert isinstance(output, torch.distributions.Distribution)
            output = output.mean
        elif net_output_type == 'quantile':
            assert len(output) == 3
            output = output[0]
        if network_type in ["ForecastingNet", "ForecastingSeq2SeqNet"]:
            assert list(output.shape) == [batch_size, n_prediction_steps, n_targets]
        elif network_type == "ForecastingDeepARNet":
            assert list(output.shape) == [batch_size, n_prediction_steps + min(3, neu_arch.window_size) - 1, n_targets]
        else:
            backcast = output[0]
            forecast = output[1]
            assert list(backcast.shape) == [batch_size, window_size, n_targets]
            assert list(forecast.shape) == [batch_size, n_prediction_steps, n_targets]

        if network_type in ["ForecastingNet", "ForecastingSeq2SeqNet"]:
            assert list(output.shape) == [batch_size, n_prediction_steps, n_targets]
        neu_arch.eval()

        output = neu_arch.predict(past_targets=past_targets,
                                  past_features=past_features,
                                  future_features=future_features,
                                  past_observed_targets=past_observed_targets)

        assert list(output.shape) == [batch_size, n_prediction_steps, n_targets]


class TestForecastingNetworkUtil(unittest.TestCase):
    def test_get_lagged_values(self):
        seq_raw = torch.arange(10).reshape([1, -1, 1]).float()
        window_size = 3
        lag_sequence = [0, 1, 2, 3, 5]
        lagged_seq1, mask = get_lagged_subsequences(seq_raw, window_size, lag_sequence)
        lagged_seq2, _ = get_lagged_subsequences(seq_raw, window_size, lag_sequence, mask)
        lagged_seq3 = get_lagged_subsequences_inference(seq_raw, window_size, lag_sequence)

        self.assertTrue(torch.equal(lagged_seq1, lagged_seq2))
        self.assertTrue(torch.equal(lagged_seq2, lagged_seq3))
        self.assertTrue(torch.equal(lagged_seq1[0], torch.Tensor([[7, 6, 5, 4, 2],
                                                                  [8, 7, 6, 5, 3],
                                                                  [9, 8, 7, 6, 4]]).float()))
        self.assertListEqual(list(mask.shape), [len(lag_sequence), max(lag_sequence) + window_size])

        seq_raw = torch.arange(5, 5 + 3).reshape([1, -1, 1]).float()
        window_size = 3
        lag_sequence = [0, 1, 2, 3, 5]
        lagged_seq1, mask = get_lagged_subsequences(seq_raw, window_size, lag_sequence)
        lagged_seq2, mask2 = get_lagged_subsequences(seq_raw, window_size, lag_sequence, mask)
        lagged_seq3 = get_lagged_subsequences_inference(seq_raw, window_size, lag_sequence)

        self.assertTrue(torch.all(lagged_seq1 == lagged_seq2))
        self.assertTrue(torch.all(lagged_seq2 == lagged_seq3))
        self.assertTrue(torch.equal(lagged_seq1[0], torch.Tensor([[5, 0, 0, 0, 0],
                                                                  [6, 5, 0, 0, 0],
                                                                  [7, 6, 5, 0, 0]]).float()))
