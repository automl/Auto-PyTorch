import copy

import pytest

from autoPyTorch.pipeline.time_series_forecasting import TimeSeriesForecastingPipeline
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates


@pytest.fixture(params=['ForecastingNet', 'ForecastingSeq2SeqNet', 'ForecastingDeepARNet', 'NBEATSNet'])
def network_type(request):
    return request.param

@pytest.fixture(params=['NBEATSNet'])
def network_type(request):
    return request.param

class TestTimeSeriesForecastingPipeline:
    @pytest.mark.parametrize("fit_dictionary_forecasting", ["uni_variant_wo_missing",
                                                            "uni_variant_w_missing",
                                                            "multi_variant_wo_missing",
                                                            "multi_variant_w_missing",
                                                            "multi_variant_only_cat",
                                                            "multi_variant_only_num"], indirect=True)
    def test_fit_predict(self, fit_dictionary_forecasting, forecasting_budgets):
        dataset_properties = fit_dictionary_forecasting['dataset_properties']
        if not dataset_properties['uni_variant'] and len(dataset_properties['categories']) > 0:
            include = {'network_embedding': ['LearnedEntityEmbedding']}
        else:
            include = None
        pipeline = TimeSeriesForecastingPipeline(dataset_properties=dataset_properties,
                                                 include=include)
        step_names = pipeline.named_steps.keys()
        step_names_multi_processing = ['impute', 'scaler', 'encoding', 'time_series_transformer', 'preprocessing']

        steps_multi_in_pipeline = [step_name_multi in step_names for step_name_multi in step_names_multi_processing]

        if not dataset_properties['uni_variant']:
            assert sum(steps_multi_in_pipeline) == len(steps_multi_in_pipeline)
        else:
            assert sum(steps_multi_in_pipeline) == 0

        fit_dict = copy.copy(fit_dictionary_forecasting)
        pipeline = pipeline.fit(fit_dict)
        datamanager = fit_dictionary_forecasting['backend'].load_datamanager()
        test_sets = datamanager.generate_test_seqs()
        predict = pipeline.predict(test_sets)

        assert list(predict.shape) == [len(test_sets) * dataset_properties['n_prediction_steps']]

    @pytest.mark.parametrize("fit_dictionary_forecasting, forecasting_budgets", [
        ["multi_variant_wo_missing", 'resolution'],
        ["multi_variant_wo_missing", 'num_seq'],
        ["multi_variant_wo_missing", 'num_sample_per_seq'],
    ], indirect=True)
    def test_fit_budgets_types(self, fit_dictionary_forecasting, forecasting_budgets):
        dataset_properties = fit_dictionary_forecasting['dataset_properties']

        pipeline = TimeSeriesForecastingPipeline(dataset_properties=dataset_properties)
        fit_dict = copy.copy(fit_dictionary_forecasting)
        pipeline = pipeline.fit(fit_dict)
        datamanager = fit_dictionary_forecasting['backend'].load_datamanager()
        test_sets = datamanager.generate_test_seqs()
        predict = pipeline.predict(test_sets)

        assert list(predict.shape) == [len(test_sets) * dataset_properties['n_prediction_steps']]

    @pytest.mark.parametrize("fit_dictionary_forecasting", ["multi_variant_w_missing"], indirect=True)
    def test_networks(self, fit_dictionary_forecasting, network_type):
        dataset_properties = fit_dictionary_forecasting['dataset_properties']

        updates = HyperparameterSearchSpaceUpdates()

        if network_type == 'NBEATSNet':
            include = {'network_backbone': ['flat_encoder:NBEATSEncoder'],
                       'loss': ['RegressionLoss']}

            updates.append(node_name='network_backbone',
                           hyperparameter='flat_encoder:NBEATSDecoder:backcast_loss_ration',
                           value_range=[0.1, 0.9],
                           default_value=0.5)
        else:
            updates.append(node_name='network_backbone',
                           hyperparameter='seq_encoder:num_blocks',
                           value_range=[1, 1],
                           default_value=1)
            include = None
            if network_type == 'ForecastingNet':
                updates.append(node_name='network_backbone',
                               hyperparameter='seq_encoder:block_1:MLPDecoder:auto_regressive',
                               value_range=[False, ],
                               default_value=False)
                updates.append(node_name='network_backbone',
                               hyperparameter='seq_encoder:decoder_auto_regressive',
                               value_range=[False, ],
                               default_value=False)

            elif network_type == 'ForecastingSeq2SeqNet':
                include = {'network_backbone': ['seq_encoder']}
                updates.append(node_name='network_backbone',
                               hyperparameter='seq_encoder:decoder_auto_regressive',
                               value_range=[True, ],
                               default_value=True)

            elif network_type == 'ForecastingDeepARNet':
                include = {'network_backbone': ['seq_encoder:RNNEncoder'],
                           'loss': ['DistributionLoss']}

                updates.append(node_name='network_backbone',
                               hyperparameter='seq_encoder:block_1:MLPDecoder:auto_regressive',
                               value_range=[False, ],
                               default_value=False)

        pipeline = TimeSeriesForecastingPipeline(dataset_properties=dataset_properties,
                                                 include=include,
                                                 search_space_updates=updates)

        cs = pipeline.get_hyperparameter_search_space()

        pipeline.set_hyperparameters(cs.get_default_configuration())

        fit_dict = copy.copy(fit_dictionary_forecasting)
        pipeline = pipeline.fit(fit_dict)
        datamanager = fit_dictionary_forecasting['backend'].load_datamanager()
        test_sets = datamanager.generate_test_seqs()
        predict = pipeline.predict(test_sets)

        assert list(predict.shape) == [len(test_sets) * dataset_properties['n_prediction_steps']]
