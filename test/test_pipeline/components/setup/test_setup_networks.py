import flaky

import numpy as np

import pytest

import torch

from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline


@pytest.fixture(params=['MLPBackbone', 'ResNetBackbone', 'ShapedMLPBackbone', 'ShapedResNetBackbone'])
def backbone(request):
    return request.param


@pytest.fixture(params=['fully_connected'])
def head(request):
    return request.param


@pytest.fixture(params=['LearnedEntityEmbedding', 'NoEmbedding'])
def embedding(request):
    return request.param


@flaky.flaky(max_runs=3)
@pytest.mark.parametrize("fit_dictionary_tabular", ['classification_numerical_only',
                                                    'classification_categorical_only',
                                                    'classification_numerical_and_categorical'], indirect=True)
class TestNetworks:
    def test_pipeline_fit(self, fit_dictionary_tabular, embedding, backbone, head):
        """This test makes sure that the pipeline is able to fit
        every combination of network embedding, backbone, head"""

        # increase number of epochs to test for performance
        fit_dictionary_tabular['epochs'] = 50

        include = {'network_backbone': [backbone], 'network_head': [head], 'network_embedding': [embedding]}

        if len(fit_dictionary_tabular['dataset_properties']
               ['categorical_columns']) == 0 and embedding == 'LearnedEntityEmbedding':
            pytest.skip("Learned Entity Embedding is not used with numerical only data")
        pipeline = TabularClassificationPipeline(
            dataset_properties=fit_dictionary_tabular['dataset_properties'],
            include=include)

        cs = pipeline.get_hyperparameter_search_space()
        config = cs.get_default_configuration()

        assert embedding == config.get('network_embedding:__choice__', None)
        assert backbone == config.get('network_backbone:__choice__', None)
        assert head == config.get('network_head:__choice__', None)
        pipeline.set_hyperparameters(config)
        # Early stop to the best configuration seen
        fit_dictionary_tabular['early_stopping'] = 50

        pipeline.fit(fit_dictionary_tabular)

        # To make sure we fitted the model, there should be a
        # run summary object with accuracy
        run_summary = pipeline.named_steps['trainer'].run_summary
        assert run_summary is not None

        # Make sure that performance was properly captured
        assert run_summary.performance_tracker['train_loss'][1] > 0
        assert run_summary.total_parameter_count > 0
        assert 'accuracy' in run_summary.performance_tracker['train_metrics'][1]

        # Make sure default pipeline achieves a good score for dummy datasets
        epoch_where_best = int(np.argmax(
            [run_summary.performance_tracker['val_metrics'][e]['accuracy']
             for e in range(1, len(run_summary.performance_tracker['val_metrics']) + 1)]
        )) + 1  # Epochs start at 1
        score = run_summary.performance_tracker['val_metrics'][epoch_where_best]['accuracy']

        assert score >= 0.8, run_summary.performance_tracker['val_metrics']

        # Check that early stopping happened, if it did

        # We should not stop before patience
        assert run_summary.get_last_epoch() >= fit_dictionary_tabular['early_stopping']

        # we should not be greater than max allowed epoch
        assert run_summary.get_last_epoch() <= fit_dictionary_tabular['epochs']

        # every trained epoch has a val metric
        assert run_summary.get_last_epoch() == max(list(run_summary.performance_tracker['train_metrics'].keys()))

        epochs_since_best = run_summary.get_last_epoch() - run_summary.get_best_epoch()
        if epochs_since_best >= fit_dictionary_tabular['early_stopping']:
            assert run_summary.get_best_epoch() == epoch_where_best

        # Make sure a network was fit
        assert isinstance(pipeline.named_steps['network'].get_network(), torch.nn.Module)
