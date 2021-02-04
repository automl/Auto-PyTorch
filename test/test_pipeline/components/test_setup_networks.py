import pytest

import torch

from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline


@pytest.fixture(params=['MLPBackbone', 'ResNetBackbone', 'ShapedMLPBackbone', 'ShapedResNetBackbone'])
def backbone(request):
    return request.param


@pytest.fixture(params=['fully_connected'])
def head(request):
    return request.param


@pytest.mark.parametrize("fit_dictionary_tabular", ['classification_numerical_only',
                                                    'classification_categorical_only',
                                                    'classification_numerical_and_categorical'], indirect=True)
class TestNetworks:
    def test_pipeline_fit(self, fit_dictionary_tabular, backbone, head):
        """This test makes sure that the pipeline is able to fit
        given random combinations of hyperparameters across the pipeline"""

        pipeline = TabularClassificationPipeline(
            dataset_properties=fit_dictionary_tabular['dataset_properties'],
            include={'network_backbone': [backbone], 'network_head': [head]})
        cs = pipeline.get_hyperparameter_search_space()
        config = cs.get_default_configuration()

        assert backbone == config.get('network_backbone:__choice__', None)
        assert head == config.get('network_head:__choice__', None)
        pipeline.set_hyperparameters(config)
        # Need more epochs to make sure validation performance is met
        fit_dictionary_tabular['epochs'] = 100
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
        epoch2loss = run_summary.performance_tracker['val_loss']
        best_loss = min(list(epoch2loss.values()))
        epoch_where_best = list(epoch2loss.keys())[list(epoch2loss.values()).index(best_loss)]
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
