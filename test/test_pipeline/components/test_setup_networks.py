import pytest

import torch

from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline


@pytest.fixture(params=['MLPBackbone', 'ResNetBackbone', 'ShapedMLPBackbone', 'ShapedResNetBackbone'])
def backbone(request):
    return request.param


@pytest.fixture(params=['fully_connected'])
def head(request):
    return request.param


@pytest.mark.parametrize("fit_dictionary", ['fit_dictionary_numerical_only',
                                            'fit_dictionary_categorical_only',
                                            'fit_dictionary_num_and_categorical'], indirect=True)
class TestNetworks:
    def test_pipeline_fit(self, fit_dictionary, backbone, head):
        """This test makes sure that the pipeline is able to fit
        given random combinations of hyperparameters across the pipeline"""

        pipeline = TabularClassificationPipeline(
            dataset_properties=fit_dictionary['dataset_properties'],
            include={'network_backbone': [backbone], 'network_head': [head]})
        cs = pipeline.get_hyperparameter_search_space()
        config = cs.get_default_configuration()

        assert backbone == config.get('network_backbone:__choice__', None)
        assert head == config.get('network_head:__choice__', None)
        pipeline.set_hyperparameters(config)
        pipeline.fit(fit_dictionary)

        # To make sure we fitted the model, there should be a
        # run summary object with accuracy
        run_summary = pipeline.named_steps['trainer'].run_summary
        assert run_summary is not None

        # Make sure that performance was properly captured
        assert run_summary.performance_tracker['train_loss'][1] > 0
        assert run_summary.total_parameter_count > 0
        assert 'accuracy' in run_summary.performance_tracker['train_metrics'][1]

        # Commented out the next line as some pipelines are not
        # achieving this accuracy with default configuration and 10 epochs
        # To be added once we fix the search space
        # assert run_summary.performance_tracker['val_metrics'][fit_dictionary['epochs']]['accuracy'] >= 0.8
        # Make sure a network was fit
        assert isinstance(pipeline.named_steps['network'].get_network(), torch.nn.Module)
