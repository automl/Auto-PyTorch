__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import unittest
import torch
import torch.nn as nn

from autoPyTorch.pipeline.base.pipeline import Pipeline
from autoPyTorch.pipeline.nodes.network_selector import NetworkSelector

from autoPyTorch.components.networks.feature.mlpnet import MlpNet
from autoPyTorch.components.networks.feature.shapedmlpnet import ShapedMlpNet

class TestNetworkSelectorMethods(unittest.TestCase):

    def test_network_selector(self):
        pipeline = Pipeline([
            NetworkSelector()
        ])

        selector = pipeline[NetworkSelector.get_name()]
        selector.add_network("mlpnet", MlpNet)
        selector.add_network("shapedmlpnet", ShapedMlpNet)
        selector.add_final_activation('none', nn.Sequential())

        pipeline_config = pipeline.get_pipeline_config()
        pipeline_config["random_seed"] = 42
        hyper_config = pipeline.get_hyperparameter_search_space().sample_configuration()
        pipeline.fit_pipeline(hyperparameter_config=hyper_config, pipeline_config=pipeline_config, 
                                X=torch.rand(3,3), Y=torch.rand(3, 2), embedding=nn.Sequential())

        sampled_network = pipeline[selector.get_name()].fit_output['network']

        self.assertIn(type(sampled_network), [MlpNet, ShapedMlpNet])



