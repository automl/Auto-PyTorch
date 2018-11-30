__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import unittest
import torch
import torch.nn as nn
import torch.optim as optim

from autonet.pipeline.base.pipeline import Pipeline
from autonet.pipeline.nodes.network_selector import NetworkSelector
from autonet.pipeline.nodes.optimizer_selector import OptimizerSelector

from autonet.components.networks.feature.mlpnet import MlpNet
from autonet.components.networks.feature.shapedmlpnet import ShapedMlpNet
from autonet.components.optimizer.optimizer import AutoNetOptimizerBase, AdamOptimizer, SgdOptimizer

class TestOptimizerSelectorMethods(unittest.TestCase):

    def test_selector(self):
        pipeline = Pipeline([
            NetworkSelector(),
            OptimizerSelector()
        ])

        net_selector = pipeline[NetworkSelector.get_name()]
        net_selector.add_network("mlpnet", MlpNet)
        net_selector.add_network("shapedmlpnet", ShapedMlpNet)
        net_selector.add_final_activation('none', nn.Sequential())

        opt_selector = pipeline[OptimizerSelector.get_name()]
        opt_selector.add_optimizer("adam", AdamOptimizer)
        opt_selector.add_optimizer("sgd", SgdOptimizer)

        pipeline_config = pipeline.get_pipeline_config()
        hyper_config = pipeline.get_hyperparameter_search_space().sample_configuration()
        pipeline.fit_pipeline(hyperparameter_config=hyper_config, pipeline_config=pipeline_config, 
                                X_train=torch.rand(3,3), Y_train=torch.rand(3, 2), embedding=nn.Sequential())

        sampled_optimizer = opt_selector.fit_output['optimizer']

        self.assertIn(type(sampled_optimizer), [optim.Adam, optim.SGD])



