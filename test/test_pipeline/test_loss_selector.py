__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import unittest
import torch
import torch.nn as nn

from autonet.pipeline.base.pipeline import Pipeline
from autonet.pipeline.nodes.loss_module_selector import LossModuleSelector
from autonet.components.preprocessing.loss_weight_strategies import LossWeightStrategyWeighted

class TestLossSelectorMethods(unittest.TestCase):

    def test_selector(self):
        pipeline = Pipeline([
            LossModuleSelector()
        ])

        selector = pipeline[LossModuleSelector.get_name()]
        selector.add_loss_module("L1", nn.L1Loss)
        selector.add_loss_module("cross_entropy", nn.CrossEntropyLoss, LossWeightStrategyWeighted(), True)

        pipeline_config = pipeline.get_pipeline_config(loss_modules=["L1", "cross_entropy"])
        pipeline_hyperparameter_config = pipeline.get_hyperparameter_search_space(**pipeline_config).sample_configuration()

        pipeline_hyperparameter_config["LossModuleSelector:loss_module"] = "L1"
        pipeline.fit_pipeline(hyperparameter_config=pipeline_hyperparameter_config, X_train=torch.rand(3,3), Y_train=torch.rand(3, 2), pipeline_config=pipeline_config, tmp=None)
        selected_loss = pipeline[selector.get_name()].fit_output['loss_function']
        self.assertEqual(type(selected_loss.function), nn.L1Loss)

        pipeline_hyperparameter_config["LossModuleSelector:loss_module"] = "cross_entropy"
        pipeline.fit_pipeline(hyperparameter_config=pipeline_hyperparameter_config, X_train=torch.rand(3,3), Y_train=torch.tensor([0, 1, 0]), pipeline_config=pipeline_config, tmp=None)
        selected_loss = pipeline[selector.get_name()].fit_output['loss_function']
        self.assertEqual(type(selected_loss.function), nn.CrossEntropyLoss)
        self.assertEqual(selected_loss(torch.tensor([[0.0, 10000.0]]), torch.tensor([[0, 1]])), 0)