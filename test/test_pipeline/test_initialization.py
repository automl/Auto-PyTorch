__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import unittest
import torch
import torch.nn as nn

from autoPyTorch.pipeline.base.pipeline import Pipeline
from autoPyTorch.pipeline.nodes.initialization_selector import InitializationSelector

from autoPyTorch.components.networks.feature.mlpnet import MlpNet
from autoPyTorch.components.networks.initialization import SparseInitialization, SimpleInitializer
from torch.nn import Linear


class TestInitializationSelectorMethods(unittest.TestCase):

    def test_initialization_selector(self):
        pipeline = Pipeline([
            InitializationSelector()
        ])

        selector = pipeline[InitializationSelector.get_name()]
        selector.add_initialization_method("sparse", SparseInitialization)
        selector.add_initializer('simple_initializer', SimpleInitializer)
        network = MlpNet({"activation": "relu", "num_layers": 1, "num_units_1": 10, "use_dropout": False}, in_features=5, out_features=1, embedding=None)

        pipeline_config = pipeline.get_pipeline_config()
        pipeline_config["random_seed"] = 42
        hyper_config = pipeline.get_hyperparameter_search_space().sample_configuration()
        hyper_config["InitializationSelector:initializer:initialize_bias"] = "No"
        hyper_config["InitializationSelector:initialization_method"] = "sparse"
        pipeline.fit_pipeline(hyperparameter_config=hyper_config, pipeline_config=pipeline_config, network=network)

        layer = [l for l in network.layers if isinstance(l, Linear)][0]
        self.assertEqual((layer.weight.data != 0).sum(), 5)



