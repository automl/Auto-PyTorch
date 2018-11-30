__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import unittest
import netifaces
import logging

import torch

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from autonet.utils.configspace_wrapper import ConfigWrapper

from autonet.pipeline.base.pipeline import Pipeline
from autonet.pipeline.base.pipeline_node import PipelineNode
from autonet.pipeline.nodes.optimization_algorithm import OptimizationAlgorithm

class TestOptimizationAlgorithmMethods(unittest.TestCase):

    def test_optimizer(self):

        class ResultNode(PipelineNode):
            def fit(self, X_train, Y_train):
                return {'loss': X_train.shape[1], 'info': {'a': X_train.shape[1], 'b': Y_train.shape[1]}}

            def get_hyperparameter_search_space(self, **pipeline_config):
                cs = CS.ConfigurationSpace()
                cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('hyper', lower=0, upper=30))
                return cs

        logger = logging.getLogger('hpbandster')
        logger.setLevel(logging.ERROR)
        logger = logging.getLogger('autonet')
        logger.setLevel(logging.ERROR)

        pipeline = Pipeline([
            OptimizationAlgorithm([
                ResultNode()
            ])
        ])

        pipeline_config = pipeline.get_pipeline_config(num_iterations=1, budget_type='epochs')
        pipeline.fit_pipeline(pipeline_config=pipeline_config, X_train=torch.rand(15,10), Y_train=torch.rand(15, 5), X_valid=None, Y_valid=None, one_hot_encoder=None)

        result_of_opt_pipeline = pipeline[OptimizationAlgorithm.get_name()].fit_output['optimized_hyperparamater_config']

        self.assertIn(result_of_opt_pipeline[ResultNode.get_name() + ConfigWrapper.delimiter + 'hyper'], list(range(0, 31)))