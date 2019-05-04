__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import unittest
import netifaces
import logging
import numpy as np

import torch

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper

from autoPyTorch.pipeline.base.pipeline import Pipeline
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.pipeline.nodes.optimization_algorithm import OptimizationAlgorithm
from autoPyTorch.utils.config.config_option import ConfigOption
from hpbandster.core.result import json_result_logger

class TestOptimizationAlgorithmMethods(unittest.TestCase):

    def test_optimizer(self):

        class ResultNode(PipelineNode):
            def fit(self, X_train, Y_train):
                return {'loss': X_train.shape[1], 'info': {'train_a': X_train.shape[1], 'train_b': Y_train.shape[1]}}

            def get_hyperparameter_search_space(self, **pipeline_config):
                cs = CS.ConfigurationSpace()
                cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('hyper', lower=0, upper=30))
                return cs
            
            def get_pipeline_config_options(self):
                return [
                    ConfigOption("result_logger_dir", default=".", type="directory"),
                    ConfigOption("optimize_metric", default="a", type=str),
                ]

        logger = logging.getLogger('hpbandster')
        logger.setLevel(logging.ERROR)
        logger = logging.getLogger('autonet')
        logger.setLevel(logging.ERROR)

        pipeline = Pipeline([
            OptimizationAlgorithm([
                ResultNode()
            ])
        ])

        pipeline_config = pipeline.get_pipeline_config(num_iterations=1, budget_type='epochs', result_logger_dir=".")
        pipeline.fit_pipeline(pipeline_config=pipeline_config, X_train=np.random.rand(15,10), Y_train=np.random.rand(15, 5), X_valid=None, Y_valid=None,
            result_loggers=[json_result_logger(directory=".", overwrite=True)], dataset_info=None, shutdownables=[])

        result_of_opt_pipeline = pipeline[OptimizationAlgorithm.get_name()].fit_output['optimized_hyperparameter_config']
        print(pipeline[OptimizationAlgorithm.get_name()].fit_output)

        self.assertIn(result_of_opt_pipeline[ResultNode.get_name() + ConfigWrapper.delimiter + 'hyper'], list(range(0, 31)))
