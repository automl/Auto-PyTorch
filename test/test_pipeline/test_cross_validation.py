__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import unittest
import numpy as np
import time

import torch
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from autonet.utils.configspace_wrapper import ConfigWrapper

from autonet.pipeline.base.pipeline import Pipeline
from autonet.pipeline.base.pipeline_node import PipelineNode
from autonet.pipeline.nodes.cross_validation import CrossValidation
from autonet.training.budget_types import BudgetTypeEpochs

class TestCrossValidationMethods(unittest.TestCase):


    def test_cross_validation(self):

        class ResultNode(PipelineNode):
            def fit(self, X_train, X_valid):
                return { 'loss': np.sum(X_valid), 'info': {'a': np.sum(X_train), 'b': np.sum(X_valid)} }

        pipeline = Pipeline([
            CrossValidation([
                ResultNode()
            ])
        ])

        x_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y_train = np.array([[1], [0], [1]])

        # test cv_splits
        pipeline_config = pipeline.get_pipeline_config(cv_splits=3)
        pipeline_config_space = pipeline.get_hyperparameter_search_space(**pipeline_config)
        pipeline_config['categorical_features'] = None

        cv_result = pipeline.fit_pipeline(hyperparameter_config=pipeline_config_space, pipeline_config=pipeline_config, 
                                          X_train=x_train, Y_train=y_train, X_valid=None, Y_valid=None, 
                                          budget=5, budget_type=BudgetTypeEpochs, one_hot_encoder=None,
                                          optimize_start_time=time.time())

        self.assertEqual(cv_result['loss'], 15)
        self.assertDictEqual(cv_result['info'], {'a': 30, 'b': 15})

        
        # test validation split
        pipeline_config = pipeline.get_pipeline_config(validation_split=0.3)
        pipeline_config_space = pipeline.get_hyperparameter_search_space(**pipeline_config)
        pipeline_config['categorical_features'] = None

        cv_result = pipeline.fit_pipeline(hyperparameter_config=pipeline_config_space, pipeline_config=pipeline_config, 
                                          X_train=x_train, Y_train=y_train, X_valid=None, Y_valid=None, 
                                          budget=5, budget_type=BudgetTypeEpochs, one_hot_encoder=None,
                                          optimize_start_time=time.time())

        self.assertEqual(cv_result['loss'], 24)
        self.assertDictEqual(cv_result['info'], {'a': 21, 'b': 24})


        # test stratified cv split
        x_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
        y_train = np.array([[1], [1], [0], [0], [1], [0]])

        pipeline_config = pipeline.get_pipeline_config(cv_splits=3, use_stratified_cv_split=True)
        pipeline_config_space = pipeline.get_hyperparameter_search_space(**pipeline_config)
        pipeline_config['categorical_features'] = None

        cv_result = pipeline.fit_pipeline(hyperparameter_config=pipeline_config_space, pipeline_config=pipeline_config, 
                                          X_train=x_train, Y_train=y_train, X_valid=None, Y_valid=None, 
                                          budget=5, budget_type=BudgetTypeEpochs, one_hot_encoder=None,
                                          optimize_start_time=time.time())

        self.assertEqual(cv_result['loss'], 57)
        self.assertDictEqual(cv_result['info'], {'a': 114, 'b': 57})