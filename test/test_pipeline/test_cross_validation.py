__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import unittest
import numpy as np
import time

import torch
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from sklearn.model_selection import KFold, StratifiedKFold
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper

from autoPyTorch.pipeline.base.pipeline import Pipeline
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.pipeline.nodes.cross_validation import CrossValidation
from autoPyTorch.components.training.budget_types import BudgetTypeEpochs
from autoPyTorch.pipeline.nodes.create_dataset_info import DataSetInfo

class TestCrossValidationMethods(unittest.TestCase):


    def test_cross_validation(self):

        class ResultNode(PipelineNode):
            def fit(self, X, Y, train_indices, valid_indices):
                return { 'loss': np.sum(X[valid_indices]), 'info': {'a': np.sum(X[train_indices]), 'b': np.sum(X[valid_indices])} }

        pipeline = Pipeline([
            CrossValidation([
                ResultNode()
            ])
        ])

        pipeline["CrossValidation"].add_cross_validator("k_fold", KFold, lambda x: x.reshape((-1 ,)))
        pipeline["CrossValidation"].add_cross_validator("stratified_k_fold", StratifiedKFold, lambda x: x.reshape((-1 ,)))

        x_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y_train = np.array([[1], [0], [1]])

        # test cv_splits
        pipeline_config = pipeline.get_pipeline_config(cross_validator="k_fold", cross_validator_args={"n_splits": 3})
        pipeline_config_space = pipeline.get_hyperparameter_search_space(**pipeline_config)
        dataset_info = DataSetInfo()
        dataset_info.categorical_features = [None] * 3
        dataset_info.x_shape = x_train.shape
        dataset_info.y_shape = y_train.shape
        pipeline_config["random_seed"] = 42

        cv_result = pipeline.fit_pipeline(hyperparameter_config=pipeline_config_space, pipeline_config=pipeline_config, 
                                          X_train=x_train, Y_train=y_train, X_valid=None, Y_valid=None, 
                                          budget=5, budget_type=BudgetTypeEpochs, one_hot_encoder=None,
                                          optimize_start_time=time.time(), refit=False, dataset_info=dataset_info, rescore=False)

        self.assertEqual(cv_result['loss'], 15)
        self.assertDictEqual(cv_result['info'], {'a': 30, 'b': 15})

        
        # test validation split
        pipeline_config = pipeline.get_pipeline_config(validation_split=0.3)
        pipeline_config_space = pipeline.get_hyperparameter_search_space(**pipeline_config)
        pipeline_config['random_seed'] = 42
        dataset_info = DataSetInfo()
        dataset_info.categorical_features = [None] * 3
        dataset_info.x_shape = x_train.shape
        dataset_info.y_shape = y_train.shape

        cv_result = pipeline.fit_pipeline(hyperparameter_config=pipeline_config_space, pipeline_config=pipeline_config, 
                                          X_train=x_train, Y_train=y_train, X_valid=None, Y_valid=None, 
                                          budget=5, budget_type=BudgetTypeEpochs, one_hot_encoder=None,
                                          optimize_start_time=time.time(), refit=False, dataset_info=dataset_info, rescore=False)

        self.assertEqual(cv_result['loss'], 24)
        self.assertDictEqual(cv_result['info'], {'a': 21, 'b': 24})


        # test stratified cv split
        x_valid = x_train
        y_valid = y_train
        x_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
        y_train = np.array([[1], [1], [0], [0], [1], [0]])

        pipeline_config = pipeline.get_pipeline_config(cross_validator="stratified_k_fold", cross_validator_args={"n_splits": 3})
        pipeline_config_space = pipeline.get_hyperparameter_search_space(**pipeline_config)
        pipeline_config['random_seed'] = 42
        dataset_info = DataSetInfo()
        dataset_info.categorical_features = [None] * 3
        dataset_info.x_shape = x_train.shape
        dataset_info.y_shape = y_train.shape

        cv_result = pipeline.fit_pipeline(hyperparameter_config=pipeline_config_space, pipeline_config=pipeline_config, 
                                          X_train=x_train, Y_train=y_train, X_valid=None, Y_valid=None, 
                                          budget=5, budget_type=BudgetTypeEpochs, one_hot_encoder=None,
                                          optimize_start_time=time.time(), refit=False, dataset_info=dataset_info, rescore=False)

        self.assertEqual(cv_result['loss'], 57)
        self.assertDictEqual(cv_result['info'], {'a': 114, 'b': 57})

        pipeline_config = pipeline.get_pipeline_config()
        pipeline_config_space = pipeline.get_hyperparameter_search_space(**pipeline_config)
        pipeline_config['random_seed'] = 42
        dataset_info = DataSetInfo()
        dataset_info.categorical_features = [None] * 3
        dataset_info.x_shape = x_train.shape
        dataset_info.y_shape = y_train.shape

        cv_result = pipeline.fit_pipeline(hyperparameter_config=pipeline_config_space, pipeline_config=pipeline_config, 
                                          X_train=x_train, Y_train=y_train, X_valid=x_valid, Y_valid=y_valid, 
                                          budget=5, budget_type=BudgetTypeEpochs, one_hot_encoder=None,
                                          optimize_start_time=time.time(), refit=False, dataset_info=dataset_info, rescore=False)

        self.assertEqual(cv_result['loss'], 45)
        self.assertDictEqual(cv_result['info'], {'a': 171, 'b': 45})