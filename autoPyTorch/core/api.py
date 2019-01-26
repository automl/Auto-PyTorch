__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


import numpy as np
import torch
import torch.nn as nn
import copy

from autoPyTorch.pipeline.base.pipeline import Pipeline

from autoPyTorch.pipeline.nodes.one_hot_encoding import OneHotEncoding
from autoPyTorch.pipeline.nodes.cross_validation import CrossValidation
from autoPyTorch.pipeline.nodes.metric_selector import MetricSelector
from autoPyTorch.pipeline.nodes.optimization_algorithm import OptimizationAlgorithm


from autoPyTorch.utils.config.config_file_parser import ConfigFileParser

class AutoNet():
    def __init__(self, pipeline=None, **autonet_config):
        """Superclass for all AutoNet variations, that specifies the API of AutoNet.
        
        Keyword Arguments:
            pipeline {Pipeline} -- Define your own Autonet Pipeline (default: {None})
            **autonet_config -- Configure AutoNet for your needs. You can also configure AutoNet in fit(). Call print_help() for more info.
        """
        self.pipeline = pipeline or self.get_default_pipeline()
        self.base_config = autonet_config
        self.autonet_config = None
        self.optimized_hyperparameter_config = None
        self.optimized_hyperparameter_config_budget = None

    def update_autonet_config(self, **autonet_config):
        """Update the configuration of AutoNet"""
        self.base_config.update(autonet_config)

    def get_autonet_config_file_parser(self):
        return ConfigFileParser(self.pipeline.get_pipeline_config_options())
    
    def print_help(self):
        """Print the kwargs to configure the current AutoNet Pipeline"""
        config_file_parser = self.get_autonet_config_file_parser()
        print("Configure AutoNet with the following keyword arguments.")
        print("Pass these arguments to either the constructor or fit().")
        print()
        config_file_parser.print_help()


    def get_current_autonet_config(self):
        """Return the current AutoNet configuration
        
        Returns:
            dict -- The Configuration of AutoNet
        """

        if (self.autonet_config is not None):
            return self.autonet_config
        return self.pipeline.get_pipeline_config(**self.base_config)
    
    def get_hyperparameter_search_space(self):
        """Return the hyperparameter search space of AutoNet
        
        Returns:
            ConfigurationSpace -- The ConfigurationSpace that should be optimized
        """

        return self.pipeline.get_hyperparameter_search_space(**self.get_current_autonet_config())

    @classmethod
    def get_default_pipeline(cls):
        """Build a pipeline for AutoNet. Should be implemented by child classes.
        
        Returns:
            Pipeline -- The Pipeline for AutoNet
        """

        # build the pipeline
        pipeline = Pipeline()
        
        cls._apply_default_pipeline_settings(pipeline)
        return pipeline

    @staticmethod
    def _apply_default_pipeline_settings(pipeline):
        """Apply some settings the pipeline. Should be implemented by child classes."""
        pass

    def fit(self, X_train, Y_train, X_valid=None, Y_valid=None, refit=True, **autonet_config):
        """Fit AutoNet to training data.
        
        Arguments:
            X_train {array} -- Training data.
            Y_train {array} -- Targets of training data.
        
        Keyword Arguments:
            X_valid {array} -- Validation data. Will be ignored if cv_splits > 1. (default: {None})
            Y_valid {array} -- Validation data. Will be ignored if cv_splits > 1. (default: {None})
            refit {bool} -- Whether final architecture should be trained again after search. (default: {True})
        
        Returns:
            optimized_hyperparameter_config -- The best found hyperparameter config.
            final_metric_score --  The final score of the specified train metric.
            **autonet_config -- Configure AutoNet for your needs. You can also configure AutoNet in the constructor(). Call print_help() for more info.
        """
        self.autonet_config = self.pipeline.get_pipeline_config(**dict(self.base_config, **autonet_config))

        self.pipeline.fit_pipeline(pipeline_config=self.autonet_config,
                                   X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid)

        output = self.pipeline[OptimizationAlgorithm.get_name()].fit_output
        self.optimized_hyperparameter_config = output["optimized_hyperparamater_config"]
        self.optimized_hyperparameter_config_budget = output["budget"]
        if (refit):
            self.refit(X_train, Y_train, X_valid, Y_valid, self.optimized_hyperparameter_config, self.autonet_config)
        return self.optimized_hyperparameter_config, output['final_metric_score']

    def refit(self, X_train, Y_train, X_valid=None, Y_valid=None, hyperparameter_config=None, autonet_config=None):
        """Refit AutoNet to given hyperparameters. This will skip hyperparameter search.
        
        Arguments:
            X_train {array} -- Training data.
            Y_train {array} -- Targets of training data.
        
        Keyword Arguments:
            X_valid {array} -- Validation  data. (default: {None})
            Y_valid {array} -- Validation targets (default: {None})
            hyperparameter_config {dict} -- The hyperparameter config that specifies architecture and hyperparameters (default: {None})
            **autonet_config -- Configure AutoNet for your needs. Call print_help() for more info.
        
        Raises:
            ValueError -- No hyperparameter config available
        """
        if (autonet_config is None):
            autonet_config = self.autonet_config
        if (hyperparameter_config is None):
            hyperparameter_config = self.optimized_hyperparameter_config
        if (autonet_config is None or hyperparameter_config is None):
            raise ValueError("You have to specify a hyperparameter and autonet config in order to be able to refit")

        assert len(hyperparameter_config) > 0, "You have to specify a non-empty hyperparameter config for refit. Probably something went wrong in fit."

        refit_data = {'hyperparameter_config': hyperparameter_config,
                      'budget': self.optimized_hyperparameter_config_budget / autonet_config['cv_splits']}
        autonet_config = copy.deepcopy(autonet_config)
        autonet_config['cv_splits'] = 1
        
        self.pipeline.fit_pipeline(pipeline_config=autonet_config, refit=refit_data,
                                    X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid)

    def predict(self, X, return_probabilities=False):
        """Predict the targets for a data matrix X.
        
        Arguments:
            X {array} -- The data matrix.
        
        Keyword Arguments:
            return_probabilities {bool} -- Whether to return a tuple, where the second entry is the true network output (default: {False})
        
        Returns:
            result -- The predicted targets.
        """

        # run predict pipeline
        self.pipeline.predict_pipeline(pipeline_config=self.autonet_config, X=X)
        Y_pred = self.pipeline[OptimizationAlgorithm.get_name()].predict_output['Y']

        # reverse one hot encoding 
        OHE = self.pipeline[OneHotEncoding.get_name()]
        result = OHE.reverse_transform_y(Y_pred, OHE.fit_output['y_one_hot_encoder'])
        return result if not return_probabilities else (result, Y_pred)

    def score(self, X_test, Y_test):
        """Calculate the sore on test data using the specified train_metric
        
        Arguments:
            X_test {array} -- The test data matrix.
            Y_test {array} -- The test targets.
        
        Returns:
            score -- The score for the test data.
        """

        # run predict pipeline
        self.pipeline.predict_pipeline(pipeline_config=self.autonet_config, X=X_test)
        Y_pred = self.pipeline[OptimizationAlgorithm.get_name()].predict_output['Y']
        
        # one hot encode Y
        OHE = self.pipeline[OneHotEncoding.get_name()]
        Y_test = OHE.transform_y(Y_test, OHE.fit_output['y_one_hot_encoder'])

        metric = self.pipeline[MetricSelector.get_name()].fit_output['train_metric']
        return metric(torch.from_numpy(Y_test), torch.from_numpy(Y_pred))
