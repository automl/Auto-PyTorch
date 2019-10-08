__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import copy
import os
import json

from autoPyTorch.pipeline.base.pipeline import Pipeline

from autoPyTorch.pipeline.nodes.one_hot_encoding import OneHotEncoding
from autoPyTorch.pipeline.nodes.cross_validation import CrossValidation
from autoPyTorch.pipeline.nodes.metric_selector import MetricSelector
from autoPyTorch.pipeline.nodes.optimization_algorithm import OptimizationAlgorithm
from autoPyTorch.pipeline.nodes.create_dataset_info import CreateDatasetInfo
from autoPyTorch.pipeline.nodes.network_selector import NetworkSelector
from autoPyTorch.pipeline.nodes.image.network_selector_datasetinfo import NetworkSelectorDatasetInfo


from autoPyTorch.utils.config.config_file_parser import ConfigFileParser

class AutoNet():
    """Find an optimal neural network given a ML-task using BOHB"""
    preset_folder_name = None

    def __init__(self, config_preset="medium_cs", pipeline=None, **autonet_config):
        """Superclass for all AutoNet variations, that specifies the API of AutoNet.

        Keyword Arguments:
            pipeline {Pipeline} -- Define your own Autonet Pipeline (default: {None})
            **autonet_config -- Configure AutoNet for your needs. You can also configure AutoNet in fit(). Call print_help() for more info.
        """
        self.pipeline = pipeline or self.get_default_pipeline()
        self.base_config = autonet_config
        self.autonet_config = None
        self.fit_result = None
        self.dataset_info = None

        if config_preset is not None:
            parser = self.get_autonet_config_file_parser()
            c = parser.read(os.path.join(os.path.dirname(__file__), "presets",
                self.preset_folder_name, config_preset + ".txt"))
            c.update(self.base_config)
            self.base_config = c

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
        config_file_parser.print_help(self.base_config)

    def get_current_autonet_config(self):
        """Return the current AutoNet configuration
        
        Returns:
            dict -- The Configuration of AutoNet
        """

        if (self.autonet_config is not None):
            return self.autonet_config
        return self.pipeline.get_pipeline_config(**self.base_config)
    
    def get_hyperparameter_search_space(self, X_train=None, Y_train=None, X_valid=None, Y_valid=None, **autonet_config):
        """Return hyperparameter search space of Auto-PyTorch. Does depend on the dataset and the configuration!
        You can either pass the dataset and the configuration or use dataset and configuration of last fit call.
        
        Keyword Arguments:
            X_train {array} -- Training data. ConfigSpace depends on Training data.
            Y_train {array} -- Targets of training data.
            X_valid {array} -- Validation data. Will be ignored if cv_splits > 1. (default: {None})
            Y_valid {array} -- Validation data. Will be ignored if cv_splits > 1. (default: {None})
            autonet_config{dict} -- if not given and fit already called, config of last fit will be used
        
        Returns:
            ConfigurationSpace -- The configuration space that should be optimized.
        """
        X_train, Y_train, X_valid, Y_valid = self.check_data_array_types(X_train, Y_train, X_valid, Y_valid)
        dataset_info = self.dataset_info
        pipeline_config = dict(self.base_config, **autonet_config) if autonet_config else \
            self.get_current_autonet_config()
        if X_train is not None and Y_train is not None:
            dataset_info_node = self.pipeline[CreateDatasetInfo.get_name()]
            dataset_info = dataset_info_node.fit(pipeline_config=pipeline_config,
                                                 X_train=X_train,
                                                 Y_train=Y_train,
                                                 X_valid=X_valid,
                                                 Y_valid=Y_valid)["dataset_info"]

        return self.pipeline.get_hyperparameter_search_space(dataset_info=dataset_info, **pipeline_config)

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
            **autonet_config -- Configure AutoNet for your needs. You can also configure AutoNet in the constructor(). Call print_help() for more info.
        """
        X_train, Y_train, X_valid, Y_valid = self.check_data_array_types(X_train, Y_train, X_valid, Y_valid)
        self.autonet_config = self.pipeline.get_pipeline_config(**dict(self.base_config, **autonet_config))

        self.fit_result = self.pipeline.fit_pipeline(pipeline_config=self.autonet_config,
                                                     X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid)
        try:
            self.dataset_info = self.pipeline[CreateDatasetInfo.get_name()].fit_output["dataset_info"]
        except:
            self.dataset_info = None
        self.pipeline.clean()

        if "optimized_hyperparameter_config" not in self.fit_result.keys() or not self.fit_result["optimized_hyperparameter_config"]: # MODIFY
            raise RuntimeError("No models fit during training, please retry with a larger max_runtime.")
        
        if (refit):
            self.refit(X_train, Y_train, X_valid, Y_valid)
        return self.fit_result

    def refit(self, X_train, Y_train, X_valid=None, Y_valid=None, hyperparameter_config=None, autonet_config=None, budget=None, rescore=False):
        """Refit AutoNet to given hyperparameters. This will skip hyperparameter search.
        
        Arguments:
            X_train {array} -- Training data.
            Y_train {array} -- Targets of training data.
        
        Keyword Arguments:
            X_valid {array} -- Validation  data. (default: {None})
            Y_valid {array} -- Validation targets (default: {None})
            hyperparameter_config {dict} -- The hyperparameter config that specifies architecture and hyperparameters (default: {None})
            autonet_config -- Configure AutoNet for your needs. Call print_help() for more info.
            budget -- The budget used for the refit.
            rescore -- Use the same validation procedure as in fit (e.g. with cv).
        
        Raises:
            ValueError -- No hyperparameter config available
        """
        X_train, Y_train, X_valid, Y_valid = self.check_data_array_types(X_train, Y_train, X_valid, Y_valid)
        if (autonet_config is None):
            autonet_config = self.autonet_config
        if (autonet_config is None):
            autonet_config = self.base_config
        if (hyperparameter_config is None and self.fit_result):
            hyperparameter_config = self.fit_result["optimized_hyperparameter_config"]
        if (budget is None and self.fit_result):
            budget = self.fit_result["budget"]
        if (budget is None):
            budget = self.autonet_config["max_budget"]
        if (autonet_config is None or hyperparameter_config is None):
            raise ValueError("You have to specify a hyperparameter and autonet config in order to be able to refit")

        assert len(hyperparameter_config) > 0, "You have to specify a non-empty hyperparameter config for refit."

        refit_data = {'hyperparameter_config': hyperparameter_config,
                      'budget': budget,
                      'rescore': rescore}

        autonet_config = copy.deepcopy(autonet_config)
        autonet_config['cv_splits'] = 1
        autonet_config['increase_number_of_trained_datasets'] = False #if training multiple datasets else ignored

        return self.pipeline.fit_pipeline(pipeline_config=autonet_config, refit=refit_data,
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
        X, = self.check_data_array_types(X)
        autonet_config = self.get_current_autonet_config()

        Y_pred = self.pipeline.predict_pipeline(pipeline_config=autonet_config, X=X)['Y']

        # reverse one hot encoding
        if OneHotEncoding.get_name() in self.pipeline:
            OHE = self.pipeline[OneHotEncoding.get_name()]
            result = OHE.reverse_transform_y(Y_pred, OHE.fit_output['y_one_hot_encoder'])
            return result if not return_probabilities else (result, Y_pred)
        else:
            result = dict()
            result['Y'] = Y_pred
            return result if not return_probabilities else (result, Y_pred)

    def score(self, X_test, Y_test, return_loss_value=False):
        """Calculate the sore on test data using the specified optimize_metric
        
        Arguments:
            X_test {array} -- The test data matrix.
            Y_test {array} -- The test targets.
        
        Returns:
            score -- The score for the test data.
        """

        # Update config if needed
        X_test, Y_test = self.check_data_array_types(X_test, Y_test)
        autonet_config = self.get_current_autonet_config()

        res = self.pipeline.predict_pipeline(pipeline_config=autonet_config, X=X_test)
        if 'score' in res:
            # in case of default dataset like CIFAR10 - the pipeline will compute the score of the according pytorch test set
            return res['score']
        Y_pred = res['Y']
        # run predict pipeline
        #self.pipeline.predict_pipeline(pipeline_config=autonet_config, X=X_test)
        #Y_pred = self.pipeline[OptimizationAlgorithm.get_name()].predict_output['Y']
        
        
        # one hot encode Y
        try:
            OHE = self.pipeline[OneHotEncoding.get_name()]
            Y_test = OHE.transform_y(Y_test, OHE.fit_output['y_one_hot_encoder'])
        except:
            print("No one-hot encodig possible. Continuing without.")
            pass

        metric = self.pipeline[MetricSelector.get_name()].fit_output['optimize_metric']

        if return_loss_value:
            return metric.get_loss_value(Y_pred, Y_test)
        return metric(torch.from_numpy(Y_pred.astype(np.float32)), torch.from_numpy(Y_test.astype(np.float32)))

    def get_pytorch_model(self):
        """Returns a pytorch sequential model of the current incumbent configuration
        
        Arguments:
        
        Returns:
            model -- PyTorch sequential model of the current incumbent configuration
        """
        if NetworkSelector.get_name() in self.pipeline:
            return self.pipeline[NetworkSelector.get_name()].fit_output["network"].layers
        else:
            return self.pipeline[NetworkSelectorDatasetInfo.get_name()].fit_output["network"].layers

    def initialize_from_checkpoint(self, hyperparameter_config, checkpoint, in_features, out_features, final_activation=None):
        """

        Arguments:
            config_file: json with output as from .fit method
            in_features: array-like object, channels first
            out_features: int, number of classes
            final_activation:

        Returns:
            PyTorch Sequential model

        """
        # load state dict
        state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))["state"]

        # read config file
        if type(hyperparameter_config)==dict:
            config = hyperparameter_config
        else:
            with open(hyperparameter_config, 'r') as file:
                config = json.load(file)[1]

        # get model
        network_type = config['NetworkSelectorDatasetInfo:network']
        network_type = self.pipeline[NetworkSelectorDatasetInfo.get_name()].networks[network_type]
        model = network_type(config=config,
                             in_features=in_features,
                             out_features=out_features,
                             final_activation=final_activation)

        # Apply state dict
        pretrained_state = state_dict
        model_state = model.state_dict()

        pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
        model_state.update(pretrained_state)
        model.load_state_dict(model_state)

        # Add to pipeline
        self.pipeline[NetworkSelectorDatasetInfo.get_name()].fit_output["network"] = model

        return model
    
    def check_data_array_types(self, *arrays):
        result = []
        for array in arrays:
            if array is None or scipy.sparse.issparse(array):
                result.append(array)
                continue
            
            result.append(np.asanyarray(array))
            if not result[-1].shape:
                raise RuntimeError("Given data-array is of unexpected type %s. Please pass numpy arrays instead." % type(array))
        return result
