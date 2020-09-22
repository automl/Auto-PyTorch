import os
import torch
import logging
import numpy as np
from autoPyTorch.core.api import AutoNet
from autoPyTorch.pipeline.base.pipeline import Pipeline
from autoPyTorch.pipeline.nodes.one_hot_encoding import OneHotEncoding
from autoPyTorch.pipeline.nodes.metric_selector import MetricSelector
from autoPyTorch.pipeline.nodes.ensemble import EnableComputePredictionsForEnsemble, SavePredictionsForEnsemble, BuildEnsemble, EnsembleServer
from autoPyTorch.pipeline.nodes.create_dataset_info import CreateDatasetInfo
from autoPyTorch.pipeline.nodes.network_selector import NetworkSelector
from autoPyTorch.pipeline.nodes import BaselineTrainer

from IPython import embed

class AutoNetEnsemble(AutoNet):
    """Build an ensemble of several neural networks that were evaluated during the architecure search"""

    # OVERRIDE
    def __init__(self, autonet, config_preset="medium_cs", **autonet_config):
        if isinstance(autonet, AutoNet):
            self.pipeline = autonet.pipeline
            self.autonet_type = type(autonet)
            self.base_config = autonet.base_config
            self.autonet_config = autonet.autonet_config
            self.fit_result = autonet.fit_result
        elif issubclass(autonet, AutoNet):
            self.pipeline = autonet.get_default_ensemble_pipeline()
            self.autonet_type = autonet
            self.base_config = dict()
            self.autonet_config = None
            self.fit_result = None
        else:
            raise("Invalid autonet argument")
        
        assert EnableComputePredictionsForEnsemble in self.pipeline
        assert SavePredictionsForEnsemble in self.pipeline
        assert EnsembleServer in self.pipeline
        assert BuildEnsemble in self.pipeline

        self.base_config.update(autonet_config)
        self.trained_autonets = None
        self.dataset_info = None

        if config_preset is not None:
            parser = self.get_autonet_config_file_parser()
            c = parser.read(os.path.join(os.path.dirname(__file__), "presets",
                autonet.preset_folder_name, config_preset + ".txt"))
            c.update(self.base_config)
            self.base_config = c

    # OVERRIDE
    def fit(self, X_train, Y_train, X_valid=None, Y_valid=None, refit=True, **autonet_config):
        X_train, Y_train, X_valid, Y_valid = self.check_data_array_types(X_train, Y_train, X_valid, Y_valid)
        self.autonet_config = self.pipeline.get_pipeline_config(**dict(self.base_config, **autonet_config))

        self.autonet_config["save_models"] = True

        self.fit_result = self.pipeline.fit_pipeline(pipeline_config=self.autonet_config,
                                                     X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid)
        self.dataset_info = self.pipeline[CreateDatasetInfo.get_name()].fit_output["dataset_info"]
        self.pipeline.clean()
        if refit:
            self.refit(X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid)
        return self.fit_result
    
    # OVERRIDE
    def refit(self, X_train, Y_train, X_valid=None, Y_valid=None, ensemble_configs=None, ensemble=None, autonet_config=None):
        X_train, Y_train, X_valid, Y_valid = self.check_data_array_types(X_train, Y_train, X_valid, Y_valid)
        # The ensemble API does mot contain the fit_output from cross_val subpipeline nodes. Fit a single pipeline here for preprocessing
        if (autonet_config is None):
            autonet_config = self.autonet_config
        if (autonet_config is None):
            autonet_config = self.base_config
        if (ensemble_configs is None and self.fit_result and "ensemble_configs" in self.fit_result.keys()):
            ensemble_configs = self.fit_result["ensemble_configs"]
        if (ensemble is None and self.fit_result):
            ensemble = self.fit_result["ensemble"]
        if (autonet_config is None or ensemble_configs is None or ensemble is None):
            raise ValueError("You have to specify ensemble and autonet config in order to be able to refit")
 
        identifiers = ensemble.get_selected_model_identifiers()
        self.trained_autonets = dict()

        autonet_config["save_models"] = False

        """
        for config_id, hyperparameter_config in ensemble_configs:
            if "model" in hyperparameter_config.keys() and hyperparameter_config["model"]=="baseline":
                print("cont.....")
                continue
            else:
                autonet = self.autonet_type(pipeline=self.pipeline)
                autonet.refit(X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid,
                    hyperparameter_config=hyperparameter_config, autonet_config=autonet_config, budget=budget)
                self.trained_autonets[tuple(identifier)] = autonet
                self.trained_autonet = autonet
                break
        """

        configspace = self.get_hyperparameter_search_space()

        hyperparameter_config = configspace.sample_configuration().get_dictionary()

        autonet = self.autonet_type(pipeline=self.pipeline)
        autonet.refit(X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid,
                hyperparameter_config=hyperparameter_config, autonet_config=autonet_config, budget=1)
        self.trained_autonets[(0,0,0,0)] = autonet
        self.trained_autonet = autonet
    
    # OVERRIDE
    def predict(self, X, return_probabilities=False, return_metric=False):
        # run predict pipeline
        X, = self.check_data_array_types(X)
        prediction = None
        autonet_config = self.get_current_autonet_config()

        identifiers_with_budget, weights = self.fit_result["ensemble"].identifiers_, self.fit_result["ensemble"].weights_

        baseline_id2model = BaselineTrainer.identifiers_ens


        model_dirs = [os.path.join(self.autonet_config["result_logger_dir"], "models", str(ident) + ".torch") for ident in identifiers_with_budget]
        
        # get data preprocessing pipeline
        for ident, weight in zip(identifiers_with_budget, weights):
            
            if weight==0:
                continue


            if ident[0]>=0:
                model_dir = os.path.join(self.autonet_config["result_logger_dir"], "models", str(ident) + ".torch")
                logging.info("==> Inferring model model " + model_dir + ", adding preds with weight " + str(weight))
                model = torch.load(model_dir)

                autonet_config["model"] = model
                current_prediction = self.trained_autonet.pipeline.predict_pipeline(pipeline_config=autonet_config, X=X)['Y']
                prediction = current_prediction if prediction is None else prediction + weight * current_prediction

                OHE = self.trained_autonet.pipeline[OneHotEncoding.get_name()]
                metric = self.trained_autonet.pipeline[MetricSelector.get_name()].fit_output['optimize_metric']

            else:
                model_dir = os.path.join(self.autonet_config["result_logger_dir"], "models", str(ident) + ".pkl")
                info_dir =  os.path.join(self.autonet_config["result_logger_dir"], "models", str(ident) + "_info.pkl")

                logging.info("==> Inferring model model " + model_dir + ", adding preds with weight " + str(weight))

                baseline_model = baseline_id2model[ident[0]]()
                baseline_model.load(model_dir, info_dir)

                current_prediction = baseline_model.predict(X_test=X, predict_proba=True)
                prediction = current_prediction if prediction is None else prediction + weight * current_prediction
                
        # reverse one hot encoding
        result = OHE.reverse_transform_y(prediction, OHE.fit_output['y_one_hot_encoder'])
        if not return_probabilities and not return_metric:
            return result
        result = [result]
        if return_probabilities:
            result.append(prediction)
        if return_metric:
            result.append(metric)
        return tuple(result)


        """
        models_with_weights = self.fit_result["ensemble"].get_models_with_weights(self.trained_autonets)
        autonet_config = self.autonet_config or self.base_config
        for weight, autonet in models_with_weights:
            current_prediction = autonet.pipeline.predict_pipeline(pipeline_config=autonet_config, X=X)["Y"]
            prediction = current_prediction if prediction is None else prediction + weight * current_prediction
            OHE = autonet.pipeline[OneHotEncoding.get_name()]
            metric = autonet.pipeline[MetricSelector.get_name()].fit_output['optimize_metric']

        # reverse one hot encoding 
        result = OHE.reverse_transform_y(prediction, OHE.fit_output['y_one_hot_encoder'])
        if not return_probabilities and not return_metric:
            return result
        result = [result]
        if return_probabilities:
            result.append(prediction)
        if return_metric:
            result.append(metric)
        return tuple(result)
        """
    
    # OVERRIDE
    def score(self, X_test, Y_test):
        # run predict pipeline
        X_test, Y_test = self.check_data_array_types(X_test, Y_test)
        _, Y_pred, metric = self.predict(X_test, return_probabilities=True, return_metric=True)
        Y_test, _ = self.pipeline[OneHotEncoding.get_name()].complete_y_tranformation(Y_test)
        return metric(Y_pred, Y_test)
