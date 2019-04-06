import os
from autoPyTorch.core.api import AutoNet
from autoPyTorch.pipeline.base.pipeline import Pipeline
from autoPyTorch.pipeline.nodes.one_hot_encoding import OneHotEncoding
from autoPyTorch.pipeline.nodes.metric_selector import MetricSelector
from autoPyTorch.pipeline.nodes.ensemble import EnableComputePredictionsForEnsemble, SavePredictionsForEnsemble, BuildEnsemble, EnsembleServer

class AutoNetEnsemble(AutoNet):
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

        if config_preset is not None:
            parser = self.get_autonet_config_file_parser()
            c = parser.read(os.path.join(os.path.dirname(__file__), "presets",
                autonet.preset_folder_name, config_preset + ".txt"))
            c.update(self.base_config)
            self.base_config = c

    def fit(self, X_train, Y_train, X_valid=None, Y_valid=None, refit=True, **autonet_config):
        self.autonet_config = self.pipeline.get_pipeline_config(**dict(self.base_config, **autonet_config))
        self.fit_result = self.pipeline.fit_pipeline(pipeline_config=self.autonet_config,
                                                     X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid)
        self.pipeline.clean()
        if refit:
            self.refit(X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid)
        return self.fit_result["ensemble_configs"], self.fit_result["ensemble_final_metric_score"], self.fit_result["ensemble"]
    
    def refit(self, X_train, Y_train, X_valid=None, Y_valid=None, ensemble_configs=None, ensemble=None, autonet_config=None):
        if (autonet_config is None):
            autonet_config = self.autonet_config
        if (autonet_config is None):
            autonet_config = self.base_config
        if (ensemble_configs is None and self.fit_result):
            ensemble_configs = self.fit_result["ensemble_configs"]
        if (ensemble is None and self.fit_result):
            ensemble = self.fit_result["ensemble"]
        if (autonet_config is None or ensemble_configs is None or ensemble is None):
            raise ValueError("You have to specify ensemble and autonet config in order to be able to refit")
        
        identifiers = ensemble.get_selected_model_identifiers()
        self.trained_autonets = dict()
        for identifier in identifiers:
            config_id = tuple(identifier[:3])
            budget = identifier[3]
            hyperparameter_config = ensemble_configs[config_id]
            autonet = self.autonet_type(pipeline=self.pipeline.clone())
            autonet.refit(X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid,
                hyperparameter_config=hyperparameter_config, autonet_config=autonet_config, budget=budget)
            self.trained_autonets[tuple(identifier)] = autonet
    
    def predict(self, X, return_probabilities=False, return_metric=False):
        # run predict pipeline
        prediction = None
        models_with_weights = self.fit_result["ensemble"].get_models_with_weights(self.trained_autonets)
        autonet_config = self.autonet_config or self.base_config
        for weight, autonet in models_with_weights:
            current_prediction = autonet.pipeline.predict_pipeline(pipeline_config=autonet_config, X=X)["Y"]
            prediction = current_prediction if prediction is None else prediction + weight * current_prediction
            OHE = autonet.pipeline[OneHotEncoding.get_name()]
            metric = autonet.pipeline[MetricSelector.get_name()].fit_output['train_metric']

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
    
    def score(self, X_test, Y_test):
        # run predict pipeline
        _, Y_pred, metric = self.predict(X_test, return_probabilities=True, return_metric=True)
        Y_test, _ = self.pipeline[OneHotEncoding.get_name()].complete_y_tranformation(Y_test)
        return metric(Y_pred, Y_test)
