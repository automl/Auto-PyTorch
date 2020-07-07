import torch
import os as os
import json
import pickle
import numpy as np
import scipy.sparse
import logging
import collections

import ConfigSpace
import ConfigSpace.hyperparameters as CSH
        
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

from autoPyTorch.utils.config.config_option import ConfigOption
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.utils.ensemble import read_ensemble_prediction_file, combine_predictions, combine_test_predictions, ensemble_logger
    
from autoPyTorch.components.baselines import baselines

def get_dimensions(a):
    if isinstance(a, list):
        return len(np.array(a).shape)
    return len(a.shape)

class BaselineTrainer(PipelineNode):

    #TODO: Order
    models = collections.OrderedDict({
            "random_forest" : baselines.RFBaseline, 
            "extra_trees" : baselines.ExtraTreesBaseline,
            "lgb" : baselines.LGBBaseline,
            "catboost" : baselines.CatboostBaseline,
            "rotation_forest" : baselines.RotationForestBaseline,
            "knn" : baselines.KNNBaseline})

    identifiers = {
            "random_forest": (-6, 0, 0, 0.0),
            "extra_trees": (-5, 0, 0, 0.0),
            "lgb": (-4, 0, 0, 0.0),
            "catboost": (-3, 0, 0, 0.0),
            "rotation_forest": (-2, 0, 0, 0.0),
            "knn": (-1, 0, 0, 0.0)}

    identifiers_ens = {
            -6: baselines.RFBaseline,
            -5: baselines.ExtraTreesBaseline,
            -4: baselines.LGBBaseline,
            -3: baselines.CatboostBaseline,
            -2: baselines.RotationForestBaseline,
            -1: baselines.KNNBaseline}

    def __init__(self):
        super(BaselineTrainer, self).__init__()

        self.X_test = None

    def add_test_data(self, X_test):
        self.X_test = X_test

    def fit(self, pipeline_config, X, Y, train_indices, valid_indices, refit):

        baseline_name = self.get_baseline_to_train(pipeline_config)

        if baseline_name is not None:
            baseline_model = BaselineTrainer.models[baseline_name]()
        else:
            return {"baseline_id": None, "baseline_predictions_for_ensemble": None}

        # Fit
        fit_output = baseline_model.fit(X[train_indices], Y[train_indices], X[valid_indices], Y[valid_indices])
        baseline_preds = np.array(fit_output["val_preds"])

        # Test data
        if self.X_test is not None:
            test_preds = baseline_model.predict(X_test=self.X_test, predict_proba=True)
            test_preds = np.array(test_preds)
        else:
            test_preds = None

        # Save model
        identifier = BaselineTrainer.identifiers[baseline_name]
        model_savedir = os.path.join(pipeline_config["result_logger_dir"], "models", str(identifier) + ".pkl")
        info_savedir = os.path.join(pipeline_config["result_logger_dir"], "models", str(identifier) + "_info.pkl")
        os.makedirs(os.path.dirname(model_savedir), exist_ok=True)
        
        baseline_model.save(model_path=model_savedir, info_path=info_savedir)

        
        return {"baseline_id": identifier, "baseline_predictions_for_ensemble": baseline_preds, "baseline_test_predictions_for_ensemble": test_preds}

    def get_baseline_to_train(self, pipeline_config):
        trained_baseline_logdir = os.path.join(pipeline_config["result_logger_dir"], "trained_baselines.txt")

        baselines = pipeline_config["baseline_models"]

        trained_baselines = []
        if os.path.isfile(trained_baseline_logdir):
            with open(trained_baseline_logdir, "r") as f:
                for l in f:
                    trained_baselines.append(l.replace("\n",""))

        for baseline in baselines:
            if baseline not in trained_baselines:
                with open(trained_baseline_logdir, "a+") as f:
                    f.write(baseline+"\n")
                return baseline
        return None


    def predict(self, X):
        return { 'X': X }

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name='baseline_models', default=list(BaselineTrainer.models.keys()), type=str, list=True, choices=list(BaselineTrainer.models.keys()))
        ]
        return options
