__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import os

from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption
from autoPyTorch.pipeline.nodes.metric_selector import MetricSelector, AutoNetMetric, no_transform
from autoPyTorch.pipeline.nodes import OneHotEncoding, OptimizationAlgorithm
from autoPyTorch.pipeline.nodes.metric_selector import AutoNetMetric
from autoPyTorch.utils.ensemble import build_ensemble, read_ensemble_prediction_file, combine_predictions, combine_test_predictions, \
    ensemble_logger, start_server
from hpbandster.core.result import logged_results_to_HBS_result
import json
import asyncio
from hpbandster.core.nameserver import nic_name_to_host
import time
import logging


def predictions_for_ensemble(y_true, y_pred):
    return y_pred

class EnableComputePredictionsForEnsemble(PipelineNode):
    """Put this Node in the training pipeline after the metric selector node"""
    def fit(self, pipeline_config, additional_metrics, refit, loss_penalty):
        if refit or pipeline_config["ensemble_size"] == 0 or loss_penalty > 0:
            return dict()
        return {'additional_metrics': additional_metrics + [
            AutoNetMetric(name="predictions_for_ensemble",
                          metric=predictions_for_ensemble,
                          loss_transform=no_transform,
                          ohe_transform=no_transform)]}


class SavePredictionsForEnsemble(PipelineNode):
    """Put this Node in the training pipeline after the training node"""
    def fit(self, pipeline_config, loss, info, refit, loss_penalty, baseline_predictions_for_ensemble=None, baseline_id=None):
        if refit or pipeline_config["ensemble_size"] == 0 or loss_penalty > 0:
            return {"loss": loss, "info": info}

        if "val_predictions_for_ensemble" in info:
            predictions = info["val_predictions_for_ensemble"]
            del info["val_predictions_for_ensemble"]
        else:
            raise ValueError("You need to specify some kind of validation for ensemble building")
        del info["train_predictions_for_ensemble"]

        combinator = {
            "combinator": combine_predictions,
            "data": predictions
        }

        # has to be int or float to be passed to logger
        info["baseline_id"] = baseline_id[0] if baseline_id is not None else None

        if baseline_predictions_for_ensemble is not None:
            baseline_predictions = baseline_predictions_for_ensemble

            baseline_combinator = {
                    "combinator": combine_predictions,
                    "data": baseline_predictions
                    }
        else:
            baseline_combinator = None

        if not "test_predictions_for_ensemble" in info:
            if baseline_combinator is not None:
                return {"loss": loss, "info": info, "predictions_for_ensemble": combinator, "baseline_predictions_for_ensemble": baseline_combinator}
            else:
                return {"loss": loss, "info": info, "predictions_for_ensemble": combinator}
        
        test_combinator = {
            "combinator": combine_test_predictions,
            "data": info["test_predictions_for_ensemble"]
        }
        del info["test_predictions_for_ensemble"]

        if baseline_combinator is not None:
            return {"loss": loss, "info": info, "predictions_for_ensemble": combinator, "test_predictions_for_ensemble": test_combinator, "baseline_predictions_for_ensemble": baseline_combinator}
        return {"loss": loss, "info": info, "predictions_for_ensemble": combinator, "test_predictions_for_ensemble": test_combinator}

    def predict(self, Y):
        return {"Y": Y}
    
    def get_pipeline_config_options(self):
        options = [
            ConfigOption("ensemble_server_credentials", default=None)
        ]
        return options


class BuildEnsemble(PipelineNode):
    """Put this node after the optimization algorithm node"""
    def fit(self, pipeline_config, optimized_hyperparameter_config, budget, loss, info, refit=None):
        if refit or pipeline_config["ensemble_size"] == 0 or pipeline_config["task_id"] not in [-1, 1]:
            return {"optimized_hyperparameter_config": optimized_hyperparameter_config, "budget": budget}
        
        filename = os.path.join(pipeline_config["result_logger_dir"], 'predictions_for_ensemble.npy')
        optimize_metric = self.pipeline[MetricSelector.get_name()].metrics[pipeline_config["optimize_metric"]]
        y_transform = self.pipeline[OneHotEncoding.get_name()].complete_y_tranformation
        result = logged_results_to_HBS_result(pipeline_config["result_logger_dir"])

        all_predictions, labels, model_identifiers, _ = read_ensemble_prediction_file(filename=filename, y_transform=y_transform)
        ensemble_selection, ensemble_configs = build_ensemble(result=result,
            optimize_metric=optimize_metric, ensemble_size=pipeline_config["ensemble_size"],
            all_predictions=all_predictions, labels=labels, model_identifiers=model_identifiers,
            only_consider_n_best=pipeline_config["ensemble_only_consider_n_best"],
            sorted_initialization_n_best=pipeline_config["ensemble_sorted_initialization_n_best"])

        return {"optimized_hyperparameter_config": optimized_hyperparameter_config, "budget": budget,
            "ensemble": ensemble_selection,
            "ensemble_configs": ensemble_configs,
            "loss": loss,
            "info": info
            }
    
    def predict(self, Y):
        return {"Y": Y}
    
    def get_pipeline_config_options(self):
        options = [
            ConfigOption("ensemble_size", default=50, type=int, info="Build a ensemble of well performing autonet configurations. 0 to disable."),
            ConfigOption("ensemble_only_consider_n_best", default=30, type=int, info="Only consider the n best models for ensemble building."),
            ConfigOption("ensemble_sorted_initialization_n_best", default=0, type=int, info="Initialize ensemble with n best models.")
        ]
        return options

class EnsembleServer(PipelineNode):
    """Put this node in front of the optimization algorithm node"""

    def fit(self, pipeline_config, result_loggers, shutdownables, refit=False):
        if refit or pipeline_config["ensemble_size"] == 0:
            return dict()
        es_credentials_file = os.path.join(pipeline_config["working_dir"], "es_credentials_%s.json" % pipeline_config["run_id"])

        # start server
        if pipeline_config["task_id"] != 1 or pipeline_config["run_worker_on_master_node"]:
            host = nic_name_to_host(OptimizationAlgorithm.get_nic_name(pipeline_config))
            host, port, process = start_server(host)
            pipeline_config["ensemble_server_credentials"] = (host, port)
            shutdownables = shutdownables + [process]

        result_loggers = [ensemble_logger(directory=pipeline_config["result_logger_dir"], overwrite=True)] + result_loggers
        return {"result_loggers": result_loggers, "shutdownables": shutdownables}
