from hpbandster.core.result import logged_results_to_HBS_result
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool, to_list
from autoPyTorch.utils.benchmarking.benchmark_pipeline.prepare_result_folder import get_run_result_dir
from autoPyTorch.pipeline.nodes.metric_selector import MetricSelector
from autoPyTorch.pipeline.nodes import OneHotEncoding
from autoPyTorch.pipeline.nodes.ensemble import read_ensemble_prediction_file
from hpbandster.core.result import logged_results_to_HBS_result
from copy import copy
import os
import logging
import math
import numpy as np
import json

class GetEnsembleTrajectories(PipelineNode):

    def fit(self, pipeline_config, run_result_dir, train_metric, trajectories):
        ensemble_log_file = os.path.join(run_result_dir, "ensemble_log.json")
        test_log_file = os.path.join(run_result_dir, "test_result.json")
        if not pipeline_config["enable_ensemble"] or train_metric is None or \
            (not os.path.exists(ensemble_log_file) and not os.path.exists(test_log_file)):
            return {"trajectories": trajectories, "train_metric": train_metric}

        try:
            started = logged_results_to_HBS_result(run_result_dir).HB_config["time_ref"]
        except:
            return {"trajectories": trajectories, "train_metric": train_metric}
        
        ensemble_trajectories = dict()
        test_trajectories = dict()
        if os.path.exists(ensemble_log_file):
            ensemble_trajectories = get_ensemble_trajectories(ensemble_log_file, started)
        if os.path.exists(test_log_file):
            test_trajectories = get_ensemble_trajectories(test_log_file, started, prefix="", only_test=True)
        
        return {"trajectories": dict(trajectories, **ensemble_trajectories, **test_trajectories), "train_metric": train_metric}
    
    def get_pipeline_config_options(self):
        options = [
            ConfigOption('enable_ensemble', default=False, type=to_bool)
        ]
        return options

def get_ensemble_trajectories(ensemble_log_file, started, prefix="ensemble_", only_test=False):
    ensemble_trajectories = dict()
    with open(ensemble_log_file) as f:
        for line in f:
            finished, metric_values, _, _, _ = json.loads(line)
            finished = finished["finished"] if isinstance(finished, dict) else finished

            for metric_name, metric_value in metric_values.items():
                if only_test and not metric_name.startswith("test"):
                    continue
                trajectory_name = prefix + metric_name
    
                # save in trajectory
                if trajectory_name not in ensemble_trajectories:
                    ensemble_trajectories[trajectory_name] = {"times_finished": [], "losses": [], "flipped": False}
                ensemble_trajectories[trajectory_name]["times_finished"].append(finished - started)
                ensemble_trajectories[trajectory_name]["losses"].append(metric_value)
    return ensemble_trajectories