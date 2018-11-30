from hpbandster.core.result import logged_results_to_HBS_result
from autonet.pipeline.base.pipeline_node import PipelineNode
from autonet.utils.config.config_option import ConfigOption, to_bool
from autonet.utils.benchmarking.benchmark_pipeline.prepare_result_folder import get_run_result_dir
from copy import copy
import os
import logging

class GetRunTrajectories(PipelineNode):

    def fit(self, pipeline_config, autonet, run_result_dir):
        parser = autonet.get_autonet_config_file_parser()
        autonet_config = parser.read(os.path.join(run_result_dir, "autonet.config"))

        if pipeline_config["only_finished_runs"] and not os.path.exists(os.path.join(run_result_dir, "summary.json")):
            logging.getLogger('benchmark').info('Skipping ' + run_result_dir + ' because the run is not finished yet')
            return {"trajectories": dict(), "train_metric": None}

        return {"trajectories": build_run_trajectories(run_result_dir, autonet_config),
                "train_metric": autonet_config["train_metric"]}
    
    def get_pipeline_config_options(self):
        options = [
            ConfigOption('only_finished_runs', default=True, type=to_bool),
            ConfigOption('result_dir', default=None, type='directory', required=True),
        ]
        return options


def build_run_trajectories(results_folder, autonet_config):
    # parse results
    res = logged_results_to_HBS_result(results_folder)
    incumbent_trajectory = res.get_incumbent_trajectory(bigger_is_better=False, non_decreasing_budget=False)

    # prepare
    metric_name = autonet_config["train_metric"]
    all_metrics = autonet_config["additional_metrics"] + [metric_name]
    additional_metric_names = ["val_" + m for m in all_metrics]
    additional_metric_names += ["train_" + m for m in all_metrics]
    additional_metric_names += autonet_config["additional_logs"]

    # initialize incumbent trajectories
    incumbent_trajectories = dict()
    
    # save incumbent trajectories
    incumbent_trajectories[metric_name] = incumbent_trajectory
    incumbent_trajectory["flipped"] = not autonet_config["minimize"]
    for name in additional_metric_names:
        tj = copy(incumbent_trajectory)
        tj["losses"] = [run["info"][name] for config_id, budget in zip(tj["config_ids"], tj["budgets"])
                                          for run in res.get_runs_by_id(config_id)
                                          if run["budget"] == budget and name in run["info"]]
        tj["flipped"] = False
        if tj["losses"]:
            incumbent_trajectories[name] = tj
            
    return incumbent_trajectories