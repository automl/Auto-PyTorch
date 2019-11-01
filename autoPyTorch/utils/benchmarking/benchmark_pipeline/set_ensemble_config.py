from hpbandster.core.result import logged_results_to_HBS_result
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.utils.benchmarking.benchmark_pipeline.prepare_result_folder import get_run_result_dir
from copy import copy
import os
import logging

class SetEnsembleConfig(PipelineNode):

    def fit(self, pipeline_config, autonet, run_result_dir):
        parser = autonet.get_autonet_config_file_parser()
        autonet_config = parser.read(os.path.join(run_result_dir, "autonet.config"))
        
        if pipeline_config["ensemble_size"]:
            autonet_config["ensemble_size"] = pipeline_config["ensemble_size"]
        
        if pipeline_config["ensemble_only_consider_n_best"]:
            autonet_config["ensemble_only_consider_n_best"] = pipeline_config["ensemble_only_consider_n_best"]

        if pipeline_config["ensemble_sorted_initialization_n_best"]:
            autonet_config["ensemble_sorted_initialization_n_best"] = pipeline_config["ensemble_sorted_initialization_n_best"]
        
        autonet.autonet_config = autonet_config

        return {"result_dir": run_result_dir,
                "optimize_metric": autonet_config["optimize_metric"],
                "trajectories": []}
    
    def get_pipeline_config_options(self):
        options = [
            ConfigOption('ensemble_size', default=0, type=int),
            ConfigOption('ensemble_only_consider_n_best', default=0, type=int),
            ConfigOption('ensemble_sorted_initialization_n_best', default=0, type=int)
        ]
        return options