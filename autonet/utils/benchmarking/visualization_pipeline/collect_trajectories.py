from autonet.utils.config.config_option import ConfigOption
from autonet.utils.benchmarking.benchmark_pipeline.for_run import ForRun
from autonet.utils.benchmarking.benchmark_pipeline.for_autonet_config import ForAutoNetConfig
from autonet.utils.benchmarking.benchmark_pipeline.prepare_result_folder import get_run_result_dir
import os
import logging

class CollectAutoNetConfigTrajectories(ForAutoNetConfig):
    def fit(self, pipeline_config, instance, run_id_range):
        logging.getLogger('benchmark').info('Collecting data for dataset ' + instance)

        result_trajectories = dict()
        result_train_metrics = set()

        # iterate over all configs
        for config_file in self.get_config_files(pipeline_config):
            autonet_config_name = os.path.basename(config_file).split(".")[0]
            pipeline_result = self.sub_pipeline.fit_pipeline(pipeline_config=pipeline_config,
                                                             instance=instance,
                                                             run_id_range=run_id_range,
                                                             autonet_config_file=config_file)
            
            # merge the trajectories into one dict
            config_trajectories = pipeline_result["trajectories"]
            train_metrics = pipeline_result["train_metrics"]

            for metric, run_trajectories in config_trajectories.items():
                if metric not in result_trajectories:
                    result_trajectories[metric] = dict()
                result_trajectories[metric][autonet_config_name] = run_trajectories

            result_train_metrics |= train_metrics
        return {"trajectories": result_trajectories,
                "train_metrics": result_train_metrics}


class CollectRunTrajectories(ForRun):
    def fit(self, pipeline_config, instance, run_id_range, autonet_config_file):
        logging.getLogger('benchmark').info('Collecting data for autonet config ' + autonet_config_file)

        result_trajectories = dict()
        train_metrics = set()

        # iterate over all run_numbers and run_ids
        for run_number in self.parse_range(pipeline_config['run_number_range'], pipeline_config['num_runs']):
            for run_id in run_id_range:
                run_result_dir = get_run_result_dir(pipeline_config, instance, autonet_config_file, run_id, run_number)
                if not os.path.exists(run_result_dir):
                    logging.getLogger('benchmark').debug("Skipping " + run_result_dir + "because it does not exist")
                    continue
                pipeline_result = self.sub_pipeline.fit_pipeline(pipeline_config=pipeline_config,
                                                                 instance=instance,
                                                                 run_number=run_number,
                                                                 run_id=run_id,
                                                                 autonet_config_file=autonet_config_file,
                                                                 run_result_dir=run_result_dir)
                run_trajectories = pipeline_result["trajectories"]
                train_metric = pipeline_result["train_metric"]

                # merge the trajectories into one dict
                for metric, trajectory in run_trajectories.items():
                    if metric not in result_trajectories:
                        result_trajectories[metric] = list()
                    result_trajectories[metric].append(trajectory)

                if train_metric is not None:
                    train_metrics |= set([train_metric])
        return {"trajectories": result_trajectories, "train_metrics": train_metrics}
