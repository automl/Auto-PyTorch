from autoPyTorch.utils.config.config_option import ConfigOption
from autoPyTorch.utils.benchmarking.benchmark_pipeline import ForRun, ForAutoNetConfig, ForInstance
from autoPyTorch.utils.benchmarking.benchmark_pipeline.prepare_result_folder import get_run_result_dir
import os
import logging
import traceback

class CollectInstanceTrajectories(ForInstance):
    def fit(self, pipeline_config, run_id_range):
        instances = self.get_instances(pipeline_config, instance_slice=self.parse_slice(pipeline_config["instance_slice"]))

        result_trajectories = dict()
        result_optimize_metrics = set()

        for instance in instances:
            try:
                pipeline_result = self.sub_pipeline.fit_pipeline(pipeline_config=pipeline_config, instance=instance, run_id_range=run_id_range)

                # merge the trajectories into one dict
                instance_trajectories = pipeline_result["trajectories"]
                optimize_metrics = pipeline_result["optimize_metrics"]

                for metric, config_trajectories in instance_trajectories.items():
                    if metric not in result_trajectories:
                        result_trajectories[metric] = dict()
                    for config, run_trajectories in config_trajectories.items():
                        if config not in result_trajectories[metric]:
                            result_trajectories[metric][config] = dict()
                        result_trajectories[metric][config][instance] = run_trajectories
                result_optimize_metrics |= optimize_metrics

            except Exception as e:
                print(e)
                traceback.print_exc()
        return {"trajectories": result_trajectories,
                "optimize_metrics": result_optimize_metrics}


class CollectAutoNetConfigTrajectories(ForAutoNetConfig):
    def fit(self, pipeline_config, instance, run_id_range):
        logging.getLogger('benchmark').info('Collecting data for dataset ' + instance)

        result_trajectories = dict()
        result_optimize_metrics = set()

        # iterate over all configs
        for config_file in self.get_config_files(pipeline_config):
            autonet_config_name = os.path.basename(config_file).split(".")[0]
            pipeline_result = self.sub_pipeline.fit_pipeline(pipeline_config=pipeline_config,
                                                             instance=instance,
                                                             run_id_range=run_id_range,
                                                             autonet_config_file=config_file)
            
            # merge the trajectories into one dict
            config_trajectories = pipeline_result["trajectories"]
            optimize_metrics = pipeline_result["optimize_metrics"]

            for metric, run_trajectories in config_trajectories.items():
                if metric not in result_trajectories:
                    result_trajectories[metric] = dict()
                result_trajectories[metric][autonet_config_name] = run_trajectories

            result_optimize_metrics |= optimize_metrics
        return {"trajectories": result_trajectories,
                "optimize_metrics": result_optimize_metrics}


class CollectRunTrajectories(ForRun):
    def fit(self, pipeline_config, instance, run_id_range, autonet_config_file):
        logging.getLogger('benchmark').info('Collecting data for autonet config ' + autonet_config_file)

        result_trajectories = dict()
        optimize_metrics = set()

        run_number_range = self.parse_range(pipeline_config['run_number_range'], pipeline_config['num_runs'])
        instance_result_dir = os.path.abspath(os.path.join(get_run_result_dir(pipeline_config, instance, autonet_config_file, "0", "0"), ".."))
        if not os.path.exists(instance_result_dir):
            logging.getLogger('benchmark').warn("Skipping %s because it no results exist" % instance_result_dir)
            return {"trajectories": result_trajectories, "optimize_metrics": optimize_metrics}
        run_result_dirs = next(os.walk(instance_result_dir))[1]

        # iterate over all run_numbers and run_ids
        for run_result_dir in run_result_dirs:
            run_id, run_number = parse_run_folder_name(run_result_dir)
            run_result_dir = get_run_result_dir(pipeline_config, instance, autonet_config_file, run_id, run_number)
            if (run_id_range is not None and run_id not in run_id_range) or run_number not in run_number_range:
                continue

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
            optimize_metric = pipeline_result["optimize_metric"]

            # merge the trajectories into one dict
            for metric, trajectory in run_trajectories.items():
                if metric not in result_trajectories:
                    result_trajectories[metric] = list()
                result_trajectories[metric].append(trajectory)

            if optimize_metric is not None:
                optimize_metrics |= set([optimize_metric])
        return {"trajectories": result_trajectories, "optimize_metrics": optimize_metrics}

def parse_run_folder_name(run_folder_name):
    assert run_folder_name.startswith("run_")
    run_folder_name = run_folder_name[4:].split("_")
    run_id = int(run_folder_name[0])
    run_number = int(run_folder_name[1])
    return run_id, run_number