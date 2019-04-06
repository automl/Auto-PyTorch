import os
import logging
from ConfigSpace.read_and_write import json
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates

class PrepareResultFolder(PipelineNode):

    def fit(self, pipeline_config, data_manager, instance, 
            autonet_config_file, autonet, run_number, run_id, task_id):

        instance_name, autonet_config_name, run_name = get_names(instance, autonet_config_file, run_id, run_number)
        run_result_dir = get_run_result_dir(pipeline_config, instance, autonet_config_file, run_id, run_number)
        instance_run_id = str(run_name) + "_" + str(instance_name) + "_" + str(autonet_config_name)
        instance_run_id = '_'.join(instance_run_id.split(':'))
        
        autonet.autonet_config = None #clean results of last fit
        autonet.update_autonet_config(task_id=task_id, run_id=instance_run_id,  result_logger_dir=run_result_dir)

        if (task_id not in [-1, 1]):
            return { 'result_dir': run_result_dir }

        if not os.path.exists(run_result_dir):
            try:
                os.makedirs(run_result_dir)
            except Exception as e:
                print(e)


        logging.getLogger('benchmark').debug("Create config and info files for current run " + str(run_name))

        instance_info = dict()
        instance_info['path'] = instance
        instance_info['is_classification'] = data_manager.is_classification
        instance_info['is_multilabel'] = data_manager.is_multilabel
        instance_info['instance_shape'] = data_manager.X_train.shape
        instance_info['categorical_features'] = data_manager.categorical_features

        if autonet.get_current_autonet_config()["hyperparameter_search_space_updates"] is not None:
            autonet.get_current_autonet_config()["hyperparameter_search_space_updates"].save_as_file(
                os.path.join(run_result_dir, "hyperparameter_search_space_updates.txt"))

        self.write_config_to_file(run_result_dir, "instance.info", instance_info)
        self.write_config_to_file(run_result_dir, "benchmark.config", pipeline_config)
        self.write_config_to_file(run_result_dir, "autonet.config", autonet.get_current_autonet_config())

        with open(os.path.join(run_result_dir, "configspace.json"), "w") as f:
            f.write(json.write(autonet.pipeline.get_hyperparameter_search_space(**autonet.get_current_autonet_config())))

        return { 'result_dir': run_result_dir }
        

    def write_config_to_file(self, folder, filename, config):
        do_not_write = ["hyperparameter_search_space_updates"]
        with open(os.path.join(folder, filename), "w") as f:
            f.write("\n".join([(key + '=' + str(value)) for key, value in config.items() if not key in do_not_write]))

    def get_pipeline_config_options(self):
        options = [
            ConfigOption('result_dir', default=None, type='directory', required=True)
        ]
        return options

def get_names(instance, autonet_config_file, run_id, run_number):
    if isinstance(instance, list):
        instance_name = "_".join([os.path.split(p)[1].split(".")[0] for p in instance])
    else:
        instance_name = os.path.basename(instance).split(".")[0]

    autonet_config_name = os.path.basename(autonet_config_file).split(".")[0]
    run_name = "run_" + str(run_id) + "_" + str(run_number)

    return "_".join(instance_name.split(':')), autonet_config_name, run_name

def get_run_result_dir(pipeline_config, instance, autonet_config_file, run_id, run_number):
    instance_name, autonet_config_name, run_name = get_names(instance, autonet_config_file, run_id, run_number)
    run_result_dir = os.path.join(pipeline_config['result_dir'],
                                  pipeline_config["benchmark_name"],
                                  instance_name,
                                  autonet_config_name,
                                  run_name)
    return run_result_dir