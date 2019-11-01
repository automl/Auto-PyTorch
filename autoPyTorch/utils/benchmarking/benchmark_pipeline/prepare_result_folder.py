import os
import logging
from ConfigSpace.read_and_write import json as cs_json, pcs_new as cs_pcs
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from autoPyTorch.utils.modify_config_space import remove_constant_hyperparameter

class PrepareResultFolder(PipelineNode):

    def fit(self, pipeline_config, data_manager, instance,
            autonet, run_number, run_id, task_id):

        instance_name, run_name = get_names(instance, run_id, run_number)
        run_result_dir = get_run_result_dir(pipeline_config, instance, run_id, run_number, autonet)
        instance_run_id = str(run_name) + "-" + str(instance_name)
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

        autonet_config = autonet.get_current_autonet_config()
        if autonet_config["hyperparameter_search_space_updates"] is not None:
            autonet_config["hyperparameter_search_space_updates"].save_as_file(
                os.path.join(run_result_dir, "hyperparameter_search_space_updates.txt"))

        if 'user_updates_config' in pipeline_config:
            user_updates_config = pipeline_config['user_updates_config']
            if user_updates_config:
                from shutil import copyfile
                copyfile(user_updates_config, os.path.join(run_result_dir, 'user_updates_config.csv'))

        self.write_config_to_file(run_result_dir, "instance.info", instance_info)
        self.write_config_to_file(run_result_dir, "benchmark.config", pipeline_config)
        self.write_config_to_file(run_result_dir, "autonet.config", autonet_config)

        # save refit config - add indent and sort keys
        if 'refit_config' in pipeline_config and pipeline_config['refit_config'] is not None:
            import json
            with open(pipeline_config['refit_config'], 'r') as f:
                refit_config = json.loads(f.read())
            with open(os.path.join(run_result_dir, 'refit_config.json'), 'w+') as f:
                f.write(json.dumps(refit_config, indent=4, sort_keys=True))

        # save search space
        search_space = autonet.pipeline.get_hyperparameter_search_space(**autonet_config)
        with open(os.path.join(run_result_dir, "configspace.json"), "w") as f:
            f.write(cs_json.write(search_space))

        # save search space without constants - used by bohb - as pcs (simple)
        simplified_search_space, _ = remove_constant_hyperparameter(search_space)
        with open(os.path.join(run_result_dir, "configspace_simple.pcs"), "w") as f:
            f.write(cs_pcs.write(simplified_search_space))

        return { 'result_dir': run_result_dir }
        

    def write_config_to_file(self, folder, filename, config):
        do_not_write = ["hyperparameter_search_space_updates"]
        with open(os.path.join(folder, filename), "w") as f:
            f.write("\n".join([(key + '=' + str(value)) for (key, value) in sorted(config.items(), key=lambda x: x[0]) if not key in do_not_write]))

    def get_pipeline_config_options(self):
        options = [
            ConfigOption('result_dir', default=None, type='directory', required=True),
            ConfigOption('name', default=None, type=str, required=True)
        ]
        return options

def get_names(instance, run_id, run_number):
    if isinstance(instance, list):
        for p in instance:
            if not os.path.exists(p):
                raise Exception('Invalid path: ' + str(p))	
        instance_name = "-".join(sorted([os.path.split(p)[1].split(".")[0] for p in instance]))
        if len(instance_name) > 40:
            instance_name = "-".join(sorted([os.path.split(q)[1] for q in sorted(set(os.path.split(p)[0] for p in instance))] + [str(len(instance))]))
    else:
        instance_name = os.path.basename(instance).split(".")[0]

    run_name = "run_" + str(run_id) + "_" + str(run_number)

    return "_".join(instance_name.split(':')), run_name

def get_run_result_dir(pipeline_config, instance, run_id, run_number, autonet):
    instance_name, run_name = get_names(instance, run_id, run_number)
    autonet_config = autonet.get_current_autonet_config()
    benchmark_name = '_'.join(pipeline_config['name'].split(' '))

    if 'refit_config' not in pipeline_config or pipeline_config['refit_config'] is None:
        benchmark_name += "[{0}_{1}]".format(int(autonet_config['min_budget']), int(autonet_config['max_budget']))
    elif 'refit_budget' not in pipeline_config or pipeline_config['refit_budget'] is None:
        benchmark_name += "[refit_{0}]".format(int(autonet_config['max_budget']))
    else:
        benchmark_name += "[refit_{0}]".format(int(pipeline_config['refit_budget']))

    run_result_dir = os.path.join(pipeline_config['result_dir'], instance_name, benchmark_name, run_name)
    return run_result_dir
