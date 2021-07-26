import os
import logging
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser
from autoPyTorch.pipeline.base.sub_pipeline_node import SubPipelineNode
import traceback

class ForInstance(SubPipelineNode):
    def fit(self, pipeline_config, task_id, run_id):
        instances = self.get_instances(pipeline_config, instance_slice=self.parse_slice(pipeline_config["instance_slice"]))
        for instance in instances:
            try:
                self.sub_pipeline.fit_pipeline(pipeline_config=pipeline_config, instance=instance, run_id=run_id, task_id=task_id)
            except Exception as e:
                print(e)
                traceback.print_exc()
        return dict()

    def get_pipeline_config_options(self):
        options = [
            ConfigOption("instances", default=None, type='directory', required=True),
            ConfigOption("instance_slice", default=None, type=str),
            ConfigOption("dataset_root", default=ConfigFileParser.get_autonet_home(), type='directory'),
            ConfigOption("multiple_datasets_indices", default=None, type=int, list=True),
        ]
        return options

    def get_instances(self, benchmark_config, instances_must_exist=True, instance_slice=None):
        # get list of instances
        instances = []
        if os.path.isfile(benchmark_config["instances"]):

            with open(benchmark_config["instances"], "r") as instances_file:
                if os.path.splitext(benchmark_config['instances'])[1] == '.json':
                    import json
                    datasets = [make_path(path, benchmark_config["dataset_root"]) for path in json.load(instances_file)]
                    instances.append(datasets if benchmark_config['multiple_datasets_indices'] is None else [datasets[i] for i in benchmark_config['multiple_datasets_indices']])
                else:
                    for line in instances_file:
                        if line.strip().startswith("openml"):
                            instances.append(line.strip())
                            continue
                            
                        if line.strip().startswith("["):
                            datasets = [make_path(path, benchmark_config["dataset_root"]) for path in line.strip(' []\n').split(',')]
                            instances.append(datasets if benchmark_config['multiple_datasets_indices'] is None else [datasets[i] for i in benchmark_config['multiple_datasets_indices']])
                            continue

                        instance = os.path.abspath(os.path.join(benchmark_config["dataset_root"], line.strip()))
                        if os.path.isfile(instance) or os.path.isdir(instance):
                            instances.append(instance)
                        else:
                            if not instances_must_exist:
                                instances.append(instance)
                            logging.getLogger('benchmark').warning(str(instance) + " does not exist")
        elif os.path.isdir(benchmark_config["instances"]):
            for root, directories, filenames in os.walk(benchmark_config["instances"]):
                for filename in filenames: 
                    instances.append(os.path.join(root,filename))
        if instance_slice is not None:
            return instances[instance_slice]
        return instances

    def parse_slice(self, splice_string):
        if (splice_string is None):
            return None

        split = splice_string.split(":")
        if len(split) == 1:
            start = int(split[0]) if split[0] != "" else 0
            stop = (int(split[0]) + 1) if split[0] != "" else None
            step = 1
        elif len(split) == 2:
            start = int(split[0]) if split[0] != "" else 0
            stop = int(split[1]) if split[1] != "" else None
            step = 1
        elif len(split) == 3:
            start = int(split[0]) if split[0] != "" else 0
            stop = int(split[1]) if split[1] != "" else None
            step = int(split[2]) if split[2] != "" else 1
        return slice(start, stop, step)


def make_path(path, root):
    path = path.strip()
    if not os.path.isabs(path):
        path = os.path.join(root, path)
    if os.path.exists(path):
        return os.path.abspath(path)
    raise Exception('Invalid dataset path: ' + str(path))
