import logging
from autoPyTorch.utils.config.config_option import ConfigOption
from autoPyTorch.pipeline.base.sub_pipeline_node import SubPipelineNode
import traceback, time

class MultipleBenchmarks(SubPipelineNode):

    def fit(self, pipeline_config, instance, task_id, run_id):
        configs = self.get_benchmark_configs(pipeline_config, instance)
        n_configs = int(len(configs) / pipeline_config['max_jobs'])

        for i in range(pipeline_config['current_job'] * n_configs, pipeline_config['current_job'] * n_configs + n_configs):
            config, instance = configs[i]
            self.sub_pipeline.fit_pipeline(pipeline_config=config, instance=instance, run_id=run_id, task_id=task_id)
                
        return dict()

    def get_pipeline_config_options(self):
        options = [
            ConfigOption("current_job", default=0, type=int),
            ConfigOption("max_jobs", default=1, type=int),
            ConfigOption("multiple_configs", default=[], type=str, list=True, required=True),
            ConfigOption("single_configs", default=[], type=str, list=True, required=True),
        ]
        return options

    def get_benchmark_configs(self, pipeline_config, instance):
        configs = []
        if not isinstance(instance, list):
            return [[pipeline_config, instance]]
        n_datasets = len(instance)

        config = dict()
        config.update(pipeline_config)
        config['autonet_configs'] = pipeline_config['autonet_configs'] + pipeline_config['multiple_configs']
        configs.append([config, instance])
        for i in range(n_datasets):
            config = dict()
            config.update(pipeline_config)
            config['autonet_configs'] = pipeline_config['autonet_configs'] + pipeline_config['single_configs']
            configs.append([config, [instance[i]]])

        return configs
