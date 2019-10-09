
from autoPyTorch.utils.config.config_option import ConfigOption
from autoPyTorch.pipeline.base.sub_pipeline_node import SubPipelineNode
import traceback

class ForAutoNetConfig(SubPipelineNode):
    def fit(self, pipeline_config, autonet, instance, data_manager, run_id, task_id):
        for config_file in self.get_config_files(pipeline_config):
            try:
                self.sub_pipeline.fit_pipeline(pipeline_config=pipeline_config,
                    autonet=autonet, instance=instance, data_manager=data_manager,
                    autonet_config_file=config_file, run_id=run_id, task_id=task_id)
            except Exception as e:
                print(e)
                traceback.print_exc()
        return dict()

    def get_pipeline_config_options(self):
        options = [
            ConfigOption("autonet_configs", default=None, type='directory', list=True, required=True),
            ConfigOption("autonet_config_root", default=ConfigFileParser.get_autonet_home(), type='directory'),
            ConfigOption("autonet_config_slice", default=None, type=str)
        ]
        return options

    @staticmethod
    def get_config_files(pipeline_config, parse_slice=True):
        config_files = pipeline_config['autonet_configs']
        if pipeline_config['autonet_config_root'] is not None:
            config_files = [os.path.join(pipeline_config['autonet_config_root'], config) if not os.path.isabs(config) else config for config in config_files]

        autonet_config_slice = ForAutoNetConfig.parse_slice(pipeline_config['autonet_config_slice'])
        if autonet_config_slice is not None and parse_slice:
            return config_files[autonet_config_slice]

        return config_files

    @staticmethod
    def parse_slice(splice_string):
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
