import logging

from autonet.utils.config.config_option import ConfigOption
from autonet.utils.config.config_file_parser import ConfigFileParser
from autonet.pipeline.base.pipeline_node import PipelineNode

class BenchmarkSettings(PipelineNode):
    def __init__(self):
        super(BenchmarkSettings, self).__init__()

        self.logger_settings = dict()
        self.logger_settings['debug'] = logging.DEBUG
        self.logger_settings['info'] = logging.INFO
        self.logger_settings['warning'] = logging.WARNING
        self.logger_settings['error'] = logging.ERROR
        self.logger_settings['critical'] = logging.CRITICAL

    def fit(self, pipeline_config):
        logging.getLogger('benchmark').info("Start benchmark")

        logger = logging.getLogger('benchmark')
        logger.setLevel(self.logger_settings[pipeline_config['log_level']])

        # log level for autonet is set in SetAutoNetConfig

        return { 'task_id': pipeline_config['task_id'], 'run_id': pipeline_config['run_id']}

    def get_pipeline_config_options(self):
        options = [
            ConfigOption("task_id", default=-1, type=int),
            ConfigOption("run_id", default="0", type=str),
            ConfigOption("log_level", default="info", type=str, choices=list(self.logger_settings.keys()))
        ]
        return options
