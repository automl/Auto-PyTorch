import logging

from autoPyTorch.utils.config.config_option import ConfigOption
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.benchmarking.benchmark_pipeline.benchmark_settings import BenchmarkSettings

class VisualizationSettings(BenchmarkSettings):
    def fit(self, pipeline_config):
        logging.getLogger('benchmark').info("Start visualization")

        logger = logging.getLogger('benchmark')
        logger.setLevel(self.logger_settings[pipeline_config['log_level']])

        # log level for autonet is set in SetAutoNetConfig

        return { 'run_id_range': pipeline_config['run_id_range']}

    def get_pipeline_config_options(self):
        options = [
            ConfigOption("run_id_range", type=str, default=None),
            ConfigOption("log_level", default="info", type=str, choices=list(self.logger_settings.keys()))
        ]
        return options
