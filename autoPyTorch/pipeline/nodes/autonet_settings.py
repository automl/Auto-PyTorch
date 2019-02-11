__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import logging
import numpy as np
import sys

from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.hyperparameter_search_space_update import parse_hyperparameter_search_space_updates

from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from hpbandster.core.result import json_result_logger

class AutoNetSettings(PipelineNode):
    def __init__(self):
        super(AutoNetSettings, self).__init__()

        self.logger_settings = dict()
        self.logger_settings['debug'] = logging.DEBUG
        self.logger_settings['info'] = logging.INFO
        self.logger_settings['warning'] = logging.WARNING
        self.logger_settings['error'] = logging.ERROR
        self.logger_settings['critical'] = logging.CRITICAL


    def fit(self, pipeline_config, X_train, Y_train, X_valid, Y_valid, refit=False):

        autonet_logger = logging.getLogger('autonet')
        hpbandster_logger = logging.getLogger('hpbandster')

        level = self.logger_settings[pipeline_config['log_level']]
        autonet_logger.setLevel(level)
        hpbandster_logger.setLevel(level)

        autonet_logger.info("Start autonet with config:\n" + str(pipeline_config))
        result_logger = []
        if not refit:
            result_logger = json_result_logger(directory=pipeline_config["result_logger_dir"], overwrite=True)
        return { 'X_train': X_train, 'Y_train': Y_train, 'X_valid': X_valid, 'Y_valid': Y_valid, 'result_loggers':  [result_logger]}

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name='log_level', default='warning', type=str, choices=list(self.logger_settings.keys())),
            ConfigOption(name='random_seed', default=lambda c: abs(hash(c["run_id"])) % (2 ** 32), type=int, depends=True, info="Make sure to specify the same seed for all workers."),
            ConfigOption(name='hyperparameter_search_space_updates', default=None, type=["directory", parse_hyperparameter_search_space_updates],
                info="object of type HyperparameterSearchSpaceUpdates"),
            ConfigOption("result_logger_dir", default=".", type="directory")
        ]
        return options