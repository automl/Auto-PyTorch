__author__ = "Michael Burkart"
__version__ = "0.0.1"
__license__ = "BSD"

import logging
import numpy as np
import sys, os
import pprint

from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.hyperparameter_search_space_update import parse_hyperparameter_search_space_updates

from autoPyTorch.utils.config.config_option import ConfigOption, to_bool

import random, torch

class AutoNetSettingsNoShuffle(PipelineNode):
    def __init__(self):
        super(AutoNetSettingsNoShuffle, self).__init__()

        self.logger_settings = dict()
        self.logger_settings['debug'] = logging.DEBUG
        self.logger_settings['info'] = logging.INFO
        self.logger_settings['warning'] = logging.WARNING
        self.logger_settings['error'] = logging.ERROR
        self.logger_settings['critical'] = logging.CRITICAL


    def fit(self, pipeline_config, X_train, Y_train, X_valid, Y_valid):

        autonet_logger = logging.getLogger('autonet')
        hpbandster_logger = logging.getLogger('hpbandster')

        level = self.logger_settings[pipeline_config['log_level']]
        autonet_logger.setLevel(level)
        hpbandster_logger.setLevel(level)

        random.seed(pipeline_config['random_seed'])
        torch.manual_seed(pipeline_config['random_seed'])
        np.random.seed(pipeline_config['random_seed'])

        if 'result_logger_dir' in pipeline_config:
            directory = os.path.join(pipeline_config['result_logger_dir'], "worker_logs_" + str(pipeline_config['task_id']))
            os.makedirs(directory, exist_ok=True)

            if level == logging.DEBUG:
                self.addHandler([autonet_logger, hpbandster_logger], level, os.path.join(directory, 'autonet_debug.log'))
                self.addHandler([autonet_logger, hpbandster_logger], logging.INFO, os.path.join(directory, 'autonet_info.log'))
            else:
                self.addHandler([autonet_logger, hpbandster_logger], level, os.path.join(directory, 'autonet.log'))

        autonet_logger.info("Start autonet with config:\n" + str(pprint.pformat(pipeline_config)))

        return { 'X_train': X_train, 'Y_train': Y_train, 'X_valid': X_valid, 'Y_valid': Y_valid }

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name='log_level', default='warning', type=str, choices=list(self.logger_settings.keys())),
            ConfigOption(name='random_seed', default=lambda c: abs(hash(c["run_id"])) % (2 ** 32), type=int, depends=True, info="Make sure to specify the same seed for all workers."),
            ConfigOption(name='hyperparameter_search_space_updates', default=None, type=["directory", parse_hyperparameter_search_space_updates],
                info="object of type HyperparameterSearchSpaceUpdates"),
        ]
        return options

    def addHandler(self, loggers, level, path):
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(path)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        for logger in loggers:
            logger.addHandler(fh)
