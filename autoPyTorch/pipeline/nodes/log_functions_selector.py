__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import inspect
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

from autoPyTorch.utils.config.config_option import ConfigOption

class LogFunctionsSelector(PipelineNode):
    def __init__(self):
        super(LogFunctionsSelector, self).__init__()

        self.log_functions = dict()

    def fit(self, pipeline_config):
        return {'log_functions': [self.log_functions[log_function] for log_function in pipeline_config["additional_logs"]]}

    def add_log_function(self, name, log_function):
        """Add a log function, will be called with the current trained network and the current training epoch
        
        Arguments:
            name {string} -- name of log function for definition in config
            log_function {function} -- log function called with network and epoch
        """

        if (not hasattr(log_function, '__call__')):
            raise ValueError("log function has to be a function")
        self.log_functions[name] = log_function
        log_function.__name__ = name

    def remove_log_function(self, name):
        del self.log_functions[name]

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="additional_logs", default=[], type=str, list=True, choices=list(self.log_functions.keys())),
        ]
        return options