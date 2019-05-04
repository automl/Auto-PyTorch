__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import inspect
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.pipeline.nodes.metric_selector import default_minimize_transform, no_transform

from autoPyTorch.utils.config.config_option import ConfigOption

class LogFunctionsSelector(PipelineNode):
    def __init__(self):
        super(LogFunctionsSelector, self).__init__()

        self.log_functions = dict()

    def fit(self, pipeline_config):
        return {'log_functions': [self.log_functions[log_function] for log_function in pipeline_config["additional_logs"]]}

    def add_log_function(self, name, log_function, loss_transform=False):
        """Add a log function, will be called with the current trained network and the current training epoch
        
        Arguments:
            name {string} -- name of log function for definition in config
            log_function {function} -- log function called with network and epoch
        """

        if (not hasattr(log_function, '__call__')):
            raise ValueError("log function has to be a function")

        if isinstance(loss_transform, bool):
            loss_transform = default_minimize_transform if loss_transform else no_transform

        self.log_functions[name] = AutoNetLog(name, log_function, loss_transform)

    def remove_log_function(self, name):
        del self.log_functions[name]

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="additional_logs", default=[], type=str, list=True, choices=list(self.log_functions.keys())),
        ]
        return options


class AutoNetLog():
    def __init__(self, name, log, loss_transform):
        self.loss_transform = loss_transform
        self.log = log
        self.name = name

    def __call__(self, *args):
        return self.log(*args)

    def get_loss_value(self, *args):
        return self.loss_transform(self.__call__(*args))
