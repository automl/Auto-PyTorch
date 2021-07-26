__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

from autoPyTorch.components.optimizer.optimizer import AutoNetOptimizerBase

import torch.nn as nn
import ConfigSpace
import ConfigSpace.hyperparameters as CSH
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.utils.config.config_option import ConfigOption

class OptimizerSelector(PipelineNode):
    def __init__(self):
        super(OptimizerSelector, self).__init__()

        self.optimizer = dict()

    def fit(self, hyperparameter_config, pipeline_config, network):
        config = ConfigWrapper(self.get_name(), hyperparameter_config)

        optimizer_name = config['optimizer']

        optimizer_type = self.optimizer[optimizer_name]
        optimizer_config = ConfigWrapper(optimizer_name, config)
        
        return {'optimizer': optimizer_type(network.parameters(), optimizer_config)}

    def add_optimizer(self, name, optimizer_type):
        if (not issubclass(optimizer_type, AutoNetOptimizerBase)):
            raise ValueError("optimizer type has to inherit from AutoNetOptimizerBase")
        self.optimizer[name] = optimizer_type

    def remove_optimizer(self, name):
        del self.optimizer[name]

    def get_hyperparameter_search_space(self, **pipeline_config):
        pipeline_config = self.pipeline.get_pipeline_config(**pipeline_config)
        cs = ConfigSpace.ConfigurationSpace()

        selector = None
        selector = cs.add_hyperparameter(CSH.CategoricalHyperparameter("optimizer", list(self.optimizer.keys())))
        
        for optimizer_name, optimizer_type in self.optimizer.items():
            optimizer_cs = optimizer_type.get_config_space()

            parent = {'parent': selector, 'value': optimizer_name}
            cs.add_configuration_space( prefix=optimizer_name, configuration_space=optimizer_cs, delimiter=ConfigWrapper.delimiter, 
                                        parent_hyperparameter=parent)

        possible_optimizer = sorted(set(pipeline_config["optimizer"]).intersection(self.optimizer.keys()))
        self._update_hyperparameter_range('optimizer', possible_optimizer, check_validity=False, override_if_already_modified=False)

        return self._apply_user_updates(cs)

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="optimizer", default=list(self.optimizer.keys()), type=str, list=True, choices=list(self.optimizer.keys())),
        ]
        return options