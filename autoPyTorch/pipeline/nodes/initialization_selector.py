__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.components.networks.initialization import BaseInitialization, SimpleInitializer

import torch
import torch.nn as nn
import ConfigSpace
import ConfigSpace.hyperparameters as CSH
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.utils.config.config_option import ConfigOption

Initializer = SimpleInitializer

class InitializationSelector(PipelineNode):
    def __init__(self):
        super(InitializationSelector, self).__init__()

        self.initialization_methods = {
            "default": BaseInitialization
        }

        self.initializers = dict()
        self.default_initializer = None

    def fit(self, hyperparameter_config, pipeline_config, network):
        config = ConfigWrapper(self.get_name(), hyperparameter_config)

        method_type = self.initialization_methods[config["initialization_method"]]
        method_config = ConfigWrapper(config["initialization_method"], config)
        initializer_type = self.initializers[pipeline_config["initializer"]]
        initializer_config = ConfigWrapper("initializer", config)

        torch.manual_seed(pipeline_config["random_seed"])
        initializer = initializer_type(initializer_config)
        method = method_type(initializer, method_config)
        method.apply(network)
        
        return dict()
    
    def add_initialization_method(self, name, initialization_method):
        if not issubclass(initialization_method, BaseInitialization):
            raise ValueError("initialization has to inherit from BaseInitialization")
        self.initialization_methods[name] = initialization_method
    
    def remove_initialization_method(self, name):
        del self.initialization_methods[name]

    def add_initializer(self, name, initializer, is_default_initializer=False):
        self.initializers[name] = initializer

        if (not self.default_initializer or is_default_initializer):
            self.default_initializer = name

    def get_hyperparameter_search_space(self, dataset_info=None, **pipeline_config):
        pipeline_config = self.pipeline.get_pipeline_config(**pipeline_config)
        cs = ConfigSpace.ConfigurationSpace()

        # add hyperparameters of initialization method
        possible_initialization_methods = set(pipeline_config["initialization_methods"]).intersection(self.initialization_methods.keys())
        selector = cs.add_hyperparameter(CSH.CategoricalHyperparameter("initialization_method", sorted(possible_initialization_methods)))

        for method_name, method_type in self.initialization_methods.items():
            if (method_name not in possible_initialization_methods):
                continue
            method_cs = method_type.get_hyperparameter_search_space(
                **self._get_search_space_updates(prefix=method_name))
            cs.add_configuration_space(prefix=method_name, configuration_space=method_cs, delimiter=ConfigWrapper.delimiter, 
                                       parent_hyperparameter={'parent': selector, 'value': method_name})

        # add hyperparameter of initializer
        initializer = self.initializers[pipeline_config["initializer"]]
        initializer_cs = initializer.get_hyperparameter_search_space(**self._get_search_space_updates(prefix="initializer"))
        cs.add_configuration_space(prefix="initializer", configuration_space=initializer_cs, delimiter=ConfigWrapper.delimiter)

        self._check_search_space_updates(("initializer", "*"), (possible_initialization_methods, "*"))
        return cs

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="initialization_methods", default=list(self.initialization_methods.keys()), type=str, list=True, choices=list(self.initialization_methods.keys())),
            ConfigOption(name="initializer", default=self.default_initializer, type=str, choices=list(self.initializers.keys()))
        ]
        return options
