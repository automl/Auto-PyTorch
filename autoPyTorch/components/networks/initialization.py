import torch
import ConfigSpace

from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter, get_hyperparameter

class SimpleInitializer():
    initialize_layers = (
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.Linear
    )

    def __init__(self, hyperparameter_config):
        self.initialize_bias = hyperparameter_config["initialize_bias"]

    def apply(self, module, initialization_method, initialization_kwargs):
        initialization_method_bias = initialization_method
        initialization_kwargs_bias = initialization_kwargs

        if self.initialize_bias == "Zero":
            initialization_method_bias = torch.nn.init.constant_
            initialization_kwargs_bias = {"val": 0}

        def perform_initialization(m):
            if isinstance(m, self.initialize_layers):
                if initialization_method is not None:
                    initialization_method(m.weight.data, **initialization_kwargs)

                if m.bias is not None and self.initialize_bias != "No" and initialization_method_bias is not None:
                    try:
                        initialization_method_bias(m.bias.data, **initialization_kwargs_bias)
                    except ValueError:
                        pass
        module.apply(perform_initialization)
    
    @staticmethod
    def get_hyperparameter_search_space(
        initialize_bias=("Yes", "No", "Zero")
    ):
        cs = ConfigSpace.ConfigurationSpace()
        add_hyperparameter(cs, ConfigSpace.CategoricalHyperparameter, "initialize_bias", initialize_bias)
        return cs


class BaseInitialization():
    initialization_method = None

    def __init__(self, initializer, hyperparameter_config):
        self.initializer = initializer
        self.hyperparameter_config = hyperparameter_config
    
    def apply(self, module):
        initialization_kwargs = self.hyperparameter_config if isinstance(self.hyperparameter_config, dict) else self.hyperparameter_config.get_dictionary()
        self.initializer.apply(module, self.initialization_method, initialization_kwargs)
    
    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigSpace.ConfigurationSpace()
        return cs


class SparseInitialization(BaseInitialization):
    initialization_method = staticmethod(torch.nn.init.sparse_)

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameter(ConfigSpace.Constant("sparsity", 0.9))
        return cs