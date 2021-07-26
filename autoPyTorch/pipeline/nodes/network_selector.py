__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.components.networks.base_net import BaseNet

import torch
import torch.nn as nn
import ConfigSpace
import ConfigSpace.hyperparameters as CSH
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.utils.config.config_option import ConfigOption

class NetworkSelector(PipelineNode):
    def __init__(self):
        super(NetworkSelector, self).__init__()

        self.networks = dict()

        self.final_activations = dict()
        self.default_final_activation = None

    def fit(self, hyperparameter_config, pipeline_config, X_train, Y_train, embedding=None):
        config = ConfigWrapper(self.get_name(), hyperparameter_config)

        network_name = config['network'] if 'network' in config else pipeline_config['networks'][0]

        network_type = self.networks[network_name]
        network_config = ConfigWrapper(network_name, config)
        activation = self.final_activations[pipeline_config["final_activation"]]

        in_features = X_train.shape[1:] if not embedding else (embedding.num_out_feats, )
        if len(in_features) == 1:
            # feature data
            in_features = in_features[0]

        torch.manual_seed(pipeline_config["random_seed"]) 
        network = network_type( config=network_config, 
                                in_features=in_features, out_features=Y_train.shape[1], 
                                embedding=embedding, final_activation=activation)
        return {'network': network}

    def predict(self, network):
        return {'network': network}

    def add_network(self, name, network_type):
        if (not issubclass(network_type, BaseNet)):
            raise ValueError("network type has to inherit from BaseNet")
        if (not hasattr(network_type, "get_config_space")):
            raise ValueError("network type has to implement the function get_config_space")
            
        self.networks[name] = network_type

    def remove_network(self, name):
        del self.networks[name]

    def add_final_activation(self, name, activation, is_default_final_activation=False):
        """Add possible final activation layer.
        One can be specified in config and will be used as a final network layer.
        
        Arguments:
            name {string} -- name of final activation, can be used to specify in the config file
            activation {nn.Module} -- final activation layer
        
        Keyword Arguments:
            is_default_final_activation {bool} -- should the given activation be the default case (default: {False})
        """

        self.final_activations[name] = activation

        if (not self.default_final_activation or is_default_final_activation):
            self.default_final_activation = name

    def get_hyperparameter_search_space(self, **pipeline_config):
        pipeline_config = self.pipeline.get_pipeline_config(**pipeline_config)
        cs = ConfigSpace.ConfigurationSpace()

        selector = None
        selector = cs.add_hyperparameter(CSH.CategoricalHyperparameter("network", list(self.networks.keys())))
        
        for network_name, network_type in self.networks.items():
            network_cs = network_type.get_config_space(user_updates=self._get_user_hyperparameter_range_updates(prefix=network_name))

            parent = {'parent': selector, 'value': network_name}
            cs.add_configuration_space(prefix=network_name, configuration_space=network_cs, delimiter=ConfigWrapper.delimiter, 
                                       parent_hyperparameter=parent)

        possible_networks = sorted(set(pipeline_config["networks"]).intersection(self.networks.keys()))
        self._update_hyperparameter_range('network', possible_networks, check_validity=False, override_if_already_modified=False)

        return self._apply_user_updates(cs)

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="networks", default=list(self.networks.keys()), type=str, list=True, choices=list(self.networks.keys())),
            ConfigOption(name="final_activation", default=self.default_final_activation, type=str, choices=list(self.final_activations.keys()))
        ]
        return options