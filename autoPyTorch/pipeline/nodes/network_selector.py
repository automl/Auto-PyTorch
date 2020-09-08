__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.components.networks.base_net import BaseNet
from autoPyTorch.components.optimizer.optimizer import Lookahead

import torch
import torch.nn as nn
import ConfigSpace
import ConfigSpace.hyperparameters as CSH
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool

class NetworkSelector(PipelineNode):
    def __init__(self):
        super(NetworkSelector, self).__init__()

        self.networks = dict()

        self.final_activations = dict()
        self.default_final_activation = None

    def fit(self, hyperparameter_config, pipeline_config, X, Y, embedding):
        config = ConfigWrapper(self.get_name(), hyperparameter_config)

        network_type = self.networks[config["network"]]
        network_config = ConfigWrapper(config["network"], config)
        activation = self.final_activations[pipeline_config["final_activation"]]

        in_features = X.shape[1:] if not embedding else (embedding.num_out_feats, )
        if len(in_features) == 1:
            # feature data
            in_features = in_features[0]

        torch.manual_seed(pipeline_config["random_seed"]) 
        network = network_type( config=network_config, 
                                in_features=in_features, out_features=Y.shape[1],
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

    def get_hyperparameter_search_space(
            self,
            dataset_info=None,
            **pipeline_config
    ):
        pipeline_config = self.pipeline.get_pipeline_config(**pipeline_config)
        cs = ConfigSpace.ConfigurationSpace()

        possible_networks = set(pipeline_config["networks"]).intersection(self.networks.keys())
        selector = cs.add_hyperparameter(CSH.CategoricalHyperparameter("network", possible_networks))
        use_swa = cs.add_hyperparameter(
            ConfigSpace.CategoricalHyperparameter(
                "use_swa",
                pipeline_config["use_swa"],
            )
        )
        look_ahead = cs.add_hyperparameter(
            ConfigSpace.CategoricalHyperparameter(
                "use_lookahead",
                pipeline_config["use_lookahead"],
            )
        )
        
        use_se = cs.add_hyperparameter(
            ConfigSpace.CategoricalHyperparameter(
                "use_se",
                pipeline_config["use_se"],
            )
        )

        if True in pipeline_config["use_se"]:
            # Note, this is not easy to be considered as a hyperparameter.
            # When used with cyclic learning rates, it depends on the number
            # of restarts.
            se_lastk = ConfigSpace.Constant('se_lastk', 3)
            cs.add_hyperparameter(se_lastk)
            cond = ConfigSpace.EqualsCondition(se_lastk, use_se, True)
            cs.add_condition(cond)

        if True in pipeline_config["use_lookahead"]:
            cs.add_configuration_space(
                prefix='lookahead',
                configuration_space=Lookahead.get_config_space(),
                delimiter=ConfigWrapper.delimiter,
                parent_hyperparameter={'parent': look_ahead, 'value': True}
            )

        if (True in pipeline_config["use_se"]) and (True in pipeline_config["use_swa"]):
            forbidden_clause = ConfigSpace.ForbiddenAndConjunction(
                ConfigSpace.ForbiddenEqualsClause(use_swa, True),
                ConfigSpace.ForbiddenEqualsClause(use_se, True)
            )
            cs.add_forbidden_clause(forbidden_clause)
        
        network_list = list()
        for network_name, network_type in self.networks.items():
            if (network_name not in possible_networks):
                continue
            network_list.append(network_name)
            network_cs = network_type.get_config_space(
                **self._get_search_space_updates(prefix=network_name))
            cs.add_configuration_space(prefix=network_name, configuration_space=network_cs, delimiter=ConfigWrapper.delimiter,
                                       parent_hyperparameter={'parent': selector, 'value': network_name})
        self._check_search_space_updates((possible_networks, "*"))

        return cs

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="networks", default=list(self.networks.keys()), type=str, list=True, choices=list(self.networks.keys())),
            ConfigOption(name="final_activation", default=self.default_final_activation, type=str, choices=list(self.final_activations.keys())),
            ConfigOption(name="use_lookahead", default=[True, False], type=to_bool, choices=[True, False], list=True, info='Use lookahead'),
            ConfigOption(name="use_swa", default=[True, False], type=to_bool, choices=[True, False], list=True, info='Use stochastic weight averaging'),
            ConfigOption(name="use_se", default=[True, False], type=to_bool, choices=[True, False], list=True, info='Use snapshot ensembling')
        ]
        return options