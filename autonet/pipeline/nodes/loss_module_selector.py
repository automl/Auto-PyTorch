__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import inspect
import torch

import ConfigSpace
import ConfigSpace.hyperparameters as CSH
from autonet.pipeline.base.pipeline_node import PipelineNode

from torch.nn.modules.loss import _Loss
from autonet.utils.configspace_wrapper import ConfigWrapper
from autonet.utils.config.config_option import ConfigOption


class LossModuleSelector(PipelineNode):
    def __init__(self):
        super(LossModuleSelector, self).__init__()
        self.loss_modules = dict()

    def fit(self, hyperparameter_config, pipeline_config, X_train, Y_train):
        hyperparameter_config = ConfigWrapper(self.get_name(), hyperparameter_config)

        weights = None
        loss_module = self.loss_modules[hyperparameter_config["loss_module"]]
        if (loss_module.weight_strategy != None):
            weights = loss_module.weight_strategy(pipeline_config, X_train, Y_train)
            weights = torch.from_numpy(weights).float()

        loss = loss_module.module
        if "pos_weight" in inspect.getfullargspec(loss)[0] and weights is not None and inspect.isclass(loss):
            loss = loss(pos_weight=weights)
        elif "weight" in inspect.getfullargspec(loss)[0] and weights is not None and inspect.isclass(loss):
            loss = loss(weight=weights)
        elif inspect.isclass(loss):
            loss = loss()
        loss_module.set_loss_function(loss)
        return {'loss_function': loss_module}

    def add_loss_module(self, name, loss_module, weight_strategy=None, requires_target_class_labels=False):
        """Add a loss module, has to be a pytorch loss module type
        
        Arguments:
            name {string} -- name of loss module for definition in config
            loss_module {type} -- a pytorch loss module type
            weight_strategy {function} -- callable that computes label weights
        """

        if (not issubclass(loss_module, _Loss)):
            raise ValueError("loss module has to be a subclass of torch.nn.modules.loss._Loss (all pytorch loss modules)")
        self.loss_modules[name] = AutoNetLossModule(loss_module, weight_strategy, requires_target_class_labels)

    def remove_loss_module(self, name):
        del self.loss_modules[name]

    def get_hyperparameter_search_space(self, **pipeline_config):
        pipeline_config = self.pipeline.get_pipeline_config(**pipeline_config)
        cs = ConfigSpace.ConfigurationSpace()

        possible_loss_modules = set(pipeline_config["loss_modules"]).intersection(self.loss_modules.keys())
        cs.add_hyperparameter(CSH.CategoricalHyperparameter('loss_module', list(possible_loss_modules)))

        return self._apply_user_updates(cs)
        

    def get_pipeline_config_options(self):
        loss_module_names = list(self.loss_modules.keys())
        options = [
            ConfigOption(name="loss_modules", default=loss_module_names, type=str, list=True, choices=loss_module_names),
        ]
        return options

class AutoNetLossModule():
    def __init__(self, module, weight_strategy, requires_target_class_labels):
        self.module = module
        self.weight_strategy = weight_strategy
        self.requires_target_class_labels = requires_target_class_labels
        self.function = None

    def set_loss_function(self, function):
        self.function = function

    def __call__(self, x, y):
        if not self.requires_target_class_labels:
            return self.function(x, y)
        else:
            return self.function(x, y.max(1)[1])

    def to(self, device):
        result = AutoNetLossModule(self.module, self.weight_strategy, self.requires_target_class_labels)
        result.set_loss_function(self.function.to(device))
        return result