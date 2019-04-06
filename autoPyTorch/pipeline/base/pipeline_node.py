__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

from copy import deepcopy
import ConfigSpace
import inspect
from autoPyTorch.utils.config.config_option import ConfigOption
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.pipeline.base.node import Node


class PipelineNode(Node):
    def __init__(self):
        """A pipeline node is a step in a pipeline.
        It can implement a fit function:
            Returns a dictionary.
            Input parameter (kwargs) are given by previous fit function computations in the pipeline.
        It can implement a predict function:
            Returns a dictionary.
            Input parameter (kwargs) are given by previous predict function computations in the pipeline or defined in fit function output of this node.

        Each node can provide a list of config options that the user can specify/customize.
        Each node can provide a config space for optimization.

        """

        super(PipelineNode, self).__init__()
        self._cs_updates = dict()
        self.pipeline = None

    @classmethod
    def get_name(cls):
        return cls.__name__
    
    def clone(self, skip=("pipeline", "fit_output", "predict_output", "child_node")):
        node_type = type(self)
        new_node = node_type.__new__(node_type)
        for key, value in self.__dict__.items():
            if key not in skip:
                setattr(new_node, key, deepcopy(value))
            else:
                setattr(new_node, key, None)
        return new_node

    # VIRTUAL
    def fit(self, **kwargs):
        """Fit pipeline node.
        Each node computes its fit function in linear order.
        All args have to be specified in a parent node fit output.
        
        Returns:
            [dict] -- output values that will be passed to child nodes, if required
        """

        return dict()

    # VIRTUAL
    def predict(self, **kwargs):
        """Predict pipeline node.
        Each node computes its predict function in linear order.
        All args have to be specified in a parent node predict output or in the fit output of this node
        
        Returns:
            [dict] -- output values that will be passed to child nodes, if required
        """

        return dict()

    # VIRTUAL
    def set_pipeline(self, pipeline):
        self.pipeline = pipeline

    # VIRTUAL
    def get_pipeline_config_options(self):
        """Get available ConfigOption parameter.
        
        Returns:
            List[ConfigOption] -- list of available config options
        """

        return []

    # VIRTUAL
    def get_pipeline_config_conditions(self):
        """Get the conditions on the pipeline config (e.g. max_budget > min_budget)
        
        Returns:
            List[ConfigCondition] -- list of functions, that take a pipeline config and raise an Error, if faulty configuration is detected.
        """

        return []


    # VIRTUAL
    def get_hyperparameter_search_space(self, dataset_info=None, **pipeline_config):
        """Get hyperparameter that should be optimized.
        
        Returns:
            ConfigSpace -- config space
        """
        return ConfigSpace.ConfigurationSpace()
    
    # VIRTUAL
    def insert_inter_node_hyperparameter_dependencies(self, config_space, dataset_info=None, **pipeline_config):
        """Insert Conditions and Forbiddens of hyperparameters of different nodes

        Returns:
            ConfigSpace -- config space
        """
        return config_space

    def _apply_search_space_update(self, name, new_value_range, log=False):
        """Allows the user to update a hyperparameter
        
        Arguments:
            name {string} -- name of hyperparameter
            new_value_range {List[?] -- value range can be either lower, upper or a list of possible conditionals
            log {bool} -- is hyperparameter logscale
        """

        if (len(new_value_range) == 0):
            raise ValueError("The new value range needs at least one value")
        self._cs_updates[name] = tuple([new_value_range, log])
    
    def _check_search_space_updates(self, *allowed_hps):
        exploded_allowed_hps = list()
        for allowed_hp in allowed_hps:
            add = [list()]
            allowed_hp = (allowed_hp, ) if isinstance(allowed_hp, str) else allowed_hp
            for part in allowed_hp:
                if isinstance(part, str):
                    add = [x + [part] for x in add]
                else:
                    add = [x + [p] for p in part for x in add]
            exploded_allowed_hps += add
        exploded_allowed_hps = [ConfigWrapper.delimiter.join(x) for x in exploded_allowed_hps]
        
        for key in self._get_search_space_updates().keys():
            if key not in exploded_allowed_hps and \
                    ConfigWrapper.delimiter.join(key.split(ConfigWrapper.delimiter)[:-1] + ["*"]) not in exploded_allowed_hps:
                raise ValueError("Invalid search space update given: %s" % key)
    
    def _get_search_space_updates(self, prefix=None):
        if prefix is None:
            return self._cs_updates
        if isinstance(prefix, tuple):
            prefix = ConfigWrapper.delimiter.join(prefix)
        result = dict()
        for key in self._cs_updates.keys():
            if key.startswith(prefix + ConfigWrapper.delimiter):
                result[key[len(prefix + ConfigWrapper.delimiter):]] = self._cs_updates[key]
        return result