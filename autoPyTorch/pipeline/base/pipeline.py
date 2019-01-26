import time
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.pipeline.base.node import Node
import ConfigSpace
import traceback
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates


class Pipeline():
    def __init__(self, pipeline_nodes=[]):
        self.root = Node()
        self._pipeline_nodes = dict()
        self.start_params = None
        self._parent_pipeline = None

        last_node = self.root
        for node in pipeline_nodes:
            last_node.child_node = node
            self.add_pipeline_node(node)
            last_node = node

    def _get_start_parameter(self):
        return self.start_params

    def __getitem__(self, key):
        return self._pipeline_nodes[key]

    def set_parent_pipeline(self, pipeline):
        """Set this pipeline as a child pipeline of the given pipeline.
        This will allow the parent pipeline to access the pipeline nodes of its child pipelines
        
        Arguments:
            pipeline {Pipeline} -- parent pipeline
        """

        if (not issubclass(type(pipeline), Pipeline)):
            raise ValueError("Given pipeline has to be of type Pipeline, got " + str(type(pipeline)))

        self._parent_pipeline = pipeline

        for _, node in self._pipeline_nodes.items():
            self._parent_pipeline.add_pipeline_node(node)


    def fit_pipeline(self, **kwargs):
        return self.root.fit_traverse(**kwargs)

    def predict_pipeline(self, **kwargs):
        return self.root.predict_traverse(**kwargs)

    def add_pipeline_node(self, pipeline_node):
        """Add a node to the pipeline
        
        Arguments:
            pipeline_node {PipelineNode} -- node
        
        Returns:
            PipelineNode -- return input node
        """

        if (not issubclass(type(pipeline_node), PipelineNode)):
            raise ValueError("You can only add PipelineElement subclasses to the pipeline")
        
        self._pipeline_nodes[pipeline_node.get_name()] = pipeline_node
        pipeline_node.set_pipeline(self)

        if (self._parent_pipeline):
            self._parent_pipeline.add_pipeline_node(pipeline_node)

        return pipeline_node

    def get_hyperparameter_search_space(self, **pipeline_config):
        pipeline_config = self.get_pipeline_config(**pipeline_config)

        if "hyperparameter_search_space_updates" in pipeline_config and pipeline_config["hyperparameter_search_space_updates"] is not None:
            assert isinstance(pipeline_config["hyperparameter_search_space_updates"], HyperparameterSearchSpaceUpdates)
            pipeline_config["hyperparameter_search_space_updates"].apply(self, pipeline_config)

        if "random_seed" in pipeline_config:
            cs = ConfigSpace.ConfigurationSpace(seed=pipeline_config["random_seed"])
        else:
            cs = ConfigSpace.ConfigurationSpace()

        for name, node in self._pipeline_nodes.items():
            config_space = node.get_hyperparameter_search_space(**pipeline_config)
            cs.add_configuration_space(prefix=name, configuration_space=config_space, delimiter=ConfigWrapper.delimiter)
        
        for name, node in self._pipeline_nodes.items():
            cs = node.insert_inter_node_hyperparameter_dependencies(cs, **pipeline_config)

        return cs

    def get_pipeline_config(self, throw_error_if_invalid=True, **pipeline_config):
        options = self.get_pipeline_config_options()
        conditions = self.get_pipeline_config_conditions()

        parser = ConfigFileParser(options)
        pipeline_config = parser.set_defaults(pipeline_config, throw_error_if_invalid=throw_error_if_invalid)

        for c in conditions:
            try:
                c(pipeline_config)
            except Exception as e:
                if throw_error_if_invalid:
                    raise
                print(e)
                traceback.print_exc()

        return pipeline_config


    def get_pipeline_config_options(self):
        if (self._parent_pipeline is not None):
            return self._parent_pipeline.get_pipeline_config_options()

        options = []

        for node in self._pipeline_nodes.values():
            options += node.get_pipeline_config_options()

        return options
    
    def get_pipeline_config_conditions(self):
        if (self._parent_pipeline is not None):
            return self._parent_pipeline.get_pipeline_config_options()
        
        conditions = []

        for node in self._pipeline_nodes.values():
            conditions += node.get_pipeline_config_conditions()
        
        return conditions


    def print_config_space(self, **pipeline_config):
        config_space = self.get_hyperparameter_search_space(**pipeline_config)

        if (len(config_space.get_hyperparameters()) == 0):
            return
        print(config_space)

    def print_config_space_per_node(self, **pipeline_config):
        for name, node in self._pipeline_nodes.items():
            config_space = node.get_hyperparameter_search_space(**pipeline_config)

            if (len(config_space.get_hyperparameters()) == 0):
                continue
            print(name)
            print(config_space)


    def print_config_options(self):
        for option in self.get_pipeline_config_options():
            print(str(option))

    def print_config_options_per_node(self):
        for name, node in self._pipeline_nodes.items():
            print(name)
            for option in node.get_pipeline_config_options():
                print("   " + str(option))

    def print_pipeline_nodes(self):
        for name, node in self._pipeline_nodes.items():
            input_str = "["
            for edge in node.in_edges:
                input_str += " (" + edge.out_idx + ", " + edge.target.get_name() + ", " + edge.kw + ") "
            input_str += "]"
            print(name + " \t\t Input: " + input_str)

