import time
from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.pipeline.base.node import Node
import ConfigSpace
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
import traceback


class Pipeline():
    """A machine learning pipeline"""

    def __init__(self, pipeline_nodes=[]):
        """Construct a Pipeline
        
        Keyword Arguments:
            pipeline_nodes {list} -- The nodes of the pipeline (default: {[]})
        """
        self.root = Node()
        self._pipeline_nodes = dict()
        self._parent_pipeline = None

        # add all the given nodes to the pipeline
        last_node = self.root
        for node in pipeline_nodes:
            last_node.child_node = node
            self.add_pipeline_node(node)
            last_node = node

    def __getitem__(self, key):
        return self._pipeline_nodes[key]
    
    def __contains__(self, key):
        if isinstance(key, str):
            return key in self._pipeline_nodes
        elif issubclass(key, PipelineNode):
            return key.get_name() in self._pipeline_nodes
        else:
            raise ValueError("Cannot check if instance " + str(key) + " of type " + str(type(key)) + " is contained in pipeline")

    def set_parent_pipeline(self, pipeline):
        """Set this pipeline as a child pipeline of the given pipeline.
        This will allow the parent pipeline to access the pipeline nodes of its child pipelines.
        
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

    def get_hyperparameter_search_space(self, dataset_info=None, **pipeline_config):
        """Get the search space of the pipeline.
        
        Keyword Arguments:
            dataset_info {DatasetInfo} -- Object describing the dataset. (default: {None})
        
        Returns:
            ConfigurationSpace -- The search space of the pipeline
        """
        pipeline_config = self.get_pipeline_config(**pipeline_config)

        # check for hyperparameter search space updates and apply them
        if "hyperparameter_search_space_updates" in pipeline_config and pipeline_config["hyperparameter_search_space_updates"] is not None:
            assert isinstance(pipeline_config["hyperparameter_search_space_updates"], HyperparameterSearchSpaceUpdates)
            pipeline_config["hyperparameter_search_space_updates"].apply(self, pipeline_config)

        # initialize the config space
        if "random_seed" in pipeline_config:
            cs = ConfigSpace.ConfigurationSpace(seed=pipeline_config["random_seed"])
        else:
            cs = ConfigSpace.ConfigurationSpace()

        # add the config space of each node
        for name, node in self._pipeline_nodes.items():
            #print("dataset_info" in pipeline_config.keys())
            config_space = node.get_hyperparameter_search_space(**pipeline_config)
            cs.add_configuration_space(prefix=name, configuration_space=config_space, delimiter=ConfigWrapper.delimiter)
        
        # add the dependencies between the nodes
        for name, node in self._pipeline_nodes.items():
            cs = node.insert_inter_node_hyperparameter_dependencies(cs, dataset_info=dataset_info, **pipeline_config)

        return cs

    def get_pipeline_config(self, throw_error_if_invalid=True, **pipeline_config):
        """Get the full pipeline config given a partial pipeline config
        
        Keyword Arguments:
            throw_error_if_invalid {bool} -- Throw an error if invalid config option is defined (default: {True})
        
        Returns:
            dict -- the full config for the pipeline, containing values for all options
        """
        options = self.get_pipeline_config_options()
        conditions = self.get_pipeline_config_conditions()
            
        parser = ConfigFileParser(options)
        pipeline_config = parser.set_defaults(pipeline_config, throw_error_if_invalid=throw_error_if_invalid)

        # check the conditions e.g. max_budget > min_budget
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
        """Get all ConfigOptions of all nodes in the pipeline.
        
        Returns:
            list -- A list of ConfigOptions.
        """
        if (self._parent_pipeline is not None):
            return self._parent_pipeline.get_pipeline_config_options()

        options = []

        for node in self._pipeline_nodes.values():
            options += node.get_pipeline_config_options()

        return options

    def get_pipeline_config_conditions(self):
        """Get all ConfigConditions of all the nodes in the pipeline.
        
        Returns:
            list -- A list of ConfigConditions
        """
        if (self._parent_pipeline is not None):
            return self._parent_pipeline.get_pipeline_config_options()
        
        conditions = []

        for node in self._pipeline_nodes.values():
            conditions += node.get_pipeline_config_conditions()
        
        return conditions
    
    def clean(self):
        self.root.clean_fit_data()

    def clone(self):
        """Clone the pipeline
        
        Returns:
            Pipeline -- The cloned pipeline
        """
        pipeline_nodes = []

        current_node = self.root.child_node
        while current_node is not None:
            pipeline_nodes.append(current_node.clone())
            current_node = current_node.child_node
        
        return type(self)(pipeline_nodes)
