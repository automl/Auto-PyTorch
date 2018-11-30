__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import torch.nn as nn
import ConfigSpace
import ConfigSpace.hyperparameters as CSH

from autonet.pipeline.base.pipeline_node import PipelineNode
from autonet.pipeline.nodes.preprocessor_selector import PreprocessorSelector

from autonet.utils.configspace_wrapper import ConfigWrapper
from autonet.utils.config.config_option import ConfigOption

from autonet.components.networks.feature.embedding import NoEmbedding

class EmbeddingSelector(PipelineNode):
    def __init__(self):
        """ Embedding selector. """

        super(EmbeddingSelector, self).__init__()

        self.embedding_modules = dict()
        self.add_embedding_module('none', NoEmbedding)

    def fit(self, hyperparameter_config, pipeline_config, X_train, one_hot_encoder):

        if not one_hot_encoder or not one_hot_encoder.categories_:
            # no categorical features -> no embedding
            return {'embedding': nn.Sequential()}

        hyperparameter_config = ConfigWrapper(self.get_name(), hyperparameter_config)
        
        embedding_name = hyperparameter_config['embedding'] if 'embedding' in hyperparameter_config else 'none'
        embedding_type = self.embedding_modules[embedding_name]
        embedding_config = ConfigWrapper(embedding_name, hyperparameter_config)

        return {'embedding': embedding_type(embedding_config, X_train.shape[1], one_hot_encoder)}


    def add_embedding_module(self, name, embedding_module):
        """Add embedding module.
        Will be created with (hyperparameter_config, in_features, categorical_embedding).
        
        Arguments:
            name {string} -- name of embedding
            embedding_module {nn.Module} -- embedding module type has to inherit from nn.Module and provide static 'get_config_space' function
        """


        if (not issubclass(embedding_module, nn.Module)):
            raise ValueError("Specified embedding module has to inherit from nn.Module")
        if (not hasattr(embedding_module, 'get_config_space')):
            raise ValueError("Specified embedding module has to implement get_config_space function")
        self.embedding_modules[name] = embedding_module


    def remove_log_function(self, name):
        del self.embedding_modules[name]

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="embeddings", default=list(self.embedding_modules.keys()), type=str, list=True, choices=list(self.embedding_modules.keys())),
        ]
        return options

    def get_hyperparameter_search_space(self, **pipeline_config):
        pipeline_config = self.pipeline.get_pipeline_config(**pipeline_config)
        cs = ConfigSpace.ConfigurationSpace()

        if pipeline_config['categorical_features'] is None or not any(pipeline_config['categorical_features']) or 'none' not in pipeline_config['preprocessors']:
            # no categorical features -> no embedding
            return cs

        possible_embeddings = set(pipeline_config["embeddings"]).intersection(self.embedding_modules.keys())
        selector = cs.add_hyperparameter(CSH.CategoricalHyperparameter("embedding", possible_embeddings, default_value="none"))
        
        for embedding_name, embedding_type in self.embedding_modules.items():
            if (embedding_name not in possible_embeddings):
                continue
            embedding_cs = embedding_type.get_config_space(pipeline_config['categorical_features'])
            cs.add_configuration_space( prefix=embedding_name, configuration_space=embedding_cs, delimiter=ConfigWrapper.delimiter, 
                                        parent_hyperparameter={'parent': selector, 'value': embedding_name})

        return self._apply_user_updates(cs)
    
    def insert_inter_node_hyperparameter_dependencies(self, config_space, **pipeline_config):
        if pipeline_config['categorical_features'] is None or not any(pipeline_config['categorical_features']) or 'none' not in pipeline_config['preprocessors']:
            # no categorical features -> no embedding
            return config_space
        embedding_hyperparameter = config_space.get_hyperparameter(EmbeddingSelector.get_name() + ConfigWrapper.delimiter + "embedding")
        preprocessor_hyperparameter = config_space.get_hyperparameter(PreprocessorSelector.get_name() + ConfigWrapper.delimiter + "preprocessor")

        condition = ConfigSpace.EqualsCondition(embedding_hyperparameter, preprocessor_hyperparameter, "none")

        config_space.add_condition(condition)
        return config_space
