__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.components.preprocessing.preprocessor_base import PreprocessorBase

import ConfigSpace
import ConfigSpace.hyperparameters as CSH
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.utils.config.config_option import ConfigOption
from autoPyTorch.components.preprocessing.preprocessor_base import PreprocessorBase

class PreprocessorSelector(PipelineNode):
    def __init__(self):
        super(PreprocessorSelector, self).__init__()
        self.preprocessors = dict()
        self.add_preprocessor('none', PreprocessorBase)

    def fit(self, hyperparameter_config, pipeline_config, X, Y, train_indices, one_hot_encoder):
        hyperparameter_config = ConfigWrapper(self.get_name(), hyperparameter_config)

        preprocessor_name = hyperparameter_config['preprocessor']
        preprocessor_type = self.preprocessors[preprocessor_name]
        preprocessor_config = ConfigWrapper(preprocessor_name, hyperparameter_config)
        preprocessor = preprocessor_type(preprocessor_config)
        preprocessor.fit(X[train_indices], Y[train_indices])

        if preprocessor_name != 'none':
            one_hot_encoder = None

        X = preprocessor.transform(X)

        return {'X': X, 'preprocessor': preprocessor, 'one_hot_encoder': one_hot_encoder}

    def predict(self, preprocessor, X):
        return { 'X': preprocessor.transform(X) }

    def add_preprocessor(self, name, preprocessor_type):
        if (not issubclass(preprocessor_type, PreprocessorBase)):
            raise ValueError("preprocessor type has to inherit from PreprocessorBase")
        if (not hasattr(preprocessor_type, "get_hyperparameter_search_space")):
            raise ValueError("preprocessor type has to implement the function get_hyperparameter_search_space")
            
        self.preprocessors[name] = preprocessor_type

    def remove_preprocessor(self, name):
        del self.preprocessors[name]

    def get_hyperparameter_search_space(self, dataset_info=None, **pipeline_config):
        pipeline_config = self.pipeline.get_pipeline_config(**pipeline_config)
        cs = ConfigSpace.ConfigurationSpace()

        possible_preprocessors = set(pipeline_config["preprocessors"]).intersection(self.preprocessors.keys())
        possible_preprocessors = list(possible_preprocessors)
        selector = cs.add_hyperparameter(CSH.CategoricalHyperparameter("preprocessor", possible_preprocessors))
        
        for preprocessor_name, preprocessor_type in self.preprocessors.items():
            if (preprocessor_name not in possible_preprocessors):
                continue
            preprocessor_cs = preprocessor_type.get_hyperparameter_search_space(dataset_info=dataset_info,
                **self._get_search_space_updates(prefix=preprocessor_name))
            cs.add_configuration_space( prefix=preprocessor_name, configuration_space=preprocessor_cs, delimiter=ConfigWrapper.delimiter, 
                                        parent_hyperparameter={'parent': selector, 'value': preprocessor_name})

        self._check_search_space_updates((possible_preprocessors, "*"))
        return cs

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="preprocessors", default=list(self.preprocessors.keys()), type=str, list=True, choices=list(self.preprocessors.keys())),
        ]
        return options
