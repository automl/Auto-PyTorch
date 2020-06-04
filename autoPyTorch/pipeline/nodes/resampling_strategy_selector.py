import numpy as np


__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption
from autoPyTorch.components.preprocessing.resampling_base import ResamplingMethodNone, ResamplingMethodBase, TargetSizeStrategyBase
from autoPyTorch.pipeline.nodes.cross_validation import CrossValidation
from sklearn.preprocessing import OneHotEncoder
import ConfigSpace
import ConfigSpace.hyperparameters as CSH
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
import logging

class ResamplingStrategySelector(PipelineNode):
    def __init__(self):
        super(ResamplingStrategySelector, self).__init__()

        self.over_sampling_methods = dict()
        self.add_over_sampling_method('none', ResamplingMethodNone)

        self.under_sampling_methods = dict()
        self.add_under_sampling_method('none',   ResamplingMethodNone)

        self.target_size_strategies = {'none': None}

    def fit(self, pipeline_config, hyperparameter_config, X, Y, train_indices, valid_indices):
        hyperparameter_config = ConfigWrapper(self.get_name(), hyperparameter_config)
        logger = logging.getLogger('autonet')
        
        if hyperparameter_config['target_size_strategy'] == 'none':
            return dict()

        over_sampling_method = self.over_sampling_methods[hyperparameter_config['over_sampling_method']](
            ConfigWrapper(hyperparameter_config['over_sampling_method'], hyperparameter_config)
        )
        under_sampling_method = self.under_sampling_methods[hyperparameter_config['under_sampling_method']](
            ConfigWrapper(hyperparameter_config['under_sampling_method'], hyperparameter_config)
        )
        target_size_strategy = self.target_size_strategies[hyperparameter_config['target_size_strategy']]()

        y = np.argmax(Y[train_indices], axis=1).astype(int)
        ohe = OneHotEncoder(categories="auto", sparse=False)
        ohe.fit(y.reshape((-1, 1)))

        over_sampling_target_size = target_size_strategy.over_sample_strategy(y)
        under_sampling_target_size = target_size_strategy.under_sample_strategy(y)

        logger.debug("Distribution before resample: " + str(np.unique(y, return_counts=True)[1]))
        X_resampled, y_resampled = over_sampling_method.resample(X[train_indices], y, over_sampling_target_size, pipeline_config["random_seed"])
        X_resampled, y_resampled  = under_sampling_method.resample(X_resampled, y_resampled, under_sampling_target_size, pipeline_config["random_seed"])
        logger.debug("Distribution after resample: " + str(np.unique(y_resampled, return_counts=True)[1]))

        if valid_indices is None:
            return {"X": X_resampled, "Y": ohe.transform(y_resampled.reshape((-1, 1))), "train_indices": np.array(list(range(X_resampled.shape[0])))}

        X, Y, split_indices = CrossValidation.get_validation_set_split_indices(pipeline_config,
                X_train=X_resampled, X_valid=X[valid_indices],
                Y_train=ohe.transform(y_resampled.reshape((-1, 1))), Y_valid=Y[valid_indices], allow_shuffle=False)
        return {"X": X, "Y": Y, "train_indices": split_indices[0], "valid_indices": split_indices[1]}

    def add_over_sampling_method(self, name, resampling_method):
        """Add a resampling strategy.
        Will be called with {X_train, Y_train}
        
        Arguments:
            name {string} -- name of resampling strategy for definition in config
            resampling_strategy {function} -- callable with {pipeline_config, X_train, Y_train}
        """

        if (not issubclass(resampling_method, ResamplingMethodBase)):
            raise ValueError("Resampling method must be subclass of ResamplingMethodBase")

        self.over_sampling_methods[name] = resampling_method

    def add_under_sampling_method(self, name, resampling_method):
        """Add a resampling strategy.
        Will be called with {X_train, Y_train}
        
        Arguments:
            name {string} -- name of resampling strategy for definition in config
            resampling_strategy {function} -- callable with {pipeline_config, X_train, Y_train}
        """

        if (not issubclass(resampling_method, ResamplingMethodBase)):
            raise ValueError("Resampling method must be subclass of ResamplingMethodBase")

        self.under_sampling_methods[name] = resampling_method
    
    def add_target_size_strategy(self, name, target_size_strategy):
        """Add a resampling strategy.
        Will be called with {X_train, Y_train}
        
        Arguments:
            name {string} -- name of resampling strategy for definition in config
            resampling_strategy {function} -- callable with {pipeline_config, X_train, Y_train}
        """

        if (not issubclass(target_size_strategy, TargetSizeStrategyBase)):
            raise ValueError("Resampling method must be subclass of TargetSizeStrategyBase")

        self.target_size_strategies[name] = target_size_strategy

    def remove_over_sampling_method(self, name):
        del self.over_sampling_methods[name]

    def remove_under_sampling_method(self, name):
        del self.under_sampling_methods[name]

    def remove_target_size_strategy(self, name):
        del self.target_size_strategies[name]

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="over_sampling_methods", default=list(self.over_sampling_methods.keys()), type=str, list=True, choices=list(self.over_sampling_methods.keys())),
            ConfigOption(name="under_sampling_methods", default=list(self.under_sampling_methods.keys()), type=str, list=True, choices=list(self.under_sampling_methods.keys())),
            ConfigOption(name="target_size_strategies", default=list(self.target_size_strategies.keys()), type=str, list=True, choices=list(self.target_size_strategies.keys())),
        ]
        return options
    
    def get_hyperparameter_search_space(self, dataset_info=None, **pipeline_config):
        pipeline_config = self.pipeline.get_pipeline_config(**pipeline_config)
        cs = ConfigSpace.ConfigurationSpace()

        possible_over_sampling_methods = set(pipeline_config["over_sampling_methods"]).intersection(self.over_sampling_methods.keys())
        possible_under_sampling_methods = set(pipeline_config["under_sampling_methods"]).intersection(self.under_sampling_methods.keys())
        possible_target_size_strategies = set(pipeline_config["target_size_strategies"]).intersection(self.target_size_strategies.keys())
        selector_over_sampling = cs.add_hyperparameter(CSH.CategoricalHyperparameter("over_sampling_method", sorted(possible_over_sampling_methods)))
        selector_under_sampling = cs.add_hyperparameter(CSH.CategoricalHyperparameter("under_sampling_method", sorted(possible_under_sampling_methods)))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter("target_size_strategy", sorted(possible_target_size_strategies)))

        for method_name, method_type in self.over_sampling_methods.items():
            if method_name not in possible_over_sampling_methods:
                continue
            method_cs = method_type.get_hyperparameter_search_space(
                **self._get_search_space_updates(prefix=method_name))
            cs.add_configuration_space( prefix=method_name, configuration_space=method_cs, delimiter=ConfigWrapper.delimiter, 
                                        parent_hyperparameter={'parent': selector_over_sampling, 'value': method_name})
        
        for method_name, method_type in self.under_sampling_methods.items():
            if method_name not in possible_under_sampling_methods:
                continue
            method_cs = method_type.get_hyperparameter_search_space(
                **self._get_search_space_updates(prefix=method_name))
            cs.add_configuration_space( prefix=method_name, configuration_space=method_cs, delimiter=ConfigWrapper.delimiter, 
                                        parent_hyperparameter={'parent': selector_under_sampling, 'value': method_name})

        self._check_search_space_updates((possible_over_sampling_methods, "*"),
                                         (possible_under_sampling_methods, "*"))
        return cs
