__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption
from autoPyTorch.components.preprocessing.preprocessor_base import PreprocessorBase
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
import ConfigSpace
import ConfigSpace.hyperparameters as CSH
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class NormalizationStrategySelector(PipelineNode):
    def __init__(self):
        super(NormalizationStrategySelector, self).__init__()

        self.normalization_strategies = {'none': None}

    def fit(self, hyperparameter_config, X_train, X_valid, categorical_features):
        hyperparameter_config = ConfigWrapper(self.get_name(), hyperparameter_config)

        normalizer_name = hyperparameter_config['normalization_strategy']

        if normalizer_name == 'none':
            return {'normalizer': None}

        normalizer = self.normalization_strategies[normalizer_name]()

        transformer = ColumnTransformer(
            transformers=[("normalize", normalizer, [i for i, c in enumerate(categorical_features) if not c])],
            remainder='passthrough'
        )
        transformer.fit(X_train)

        X_train = transformer.transform(X_train)
        if (X_valid is not None):
            X_valid = transformer.transform(X_valid)
        
        categorical_features = sorted(categorical_features)

        return {'X_train': X_train, 'X_valid': X_valid, 'normalizer': transformer, 'categorical_features': categorical_features}

    def predict(self, X, normalizer):
        if normalizer is None:
            return {'X': X}
        return {'X': normalizer.transform(X)}

    def add_normalization_strategy(self, name, normalization_type, is_default_normalization_strategy=False):
        """Add a normalization strategy.
        Will be called with {pipeline_config, X, Y}
        
        Arguments:
            name {string} -- name of normalization strategy for definition in config
            normalization_strategy {function} -- callable with {pipeline_config, X}
            is_default_normalization_strategy {bool} -- should the given normalization_strategy be the default normalization_strategy if not specified in config
        """

        if (not issubclass(normalization_type, BaseEstimator) and not issubclass(normalization_type, TransformerMixin)):
            raise ValueError("normalization_type must be subclass of BaseEstimator")
        self.normalization_strategies[name] = normalization_type

    def remove_normalization_strategy(self, name):
        del self.normalization_strategies[name]

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="normalization_strategies", default=list(self.normalization_strategies.keys()), type=str, list=True, choices=list(self.normalization_strategies.keys())),
        ]
        return options
    
    def get_hyperparameter_search_space(self, **pipeline_config):
        pipeline_config = self.pipeline.get_pipeline_config(**pipeline_config)
        cs = ConfigSpace.ConfigurationSpace()

        possible_normalization_strategies = set(pipeline_config["normalization_strategies"]).intersection(self.normalization_strategies.keys())
        cs.add_hyperparameter(CSH.CategoricalHyperparameter("normalization_strategy", possible_normalization_strategies))

        return self._apply_user_updates(cs)