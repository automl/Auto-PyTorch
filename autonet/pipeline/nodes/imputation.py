import torch
import numpy as np
import scipy.sparse

import ConfigSpace
import ConfigSpace.hyperparameters as CSH
        
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from autonet.pipeline.base.pipeline_node import PipelineNode

from autonet.utils.config.config_option import ConfigOption
from autonet.utils.configspace_wrapper import ConfigWrapper

class Imputation(PipelineNode):

    strategies = ["mean", "median", "most_frequent"]

    def fit(self, hyperparameter_config, X_train, X_valid, categorical_features):
        hyperparameter_config = ConfigWrapper(self.get_name(), hyperparameter_config)

        strategy = hyperparameter_config['strategy']
        fill_value = int(np.nanmax(X_train)) + 1 if not scipy.sparse.issparse(X_train) else 0
        numerical_imputer = SimpleImputer(strategy=strategy, copy=False)
        categorical_imputer = SimpleImputer(strategy='constant', copy=False, fill_value=fill_value)
        transformer = ColumnTransformer(
            transformers=[('numerical_imputer', numerical_imputer, [i for i, c in enumerate(categorical_features) if not c]),
                          ('categorical_imputer', categorical_imputer,  [i for i, c in enumerate(categorical_features) if c])])
        transformer.fit(X_train)
        
        X_train = transformer.transform(X_train)
        if (X_valid is not None):
            X_valid = transformer.transform(X_valid)
        
        categorical_features = sorted(categorical_features)
        return { 'X_train': X_train, 'X_valid': X_valid, 'imputation_preprocessor': transformer, 'categorical_features': categorical_features }


    def predict(self, X, imputation_preprocessor):
        return { 'X': imputation_preprocessor.transform(X) }

    @staticmethod
    def get_hyperparameter_search_space(**pipeline_config):

        possible_strategies = set(Imputation.strategies).intersection(pipeline_config['imputation_strategies'])

        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameter(CSH.CategoricalHyperparameter("strategy", possible_strategies))
        return cs

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name='imputation_strategies', default=Imputation.strategies, type=str, list=True, choices=Imputation.strategies)
        ]
        return options
        