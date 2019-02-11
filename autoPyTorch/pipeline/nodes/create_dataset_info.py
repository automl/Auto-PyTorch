__author__ = "Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import os
import numpy as np
import scipy.sparse
from torchvision import datasets

from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser


class DataSetInfo():
    def __init__(self):
        self.categorical_features = []
        self.x_shape = []
        self.y_shape = []
        self.x_min_value = None
        self.x_max_value = None
        self.is_sparse = False

class CreateDatasetInfo(PipelineNode):

    def fit(self, pipeline_config, X_train, Y_train, X_valid, Y_valid):
        info = DataSetInfo()
        info.is_sparse = scipy.sparse.issparse(X_train)

        info.x_shape = X_train.shape
        info.y_shape = Y_train.shape

        info.x_min_value = X_train.min()
        info.x_max_value = X_train.max()

        if 'categorical_features' in pipeline_config and pipeline_config['categorical_features']:
            info.categorical_features = pipeline_config['categorical_features']
        else:
            info.categorical_features = [False] * info.x_shape[1]

        return {'X_train' : X_train, 'Y_train' : Y_train, 'X_valid' : X_valid, 'Y_valid' : Y_valid, 'dataset_info' : info}
        

    def predict(self, pipeline_config, X_train, Y_train, X_valid, Y_valid):
        return self.fit(pipeline_config, X_train, Y_train, X_valid, Y_valid)

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name='categorical_features', default=None, type=to_bool, list=True,
                info='List of booleans that specifies for each feature whether it is categorical.')
        ]
        return options
            

