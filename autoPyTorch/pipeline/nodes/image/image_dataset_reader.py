__author__ = "Michael Burkart"
__version__ = "0.0.1"
__license__ = "BSD"

import os
import numpy as np
import math

from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser

from autoPyTorch.data_management.data_manager import ImageManager

class ImageDatasetReader(PipelineNode):
    def __init__(self):
        super(ImageDatasetReader, self).__init__()

    def fit(self, pipeline_config, X_train, Y_train, X_valid, Y_valid):

        if len(X_train.shape)==1 and len(X_train)==1:
            X_train = X_train[0]
            Y_train = 0

        if X_valid is not None:
            if len(X_valid.shape)==1 and len(X_valid)==1:
                X_valid = X_valid[0]
                Y_valid = None

        X_train, Y_train, path = self.read_data(X_train, Y_train)
        X_valid, Y_valid, _ = self.read_data(X_valid, Y_valid)

        return { 'X_train': X_train, 'Y_train': Y_train, 'X_valid': X_valid, 'Y_valid': Y_valid, 'dataset_path': path }

    def get_pipeline_config_options(self):
        options = [
        ]
        return options

    def read_data(self, path, y):
        if path is None:
            return None, None, None
        
        if not isinstance(path, str):
            return path, y, str(path)[0:300]
        
        if not os.path.isabs(path):
            path = os.path.abspath(os.path.join(ConfigFileParser.get_autonet_home(), path))

        if not os.path.exists(path):
            raise ValueError('Path ' + str(path) + ' is not a valid path.')

        im = ImageManager()
        im.read_data(path, is_classification=True)

        return im.X_train, im.Y_train, path
