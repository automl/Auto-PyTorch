__author__ = "Michael Burkart"
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
        self.is_sparse = False
        self.default_dataset = None # could be set to CIFAR to download official CIFAR dataset from pytorch

class CreateDatasetInfo(PipelineNode):

    default_datasets = {
        # NAME       # dataset              # shape                 # classes
        'CIFAR10' :  (datasets.CIFAR10,     [50000, 3, 32, 32],     10),
        'CIFAR100' : (datasets.CIFAR100,    [50000, 3, 32, 32],     10),
        'SVHN' :     (datasets.SVHN,        [70000, 3, 32, 32],     10),
        'MNIST' :    (datasets.MNIST,       [60000, 28, 28],        10),
    }


    def fit(self, pipeline_config, X_train, Y_train, X_valid, Y_valid, dataset_path):
        info = DataSetInfo()
        info.is_sparse = scipy.sparse.issparse(X_train)
        info.path = dataset_path

        if X_train[0] in self.default_datasets:
            dataset_type, shape, classes = self.default_datasets[X_train[0]]
            info.default_dataset = dataset_type
            info.x_shape = shape
            info.y_shape = [shape[0], classes]
            X_train = np.array([X_train[0]])
            Y_train = np.array([])

        elif len(X_train.shape) == 1:
            if 'max_class_size' not in pipeline_config.keys():
                pipeline_config['max_class_size'] = None # backwards compatibility
            
            if "file_extensions" not in pipeline_config.keys():
                pipeline_config['file_extensions'] = ['.png', '.jpg', '.JPEG', '.pgm']

            X_train, Y_train = self.add_subpaths(X_train, Y_train, 
                pipeline_config['images_root_folders'], pipeline_config['file_extensions'], pipeline_config['max_class_size'] or float('inf'))
            X_valid, Y_valid = self.add_subpaths(X_valid, Y_valid, 
                pipeline_config['images_root_folders'], pipeline_config['file_extensions'], pipeline_config['max_class_size'] or float('inf'))

            info.x_shape = [X_train.shape[0]] + pipeline_config['images_shape']
            info.y_shape = Y_train.shape
            
            if len(info.y_shape) == 1 or info.y_shape[1] == 1:
                info.y_shape = (info.y_shape[0], len(np.unique(Y_train)))
        else:
            info.x_shape = X_train.shape
            info.y_shape = Y_train.shape

        return {'X_train' : X_train, 'Y_train' : Y_train, 'X_valid' : X_valid, 'Y_valid' : Y_valid, 'dataset_info' : info}
        

    def predict(self, pipeline_config, X):
        fit_res = self.fit(pipeline_config, X, np.zeros(X.shape[0]), None, None, pipeline_config)
        return { 'X': fit_res['X_train'], 'dataset_info': fit_res['dataset_info'] }

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="file_extensions", default=['.png', '.jpg', '.JPEG', '.pgm'], type=str, list=True),
            ConfigOption(name="images_shape", default=[3, 32, 32], type=int, list=True),
            ConfigOption(name="images_root_folders", default=[ConfigFileParser.get_autonet_home()], type='directory', list=True),
            ConfigOption(name="max_class_size", default=None, type=int),
        ]
        return options

    def add_subpaths(self, X, Y, root_folders, extensions, max_class_size):
        if X is None or Y is None:
            return None, None

        new_X, new_Y = [], []
        #for i, path in enumerate(X):
        #    for root in root_folders:
        #        tmp = os.path.join(root, path)
        #        if os.path.exists(tmp):
        #            path = tmp
        #            break
        #    if "."+path.split(".")[1] in extensions:
        #        new_X.append(X)
        #        new_Y = Y
        #        continue
        #    if not os.path.exists(path):
        #        print(path)
        #        raise Exception('Invalid path: ' + str(root_folders) + str(path))
        #    if os.path.isfile(path) and os.path.splitext(path)[1] == '.h5':
        #        import h5py
        #        return h5py.File(path, 'r')['x'].value, h5py.File(os.path.join(root, Y[i]), 'r')['y'].value.squeeze()
        #    self.add_path(path, Y[i], new_X, new_Y, extensions, max_class_size)

        for i, path in enumerate(X):
            for root in root_folders:
                tmp = os.path.join(root, path)
                if os.path.exists(tmp):
                    path = tmp
                    break
            if not os.path.exists(path):
                raise Exception('Invalid path: ' + str(root_folders) + str(path))
            if os.path.isfile(path) and os.path.splitext(path)[1] == '.h5':
                import h5py
                return h5py.File(path, 'r')['x'].value, h5py.File(os.path.join(root, Y[i]), 'r')['y'].value.squeeze()
            self.add_path(path, Y[i], new_X, new_Y, extensions, max_class_size)

        if len(new_X) == 0:
            raise Exception('Could not find any images in ' + str(root_folders) + '...' + str(extensions))
        return np.array(new_X), np.array(new_Y)

    def add_path(self, cur_X, cur_Y, new_X, new_Y, extensions, max_class_size):
        is_file, max_class_size = self.add_file(cur_X, cur_Y, new_X, new_Y, extensions, max_class_size)
        if is_file:
            return

        for sub in os.listdir(cur_X):
            if max_class_size <= 0:
                return max_class_size
            path = os.path.join(cur_X, sub)
            is_file, max_class_size = self.add_file(path, cur_Y, new_X, new_Y, extensions, max_class_size)

            if not is_file:
                max_class_size = self.add_path(path, cur_Y, new_X, new_Y, extensions, max_class_size)
                
    def add_file(self, cur_X, cur_Y, new_X, new_Y, extensions, max_class_size):
        if not os.path.isfile(cur_X):
            return False, max_class_size
        if not os.path.splitext(cur_X)[1] in extensions:
            return True, max_class_size
        if os.path.getsize(cur_X) > 0:
            new_X.append(cur_X)
            new_Y.append(cur_Y)
            max_class_size -= 1
            return True, max_class_size - 1
        else:
            import logging
            logging.getLogger('autonet').debug('Image is invalid! - size == 0:' + str(cur_X))
        return True, max_class_size
        
