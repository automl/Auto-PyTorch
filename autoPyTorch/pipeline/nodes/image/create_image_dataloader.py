__author__ = "Michael Burkart"
__version__ = "0.0.1"
__license__ = "BSD"

import inspect
import logging
import numpy as np

from autoPyTorch.pipeline.nodes.create_dataloader import CreateDataLoader
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser
from autoPyTorch.utils.config.config_option import ConfigOption

import torch
import scipy.sparse
from torch.utils.data import DataLoader, TensorDataset, Dataset
from autoPyTorch.data_management.data_loader import DataPrefetchLoader
from autoPyTorch.data_management.image_loader import ImageFilelist, XYDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, models, transforms

class CreateImageDataLoader(CreateDataLoader):

    def fit(self, pipeline_config, hyperparameter_config, X, Y, train_indices, valid_indices, train_transform, valid_transform, dataset_info):
        
        # if len(X.shape) > 1:
        #     return super(CreateImageDataLoader, self).fit(pipeline_config, hyperparameter_config, X, Y, train_indices, valid_indices)

        torch.manual_seed(pipeline_config["random_seed"])
        hyperparameter_config = ConfigWrapper(self.get_name(), hyperparameter_config)

        if dataset_info.default_dataset:
            train_dataset = dataset_info.default_dataset(root=pipeline_config['default_dataset_download_dir'], train=True, download=True, transform=train_transform)
            if valid_indices is not None:
                valid_dataset = dataset_info.default_dataset(root=pipeline_config['default_dataset_download_dir'], train=True, download=True, transform=valid_transform)
        elif len(X.shape) > 1:
            train_dataset = XYDataset(X, Y, transform=train_transform, target_transform=lambda y: y.astype(np.int64))
            valid_dataset = XYDataset(X, Y, transform=valid_transform, target_transform=lambda y: y.astype(np.int64))
        else:
            train_dataset = ImageFilelist(X, Y, transform=train_transform, target_transform=lambda y: y.astype(np.int64), cache_size=pipeline_config['dataloader_cache_size_mb'] * 1000, image_size=dataset_info.x_shape[2:])
            if valid_indices is not None:
                valid_dataset = ImageFilelist(X, Y, transform=valid_transform, target_transform=lambda y: y.astype(np.int64), cache_size=0, image_size=dataset_info.x_shape[2:])
                valid_dataset.cache = train_dataset.cache

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=int(hyperparameter_config['batch_size']),
            sampler=SubsetRandomSampler(train_indices),
            drop_last=True,
            pin_memory=True,
            num_workers=pipeline_config['dataloader_worker'])

        valid_loader = None
        if valid_indices is not None:
            valid_loader = DataLoader(
                dataset=valid_dataset,
                batch_size=int(hyperparameter_config['batch_size']),
                sampler=SubsetRandomSampler(valid_indices),
                drop_last=False,
                pin_memory=True,
                num_workers=pipeline_config['dataloader_worker'])
        
        if pipeline_config['prefetch']:
            train_loader = DataPrefetchLoader(train_loader)
            if valid_loader is not None:
                valid_loader = DataPrefetchLoader(valid_loader)
        
        return {'train_loader': train_loader, 'valid_loader': valid_loader, 'batch_size': hyperparameter_config['batch_size']}

    def get_pipeline_config_options(self):
        options = [
            ConfigOption("default_dataset_download_dir", default=ConfigFileParser.get_autonet_home(), type='directory', info="Directory default datasets will be downloaded to."),
            ConfigOption("dataloader_worker", default=1, type=int),
            ConfigOption("dataloader_cache_size_mb", default=0, type=int),
            ConfigOption("prefetch", default=False, type=bool)
        ]
        return options

    def predict(self, pipeline_config, X, batch_size, predict_transform, dataset_info):

        if len(X.shape) > 1:
            return super(CreateImageDataLoader, self).predict(pipeline_config, X, batch_size)


        if dataset_info.default_dataset:
            predict_dataset = dataset_info.default_dataset(root=pipeline_config['default_dataset_download_dir'], train=False, download=True, transform=predict_transform)
        else:
            try:
                y_placeholder = torch.zeros(X.size()[0])
            except:
                y_placeholder = torch.zeros(len(X))
            predict_dataset = ImageFilelist(X, y_placeholder, transform=predict_transform)
        
        predict_loader = DataLoader(
            dataset=predict_dataset,
            batch_size=int(batch_size),
            shuffle=False,
            pin_memory=True,
            num_workers=pipeline_config['dataloader_worker'])
        if pipeline_config['prefetch']:
            predict_loader = DataPrefetchLoader(predict_loader)
        return {'predict_loader': predict_loader}


    
