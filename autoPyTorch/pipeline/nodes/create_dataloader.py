__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import inspect
import logging
import numpy as np

from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper

import torch
import scipy.sparse
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import SubsetRandomSampler


class CreateDataLoader(PipelineNode):

    def fit(self, pipeline_config, hyperparameter_config, X, Y, train_indices, valid_indices):
    
        torch.manual_seed(pipeline_config["random_seed"])
        hyperparameter_config = ConfigWrapper(self.get_name(), hyperparameter_config)

        # prepare data
        drop_last = hyperparameter_config['batch_size'] < train_indices.shape[0]
        X, Y = to_dense(X), to_dense(Y)
        X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y)

        train_dataset = TensorDataset(X, Y)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=hyperparameter_config['batch_size'], 
            sampler=SubsetRandomSampler(train_indices),
            shuffle=False,
            drop_last=drop_last)
            
        valid_loader = None
        if valid_indices is not None:
            valid_loader = DataLoader(
                dataset=Subset(train_dataset, valid_indices),
                batch_size=hyperparameter_config['batch_size'],
                shuffle=False,
                drop_last=False)

        return {'train_loader': train_loader, 'valid_loader': valid_loader, 'batch_size': hyperparameter_config['batch_size']}

    def predict(self, pipeline_config, X, batch_size):
        X = torch.from_numpy(to_dense(X)).float()
        y_placeholder = torch.Tensor(X.size()[0])

        predict_loader = DataLoader(TensorDataset(X.float(), y_placeholder), batch_size)

        return {'predict_loader': predict_loader}

    def get_hyperparameter_search_space(self, dataset_info=None, **pipeline_config):
        import ConfigSpace
        import ConfigSpace.hyperparameters as CSH

        pipeline_config = self.pipeline.get_pipeline_config(**pipeline_config)
        cs = ConfigSpace.ConfigurationSpace()

        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('batch_size', lower=32, upper=500, log=True))
        return self._apply_user_updates(cs)

    
def to_dense(matrix):
    if (matrix is not None and scipy.sparse.issparse(matrix)):
        return matrix.todense()
    return matrix
