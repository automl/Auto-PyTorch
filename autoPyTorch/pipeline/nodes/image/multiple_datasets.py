__author__ = "Michael Burkart"
__version__ = "0.0.1"
__license__ = "BSD"

import os
import numpy as np
import math
import time
import pandas as pd
import logging
import random
import torch

from autoPyTorch.pipeline.base.sub_pipeline_node import SubPipelineNode

from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser

class MultipleDatasets(SubPipelineNode):
    
    def __init__(self, sub_pipeline_nodes):
        super(MultipleDatasets, self).__init__(sub_pipeline_nodes)
        
        self.logger = logging.getLogger('autonet')


    def fit(self, hyperparameter_config, pipeline_config, X_train, Y_train, X_valid, Y_valid, budget, budget_type, config_id, working_directory):
        if len(X_train.shape) > 1:
            return self.sub_pipeline.fit_pipeline(  hyperparameter_config=hyperparameter_config, 
                                                    pipeline_config=pipeline_config, 
                                                    X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid, 
                                                    budget=budget, budget_type=budget_type, config_id=config_id, working_directory=working_directory)


        max_datasets = X_train.shape[0]
        max_steps = math.floor((math.log(pipeline_config['max_budget']) - math.log(pipeline_config['min_budget'])) / math.log(pipeline_config['eta']))
        current_step = max_steps - math.floor((math.log(pipeline_config['max_budget']) - math.log(budget)) / math.log(pipeline_config['eta'])) if budget > 1e-10 else 0
        n_datasets = math.floor(math.pow(max_datasets, current_step/max(1, max_steps)) + 1e-10)
        n_datasets = max(n_datasets,1)
        
        # refit can cause issues with different budget
        if max_steps == 0 or n_datasets > max_datasets or not pipeline_config['increase_number_of_trained_datasets']:
            n_datasets = max_datasets

        if X_valid is None or Y_valid is None:
            X_valid = [None] * n_datasets
            Y_valid = [None] * n_datasets
        
        if 'use_tensorboard_logger' in pipeline_config and pipeline_config['use_tensorboard_logger']:
            import tensorboard_logger as tl
            tl.log_value('Train/datasets', float(n_datasets), int(time.time()))

        infos = []
        loss = 0
        losses = []

        self.logger.debug('Start fitting ' + str(n_datasets) + ' dataset(s). Current budget: ' + str(budget) + ' - Step: ' + str(current_step) + '/' + str(max_steps))

        #dataset_order = list(range(n_datasets))
        #random.shuffle(dataset_order)
        #if pipeline_config['dataset_order'] and len(pipeline_config['dataset_order']) == n_datasets:
        #    dataset_order = pipeline_config['dataset_order']
        #    dataset_order = [i for i in dataset_order if i < n_datasets]
        #X_train = X_train[dataset_order]
        if np.any(pipeline_config['dataset_order']):
            dataset_order = pipeline_config['dataset_order']
        else:
            dataset_order = list(range(n_datasets))
        X_train = X_train[dataset_order]

        for dataset in range(n_datasets):
            self.logger.info('Fit dataset (' + str(dataset+1) + '/' + str(n_datasets) + '): ' + str(X_train[dataset]) + ' for ' + str(round(budget / n_datasets)) + 's')

            result = self.sub_pipeline.fit_pipeline(hyperparameter_config=hyperparameter_config, 
                                                    pipeline_config=pipeline_config, 
                                                    X_train=X_train[dataset], Y_train=Y_train[dataset], X_valid=X_valid[dataset], Y_valid=Y_valid[dataset], 
                                                    budget=budget / n_datasets, budget_type=budget_type, config_id=config_id, working_directory=working_directory)
            
            # copy/rename checkpoint - save one checkpoint for each trained dataset
            if 'checkpoint' in result['info']:
                src = result['info']['checkpoint']
                folder, file = os.path.split(src)
                dest = os.path.join(folder, os.path.splitext(file)[0] + '_' + str(dataset) + '.pt')
                import shutil
                if dataset < n_datasets - 1:
                    shutil.copy(src, dest)
                else:
                    os.rename(src, dest)
                result['info']['checkpoint'] = dest

            result['info']['dataset_path'] = str(X_train[dataset])
            result['info']['dataset_id'] = dataset_order[dataset]
            
            infos.append(result['info'])
            loss += result['loss']
            losses.append(result['loss'])
        
        if 'use_tensorboard_logger' in pipeline_config and pipeline_config['use_tensorboard_logger']:
            import tensorboard_logger as tl
            tl.log_value('Train/datasets', float(n_datasets), int(time.time()))

        loss = loss / n_datasets

        return {'loss': loss, 'losses': losses, 'info': infos}

    def predict(self, pipeline_config, X):
        return self.sub_pipeline.predict_pipeline(pipeline_config=pipeline_config, X=X)

    def get_pipeline_config_options(self):
        options = [
            ConfigOption('dataset_order', default=None, type=int, list=True),

            #autonet.refit sets this to false to avoid refit budget issues
            ConfigOption('increase_number_of_trained_datasets', default=True, type=to_bool) 
        ]
        return options
