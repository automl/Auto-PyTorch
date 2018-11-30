__author__ = "Michael Burkart"
__version__ = "0.0.1"
__license__ = "BSD"

import os
import numpy as np
import math
import time
import pandas as pd

from autoPyTorch.pipeline.base.sub_pipeline_node import SubPipelineNode

from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.utils.config.config_file_parser import ConfigFileParser

class MultipleDatasets(SubPipelineNode):



    def fit(self, hyperparameter_config, pipeline_config, X_train, Y_train, X_valid, Y_valid, budget, budget_type, max_budget, optimize_start_time):
        if len(X_train.shape) > 1:
            return self.sub_pipeline.fit_pipeline(  hyperparameter_config=hyperparameter_config, 
                                                    pipeline_config=pipeline_config, 
                                                    X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid, 
                                                    budget=budget, budget_type=budget_type, optimize_start_time=optimize_start_time)

        max_datasets = X_train.shape[0]
        n_datasets = math.ceil(budget * max_datasets/ max_budget)
        n_datasets = min(n_datasets, max_datasets) # this shouldnt be necessary but if we get some floating point rounding errors it avoids an exception

        if X_valid is None:
            X_valid = [None] * n_datasets
            Y_valid = [None] * n_datasets

        init_time = time.time() - optimize_start_time
        
        
        if 'use_tensorboard_logger' in pipeline_config and pipeline_config['use_tensorboard_logger']:
            import tensorboard_logger as tl
            worker_path = 'Worker_' + str(pipeline_config['task_id']) + '/'
            tl.log_value(worker_path + 'datasets', float(n_datasets), int(time.time()))

        infos = dict()
        loss = 0
        for dataset in range(n_datasets):
            result = self.sub_pipeline.fit_pipeline(hyperparameter_config=hyperparameter_config, 
                                                    pipeline_config=pipeline_config, 
                                                    X_train=X_train[dataset], Y_train=Y_train[dataset], X_valid=X_valid[dataset], Y_valid=Y_valid[dataset], 
                                                    budget=budget / n_datasets, budget_type=budget_type, optimize_start_time=time.time() - (init_time / n_datasets))
            infos[str(X_train[dataset])] = result['info']
            loss += result['loss']
            
        import logging
        logging.getLogger('autonet').info(str(infos))
        
        
        if 'use_tensorboard_logger' in pipeline_config and pipeline_config['use_tensorboard_logger']:
            import tensorboard_logger as tl
            worker_path = 'Worker_' + str(pipeline_config['task_id']) + '/'
            tl.log_value(worker_path + 'datasets', float(n_datasets), int(time.time()))

        # df = pd.DataFrame(infos)
        # info = dict(df.mean())
        loss = loss / n_datasets

        return {'loss': loss, 'info': infos}


    def get_pipeline_config_options(self):
        options = [
        ]
        return options
