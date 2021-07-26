__author__ = "Michael Burkart"
__version__ = "0.0.1"
__license__ = "BSD"

import torch
import logging
import scipy.sparse
import numpy as np
import pandas as pd
import signal
import time
import math
import copy

from sklearn.model_selection import StratifiedKFold
from autoPyTorch.pipeline.base.sub_pipeline_node import SubPipelineNode
from autoPyTorch.pipeline.base.pipeline import Pipeline

from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.training.budget_types import BudgetTypeTime
from sklearn.model_selection import StratifiedShuffleSplit

import time

class CrossValidationIndices(SubPipelineNode):
    def __init__(self, train_pipeline_nodes):
        """CrossValidation pipeline node.
        It will run the train_pipeline by providing different train and validation datasets given the cv_split value defined in the config.
        Cross validation can be disabled by setting cv_splits to <= 1 in the config
        This enables the validation_split config parameter which, if no validation data is provided, will split the train dataset according its value (percent of train dataset)

        Train:
        The train_pipeline will receive the following inputs:
        {hyperparameter_config, pipeline_config, X, Y, train_sampler, valid_sampler, budget, training_techniques, fit_start_time, categorical_features}

        Prediction:
        The train_pipeline will receive the following inputs:
        {pipeline_config, X}
        
        Arguments:
            train_pipeline {Pipeline} -- training pipeline that will be computed cv_split times
            train_result_node {PipelineNode} -- pipeline node that provides the results of the train_pipeline
        """

        super(CrossValidationIndices, self).__init__(train_pipeline_nodes)

        self.use_stratified_cv_split_default = False
        self.logger = logging.getLogger('autonet')
        

    def fit(self, hyperparameter_config, pipeline_config, X_train, Y_train, X_valid, Y_valid, budget, budget_type, dataset_info, config_id, working_directory):
        
        cv_splits = max(1, pipeline_config['cv_splits'])
        val_split = max(0, min(1, pipeline_config['validation_split']))

        budget_too_low_for_cv, cv_splits, loss_penalty = self.incorporate_num_cv_splits_in_budget(budget, pipeline_config, cv_splits)

        loss = 0
        infos = []

        np.random.seed(pipeline_config['random_seed'])

        split_indices = []
        X = X_train
        Y = Y_train

        if X_valid is not None and Y_valid is not None:
            if cv_splits > 1:
                self.logger.warning('CV splits are set to ' + str(cv_splits) + ' and validation set is specified, autonet will ignore cv splits and evaluate on given validation set')
            if val_split > 0.0:
                self.logger.warning('Validation split is set to ' + str(val_split) + ' and validation set is specified, autonet will ignore split and evaluate on given validation set')
            
            train_indices = self.shuffle_indices(list(range(X_train.shape[0])), pipeline_config['shuffle'])
            valid_indices = self.shuffle_indices(list(range(X_train.shape[0], X_train.shape[0] + X_valid.shape[0])), pipeline_config['shuffle'])

            X = self.concat(X_train, X_valid)
            Y = self.concat(Y_train, Y_valid)

            split_indices.append([train_indices, valid_indices])

        elif cv_splits > 1:
            if val_split > 0.0:
                self.logger.warning('Validation split is set to ' + str(val_split) + ' and cv splits are specified, autonet will ignore validation split and evaluate on ' + str(cv_splits) + ' cv splits')
            
            if pipeline_config['use_stratified_cv_split'] and Y.shape[0] == dataset_info.x_shape[0]:
                assert len(Y.shape) == 1 or Y.shape[1] == 1, "Y is in wrong shape for stratified CV split"
                skf = StratifiedKFold(n_splits=cv_splits, shuffle=pipeline_config['shuffle'])
                split_indices = list(skf.split(np.zeros(dataset_info.x_shape[0]), Y.reshape((-1, ))))
            else:
                indices = self.shuffle_indices(list(range(dataset_info.x_shape[0])), pipeline_config['shuffle'])
                split_size = len(indices) / cv_splits
                for split in range(cv_splits):
                    i1 = int(split*split_size)
                    i2 = int((split+1)*split_size)
                    train_indices, valid_indices = indices[:i1] + indices[i2:], indices[i1:i2]
                    split_indices.append([train_indices, valid_indices])

        elif val_split > 0.0:
            if pipeline_config['use_stratified_cv_split'] and Y.shape[0] == dataset_info.x_shape[0] and (len(Y.shape) == 1 or Y.shape[1] == 1):
                sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=pipeline_config['random_seed'])
                train, valid = list(sss.split(np.zeros(dataset_info.x_shape[0]), Y.reshape((-1, ))))[0]
                split_indices.append([train.tolist(), valid.tolist()])
                
                # samples = dataset_info.x_shape[0]
                # skf = StratifiedKFold(n_splits=math.ceil(samples / (samples * val_split)), shuffle=pipeline_config['shuffle'])
                # split_indices = [list(skf.split(np.zeros(dataset_info.x_shape[0]), Y.reshape((-1, ))))[0]]
            else:
                indices = self.shuffle_indices(list(range(dataset_info.x_shape[0])), pipeline_config['shuffle'])
                split = int(len(indices) * (1-val_split))

                train_indices, valid_indices = indices[:split], indices[split:]
                split_indices.append([train_indices, valid_indices])
        else:
            train_indices = self.shuffle_indices(list(range(dataset_info.x_shape[0])), pipeline_config['shuffle'])
            split_indices.append([train_indices, []])




        if 'categorical_features' in pipeline_config and pipeline_config['categorical_features']:
            categorical_features = pipeline_config['categorical_features']
        else:
            categorical_features = [False] * dataset_info.x_shape[1]
        
        for i, split in enumerate(split_indices):
            
            self.logger.debug("CV split " + str(i))

            train_indices = split[0]
            valid_indices = split[1] if len(split[1]) > 0 else None

            if budget_too_low_for_cv:
                cv_splits = 1

            cur_budget = budget/cv_splits

            result = self.sub_pipeline.fit_pipeline(
                hyperparameter_config=hyperparameter_config, pipeline_config=pipeline_config, 
                X=X, Y=Y, dataset_info=dataset_info,
                train_indices=train_indices, valid_indices=valid_indices, 
                budget=cur_budget, budget_type=budget_type,
                categorical_features=categorical_features,
                config_id=config_id,
                working_directory=working_directory)

            if result is not None:
                loss += result['loss']
                infos.append(result['info'])

            if budget_too_low_for_cv:
                break

        if (len(infos) == 0):
            raise Exception("Could not finish a single cv split due to memory or time limitation")

        if len(infos) == 1:
            info = infos[0]
        else:
            df = pd.DataFrame(infos)
            info = dict(df.mean())

        loss = loss / cv_splits + loss_penalty

        return {'loss': loss, 'info': info}

    def predict(self, pipeline_config, X, dataset_info):
        return self.sub_pipeline.predict_pipeline(pipeline_config=pipeline_config, X=X, dataset_info=dataset_info)

    def get_pipeline_config_options(self):
        options = [
            # percent/100 of train dataset used for validation if no validation and cv_splits == 1
            ConfigOption("validation_split", default=0.0, type=float, choices=[0, 1]),
            # number of cross validation splits 1 -> no cv
            ConfigOption("cv_splits", default=1, type=int),
            ConfigOption("use_stratified_cv_split", default=self.use_stratified_cv_split_default, type=to_bool, choices=[True, False]),
            # specify minimum budget for cv. If budget is smaller only evaluate a single fold.
            ConfigOption("min_budget_for_cv", default=0, type=float),
            # incorporate number of cv splits in budget: Use half the number of specified cv splits below given budget.
            ConfigOption("half_num_cv_splits_below_budget", default=0, type=float),
            # shuffle train and validation set
            ConfigOption('shuffle', default=True, type=to_bool, choices=[True, False]),
        ]
        return options

    def split_cv(self, X_shape, split, max_splits):
        split_size = X_shape[0] / max_splits
        i1 = int(split*split_size)
        i2 = int((split+1)*split_size)

        train_indices = list(range(0, i1)) + list(range(i2, X_shape[0]))
        valid_indices = list(range(i1, i2))

        return train_indices, valid_indices

    def concat(self, upper, lower):
        if (scipy.sparse.issparse(upper)):
            return scipy.sparse.vstack([upper, lower])
        else:
            return np.concatenate([upper, lower])

    def shuffle_indices(self, indices, shuffle):
        if shuffle:
            np.random.shuffle(indices)
        return indices


    def clean_fit_data(self):
        super(CrossValidationIndices, self).clean_fit_data()
        self.sub_pipeline.root.clean_fit_data()
    
    def incorporate_num_cv_splits_in_budget(self, budget, pipeline_config, cv_splits):
        budget_too_low_for_cv = budget < pipeline_config["min_budget_for_cv"] and cv_splits > 1
        half_num_cv_splits = not budget_too_low_for_cv and budget < pipeline_config["half_num_cv_splits_below_budget"] and cv_splits > 1

        if budget_too_low_for_cv:
            self.logger.debug("Only evaluate a single fold of CV, since the budget is lower than the min_budget for cv")
            return True, cv_splits, 1000
        
        if half_num_cv_splits:
            self.logger.debug("Using half number of cv splits since budget is lower than the budget you specified for half number of cv splits")
            return False, int(math.ceil(cv_splits / 2)), 1000

        return False, cv_splits, 0
