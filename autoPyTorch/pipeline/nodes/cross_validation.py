__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
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

from sklearn.model_selection import StratifiedKFold
from autoPyTorch.pipeline.base.sub_pipeline_node import SubPipelineNode
from autoPyTorch.pipeline.base.pipeline import Pipeline

from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from autoPyTorch.training.budget_types import BudgetTypeTime

import time

class CrossValidation(SubPipelineNode):
    def __init__(self, train_pipeline_nodes):
        """CrossValidation pipeline node.
        It will run the train_pipeline by providing different train and validation datasets given the cv_split value defined in the config.
        Cross validation can be disabled by setting cv_splits to <= 1 in the config
        This enables the validation_split config parameter which, if no validation data is provided, will split the train dataset according its value (percent of train dataset)

        Train:
        The train_pipeline will receive the following inputs:
        {hyperparameter_config, pipeline_config, X_train, Y_train, X_valid, Y_valid, budget, training_techniques, fit_start_time, categorical_features}

        Prediction:
        The train_pipeline will receive the following inputs:
        {pipeline_config, X}
        
        Arguments:
            train_pipeline {Pipeline} -- training pipeline that will be computed cv_split times
            train_result_node {PipelineNode} -- pipeline node that provides the results of the train_pipeline
        """

        super(CrossValidation, self).__init__(train_pipeline_nodes)

        self.use_stratified_cv_split_default = False
        self.logger = logging.getLogger('autonet')
        

    def fit(self, hyperparameter_config, pipeline_config, X_train, Y_train, X_valid, Y_valid, budget, budget_type, optimize_start_time):
        
        cv_splits = max(1, pipeline_config['cv_splits'])
        val_split = max(0, min(1, pipeline_config['validation_split']))

        budget_too_low_for_cv, cv_splits, loss_penalty = self.incorporate_num_cv_splits_in_budget(budget, pipeline_config, cv_splits)
       
        self.logger.debug("Took " + str(time.time() - optimize_start_time) + " s to initialize optimization.")

        loss = 0
        infos = []

        split_indices = None
        if (cv_splits > 1 and pipeline_config['use_stratified_cv_split']):
            assert len(Y_train.shape) == 1 or Y_train.shape[1] == 1, "Y is in wrong shape for stratified CV split"
            skf = StratifiedKFold(n_splits=cv_splits, shuffle=False)
            split_indices = list(skf.split(X_train, Y_train.reshape((-1, ))))

        if 'categorical_features' in pipeline_config and pipeline_config['categorical_features']:
            categorical_features = pipeline_config['categorical_features']
        else:
            categorical_features = [False] * X_train.shape[1]
        

        if budget_type == BudgetTypeTime:
            cv_start_time = time.time()
            budget = budget - (cv_start_time - optimize_start_time)

        for i in range(cv_splits):
            
            self.logger.debug("[AutoNet] CV split " + str(i))

            x_train, y_train, x_valid, y_valid = self.split_data(X_train, Y_train, X_valid, Y_valid, i, cv_splits, val_split, split_indices)

            if budget_too_low_for_cv:
                cv_splits = 1

            if budget_type == BudgetTypeTime:
                remaining_budget = budget - (time.time() - cv_start_time)
                should_be_remaining_budget = (budget - i * budget / cv_splits)
                budget_type.compensate = max(10, should_be_remaining_budget - remaining_budget)
                cur_budget = remaining_budget / (cv_splits - i)
                self.logger.info("Reduced initial budget " + str(budget/cv_splits) + " to cv budget " + str(cur_budget) + " compensate for " + str(should_be_remaining_budget - remaining_budget))
            else:
                cur_budget = budget/cv_splits

            result = self.sub_pipeline.fit_pipeline(
                hyperparameter_config=hyperparameter_config, pipeline_config=pipeline_config, 
                X_train=x_train, Y_train=y_train, X_valid=x_valid, Y_valid=y_valid, 
                budget=cur_budget, training_techniques=[budget_type()],
                fit_start_time=time.time(),
                categorical_features=categorical_features)

            if result is not None:
                loss += result['loss']
                infos.append(result['info'])

            if budget_too_low_for_cv:
                break

        if (len(infos) == 0):
            raise Exception("Could not finish a single cv split due to memory or time limitation")

        df = pd.DataFrame(infos)
        info = dict(df.mean())

        loss = loss / cv_splits + loss_penalty

        return {'loss': loss, 'info': info}

    def predict(self, pipeline_config, X):
       
        result = self.sub_pipeline.predict_pipeline(pipeline_config=pipeline_config, X=X)

        return {'Y': result['Y']}

    def get_pipeline_config_options(self):
        options = [
            ConfigOption("validation_split", default=0.0, type=float, choices=[0, 1],
                info='In range [0, 1). Part of train dataset used for validation. Ignored in fit if cv_splits > 1 or valid data given.'),
            ConfigOption("cv_splits", default=1, type=int, info='The number of CV splits.'),
            ConfigOption("use_stratified_cv_split", default=self.use_stratified_cv_split_default, type=to_bool, choices=[True, False]),
            ConfigOption("min_budget_for_cv", default=0, type=float,
                info='Specify minimum budget for cv. If budget is smaller only evaluate a single fold.'),
            ConfigOption("half_num_cv_splits_below_budget", default=0, type=float,
                info='Incorporate number of cv splits in budget: Use half the number of specified cv splits below given budget.')
        ]
        return options


    def split_data(self, X_train, Y_train, X_valid, Y_valid, cv_split, max_cv_splits, val_split, split_indices):
        if (split_indices):
            train_indices = split_indices[cv_split][0]
            valid_indices = split_indices[cv_split][1]
            return X_train[train_indices], Y_train[train_indices], X_train[valid_indices], Y_train[valid_indices]

        if (max_cv_splits > 1):
            if (X_valid is not None or Y_valid is not None):
                self.logger.warning("[AutoNet] Cross validation is enabled but a validation set is provided - continue by ignoring validation set")
            return self.split_cv(X_train, Y_train, cv_split, max_cv_splits)

        if (val_split > 0.0):
            if (X_valid is None and Y_valid is None):
                return self.split_val(X_train, Y_train, val_split)
            else:
                self.logger.warning("[AutoNet] Validation split is not 0 and a validation set is provided - continue with provided validation set")

        return X_train, Y_train, X_valid, Y_valid

    def split_val(self, X_train_full, Y_train_full, percent):
        split_size = X_train_full.shape[0] * (1 - percent)
        i1 = int(split_size)
        X_train = X_train_full[0:i1]
        Y_train = Y_train_full[0:i1]

        X_valid = X_train_full[i1:]
        Y_valid = Y_train_full[i1:]

        return X_train, Y_train, X_valid, Y_valid

    def split_cv(self, X_train_full, Y_train_full, split, max_splits):
        split_size = X_train_full.shape[0] / max_splits
        i1 = int(split*split_size)
        i2 = int((split+1)*split_size)

        X_train = self.concat(X_train_full[0:i1], X_train_full[i2:])
        Y_train = self.concat(Y_train_full[0:i1], Y_train_full[i2:])

        X_valid = X_train_full[i1:i2]
        Y_valid = Y_train_full[i1:i2]

        if (X_valid.shape[0] + X_train.shape[0] != X_train_full.shape[0]):
            raise ValueError("Error while splitting data, " + str(X_train_full.shape) + " -> " + str(X_valid.shape) + " and " + str(X_train.shape))

        return X_train, Y_train, X_valid, Y_valid

    def concat(self, upper, lower):
        if (scipy.sparse.issparse(upper)):
            return scipy.sparse.vstack([upper, lower])
        else:
            return np.concatenate([upper, lower])

    def clean_fit_data(self):
        super(CrossValidation, self).clean_fit_data()
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