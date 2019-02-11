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
import inspect
from copy import deepcopy

from sklearn.model_selection import BaseCrossValidator
from autoPyTorch.pipeline.base.sub_pipeline_node import SubPipelineNode
from autoPyTorch.pipeline.base.pipeline import Pipeline

from autoPyTorch.utils.config.config_option import ConfigOption, to_bool, to_dict
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

        self.cross_validators = {'none': None}
        self.cross_validators_adjust_y = dict()


    def fit(self, hyperparameter_config, pipeline_config, X_train, Y_train, X_valid, Y_valid, budget, budget_type, optimize_start_time,
            refit, rescore, dataset_info):
        logger = logging.getLogger('autonet')
        loss = 0
        infos = []
        X, Y, num_cv_splits, cv_splits, loss_penalty, budget = self.initialize_cross_validation(
            pipeline_config=pipeline_config, budget=budget, X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid,
            dataset_info=dataset_info, refit=(refit and not rescore), logger=logger)
        
        # adjust budget in case of budget type time
        cv_start_time = time.time()
        if budget_type == BudgetTypeTime:
            budget = budget - (cv_start_time - optimize_start_time)

        # start cross validation
        logger.debug("Took " + str(time.time() - optimize_start_time) + " s to initialize optimization.")
        all_sub_pipeline_kwargs = dict()
        additional_results = dict()
        for i, split_indices in enumerate(cv_splits):
            logger.info("[AutoNet] CV split " + str(i) + " of " + str(num_cv_splits))

            # fit training pipeline
            cur_budget = self.get_current_budget(cv_index=i, budget=budget, budget_type=budget_type,
                cv_start_time=cv_start_time, num_cv_splits=num_cv_splits, logger=logger)
            sub_pipeline_kwargs = {
                "hyperparameter_config": hyperparameter_config, "pipeline_config": pipeline_config,
                "budget": cur_budget, "training_techniques": [budget_type()],
                "fit_start_time": time.time(),
                "train_indices": split_indices[0],
                "valid_indices": split_indices[1],
                "dataset_info": deepcopy(dataset_info),
                "refit": refit,
                "loss_penalty": loss_penalty}
            all_sub_pipeline_kwargs[i] = deepcopy(sub_pipeline_kwargs)
            result = self.sub_pipeline.fit_pipeline(X=X, Y=Y, **sub_pipeline_kwargs)
            logger.info("[AutoNet] Done with current split!")

            if result is not None:
                loss += result['loss']
                infos.append(result['info'])
                additional_results[i] = {key: value for key, value in result.items() if key not in ["loss", "info"]}

        if (len(infos) == 0):
            raise Exception("Could not finish a single cv split due to memory or time limitation")

        # aggregate logs
        logger.info("Aggregate the results across the splits")
        df = pd.DataFrame(infos)
        info = dict(df.mean())
        additional_results = self.process_additional_results(additional_results=additional_results, all_sub_pipeline_kwargs=all_sub_pipeline_kwargs,
            X=X, Y=Y, logger=logger)
        loss = loss / num_cv_splits + loss_penalty
        return dict({'loss': loss, 'info': info}, **additional_results)

    def predict(self, pipeline_config, X):
       
        result = self.sub_pipeline.predict_pipeline(pipeline_config=pipeline_config, X=X)

        return {'Y': result['Y']}

    def get_pipeline_config_options(self):
        options = [
            ConfigOption("validation_split", default=0.0, type=float, choices=[0, 1],
                info='In range [0, 1). Part of train dataset used for validation. Ignored in fit if cross validator or valid data given.'),
            ConfigOption("refit_validation_split", default=0.0, type=float, choices=[0, 1],
                info='In range [0, 1). Part of train dataset used for validation in refit.'),
            ConfigOption("cross_validator", default="none", type=str, choices=self.cross_validators.keys(),
                info='Class inheriting from sklearn.model_selection.BaseCrossValidator. Ignored if validation data is given.'),
            ConfigOption("cross_validator_args", default=dict(), type=to_dict,
                info="Args of cross validator. \n\t\tNote that random_state and shuffle are set by " +
                     "pipeline config options random_seed and shuffle, if not specified here."),
            ConfigOption("min_budget_for_cv", default=0, type=float,
                info='Specify minimum budget for cv. If budget is smaller use specified validation split.'),
            ConfigOption('shuffle', default=True, type=to_bool, choices=[True, False],
                info='Shuffle train and validation set'),
        ]
        return options

    def clean_fit_data(self):
        super(CrossValidation, self).clean_fit_data()
        self.sub_pipeline.root.clean_fit_data()
    
    def initialize_cross_validation(self, pipeline_config, budget, X_train, Y_train, X_valid, Y_valid, dataset_info, refit, logger):
        budget_too_low_for_cv = budget < pipeline_config['min_budget_for_cv']
        val_split = max(0, min(1, pipeline_config['validation_split']))
        if refit:
            val_split = max(0, min(1, pipeline_config['refit_validation_split']))

        # validation set given. cv ignored.
        if X_valid is not None and Y_valid is not None:
            if pipeline_config['cross_validator'] != "none":
                logger.warning('Cross validator ' + pipeline_config['cross_validator'] + ' given and validation set is specified, ' +
                                    'autonet will ignore cv splits and evaluate on given validation set')
            if val_split > 0.0:
                logger.warning('Validation split is set to ' + str(val_split) + ' and validation set is specified, ' +
                                    'autonet will ignore split and evaluate on given validation set')
            
            X, Y, indices = self.get_validation_set_split_indices(pipeline_config,
                X_train=X_train, X_valid=X_valid, Y_train=Y_train, Y_valid=Y_valid)

            logger.info("[AutoNet] Validation set given. Continue with validation set (no cross validation).")
            return X, Y, 1, [indices], 0, budget
        
        # no cv, split train data
        if pipeline_config['cross_validator'] == "none" or budget_too_low_for_cv:
            logger.info("[AutoNet] No validation set given and either no cross validator given or budget to low for CV." + 
                             " Continue by splitting " + str(val_split) + " of training data.")
            indices = self.shuffle_indices(np.array(list(range(dataset_info.x_shape[0]))), pipeline_config['shuffle'], pipeline_config["random_seed"])
            split = int(len(indices) * (1-val_split))
            train_indices, valid_indices = indices[:split], indices[split:]
            valid_indices = None if val_split == 0 else valid_indices
            return X_train, Y_train, 1, [(train_indices, valid_indices)], (1000 if budget_too_low_for_cv else 0), budget

        # cross validation
        logger.warning('Validation split is set to ' + str(val_split) + ' and cross validator specified, autonet will ignore validation split')
        cross_validator_class = self.cross_validators[pipeline_config['cross_validator']]
        adjust_y = self.cross_validators_adjust_y[pipeline_config['cross_validator']]
        available_cross_validator_args = inspect.getfullargspec(cross_validator_class.__init__)[0]
        cross_validator_args = pipeline_config['cross_validator_args']

        if "shuffle" not in cross_validator_args and "shuffle" in available_cross_validator_args:
            cross_validator_args["shuffle"] = pipeline_config["shuffle"]
        if "random_state" not in cross_validator_args and "random_state" in available_cross_validator_args:
            cross_validator_args["random_state"] = pipeline_config["random_seed"]

        cross_validator = cross_validator_class(**cross_validator_args)
        num_cv_splits = cross_validator.get_n_splits(X_train, adjust_y(Y_train))
        cv_splits = cross_validator.split(X_train, adjust_y(Y_train))
        if not refit:
            logger.info("[Autonet] Continue with cross validation using " + str(pipeline_config['cross_validator']))
            return X_train, Y_train, num_cv_splits, cv_splits, 0, budget
        
        # refit
        indices = self.shuffle_indices(np.array(list(range(dataset_info.x_shape[0]))), pipeline_config['shuffle'], pipeline_config["random_seed"])
        split = int(len(indices) * (1-val_split))
        train_indices, valid_indices = indices[:split], indices[split:]
        valid_indices = None if val_split == 0 else valid_indices
        logger.info("[Autonet] No cross validation when refitting! Continue by splitting " + str(val_split) + " of training data.")
        return X_train, Y_train, 1, [(train_indices, valid_indices)], 0, budget / num_cv_splits

    def add_cross_validator(self, name, cross_validator, adjust_y=None):
        self.cross_validators[name] = cross_validator
        self.cross_validators_adjust_y[name] = adjust_y if adjust_y is not None else identity
    
    def remove_cross_validator(self, name):
        del self.cross_validators[name]
        del self.cross_validators_adjust_y[name]
    
    def get_current_budget(self, cv_index, budget, budget_type, cv_start_time, num_cv_splits, logger):
        # adjust budget in case of budget type time
        if budget_type == BudgetTypeTime:
            remaining_budget = budget - (time.time() - cv_start_time)
            should_be_remaining_budget = (budget - cv_index * budget / num_cv_splits)
            budget_type.compensate = max(10, should_be_remaining_budget - remaining_budget)
            cur_budget = remaining_budget / (num_cv_splits - cv_index)
            logger.info("Reduced initial budget " + str(budget / num_cv_splits) + " to cv budget " + 
                                str(cur_budget) + " compensate for " + str(should_be_remaining_budget - remaining_budget))
        else:
            cur_budget = budget / num_cv_splits
        return cur_budget
    
    def process_additional_results(self, additional_results, all_sub_pipeline_kwargs, X, Y, logger):
        combinators = dict()
        data = dict()
        result = dict()
        logger.info("Process %s additional result(s)" % len(additional_results))
        for split in additional_results.keys():
            for name in additional_results[split].keys():
                combinators[name] = additional_results[split][name]["combinator"]
                if name not in data:
                    data[name] = dict()
                data[name][split] = additional_results[split][name]["data"]
        for name in data.keys():
            result[name] = combinators[name](data=data[name], pipeline_kwargs=all_sub_pipeline_kwargs, X=X, Y=Y)
        return result
    
    @staticmethod
    def concat(upper, lower):
        if (scipy.sparse.issparse(upper)):
            return scipy.sparse.vstack([upper, lower])
        else:
            return np.concatenate([upper, lower])

    @staticmethod
    def shuffle_indices(indices, shuffle=True, seed=42):
        rng = np.random.RandomState(42)
        if shuffle:
            rng.shuffle(indices)
        return indices
    
    @staticmethod
    def get_validation_set_split_indices(pipeline_config, X_train, X_valid, Y_train, Y_valid, allow_shuffle=True):
        train_indices = CrossValidation.shuffle_indices(np.array(list(range(X_train.shape[0]))),
            pipeline_config['shuffle'] and allow_shuffle, pipeline_config['random_seed'])
        valid_indices = CrossValidation.shuffle_indices(np.array(list(range(X_train.shape[0], X_train.shape[0] + X_valid.shape[0]))),
            pipeline_config['shuffle'] and allow_shuffle, pipeline_config['random_seed'])

        X = CrossValidation.concat(X_train, X_valid)
        Y = CrossValidation.concat(Y_train, Y_valid)
        return X, Y, (train_indices, valid_indices)

def identity(x):
    return x