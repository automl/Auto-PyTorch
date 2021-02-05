from copy import deepcopy
from functools import partial
from typing import Tuple

import ConfigSpace as CS
from hpbandster.core.worker import Worker
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import balanced_accuracy_score
import torch
import xgboost as xgb


def balanced_error(
    threshold_predictions,
    predt: np.ndarray,
    dtrain: xgb.DMatrix,
) -> Tuple[str, float]:

    if threshold_predictions:
        predt = np.array(predt)
        predt = predt > 0.5
        predt = predt.astype(int)
    else:
        predt = np.argmax(predt, axis=1)
    y_train = dtrain.get_label()
    accuracy_score = balanced_accuracy_score(y_train, predt)

    return 'Balanced_error', 1 - accuracy_score


class XGBoostWorker(Worker):

    def __init__(self, *args, param=None, splits=None, categorical_information=None, **kwargs):

        super().__init__(*args, **kwargs)
        self.param=param
        self.splits = splits
        self.categorical_ind = categorical_information

        if self.param['objective'] == 'binary:logistic':
            self.threshold_predictions = True
        else:
            self.threshold_predictions = False

    def compute(self, config, budget, **kwargs):
        """
        Simple example for a compute function
        The loss is just a the config + some noise (that decreases with the budget)
        For dramatization, the function can sleep for a given interval to emphasizes
        the speed ups achievable with parallel workers.
        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train
        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        xgboost_config = deepcopy(self.param)
        xgboost_config.update(config)
        num_rounds = xgboost_config['num_round']
        del xgboost_config['num_round']
        X_train = self.splits['X_train']
        X_val = self.splits['X_val']
        X_test = self.splits['X_test']
        y_train = self.splits['y_train']
        y_val = self.splits['y_val']
        y_test = self.splits['y_test']


        d_train = xgb.DMatrix(X_train, label=y_train)
        d_val = xgb.DMatrix(X_val, label=y_val)
        d_test = xgb.DMatrix(X_test, label=y_test)


        eval_results = {}
        gb_model = xgb.train(
            xgboost_config,
            d_train,
            num_rounds,
            feval=partial(balanced_error, self.threshold_predictions),
            evals=[(d_train, 'd_train'), (d_val, 'd_val')],
            evals_result=eval_results,
        )
        #TODO Do something with eval_results in the future
        # print(eval_results)
        # make prediction
        y_train_preds = gb_model.predict(d_train)
        y_val_preds = gb_model.predict(d_val)
        y_test_preds = gb_model.predict(d_test)

        if self.threshold_predictions:
            y_train_preds = np.array(y_train_preds)
            y_train_preds = y_train_preds > 0.5
            y_train_preds = y_train_preds.astype(int)

            y_val_preds = np.array(y_val_preds)
            y_val_preds = y_val_preds > 0.5
            y_val_preds = y_val_preds.astype(int)

            y_test_preds = np.array(y_test_preds)
            y_test_preds = y_test_preds > 0.5
            y_test_preds = y_test_preds.astype(int)

        train_performance = balanced_accuracy_score(y_train, y_train_preds)
        val_performance = balanced_accuracy_score(y_val, y_val_preds)
        test_performance = balanced_accuracy_score(y_test, y_test_preds)

        if val_performance is None or val_performance is np.inf:
            val_error_rate = 1
        else:
            val_error_rate = 1 - val_performance

        res = {
            'train_accuracy': float(train_performance),
            'val_accuracy': float(val_performance),
            'test_accuracy': float(test_performance),
        }

        return ({
            'loss': float(val_error_rate),  # this is the a mandatory field to run hyperband
            'info': res  # can be used for any user-defined information - also mandatory
        })

    def refit(self, config):

        xgboost_config = deepcopy(self.param)
        xgboost_config.update(config)
        num_rounds = xgboost_config['num_round']
        del xgboost_config['num_round']
        X_train = self.splits['X_train']
        X_test = self.splits['X_test']
        y_train = self.splits['y_train']
        y_test = self.splits['y_test']

        d_train = xgb.DMatrix(X_train, label=y_train)
        d_test = xgb.DMatrix(X_test, label=y_test)

        eval_results = {}
        gb_model = xgb.train(
            xgboost_config,
            d_train,
            num_rounds,
            feval=partial(balanced_error, self.threshold_predictions),
            evals=[(d_train, 'd_train'), (d_test, 'd_test')],
            evals_result=eval_results,
        )
        #TODO do something with eval_results
        #print(eval_results)
        #make prediction
        y_train_preds = gb_model.predict(d_train)
        y_test_preds = gb_model.predict(d_test)

        if self.threshold_predictions:
            y_train_preds = np.array(y_train_preds)
            y_train_preds = y_train_preds > 0.5
            y_train_preds = y_train_preds.astype(int)

            y_test_preds = np.array(y_test_preds)
            y_test_preds = y_test_preds > 0.5
            y_test_preds = y_test_preds.astype(int)

        train_performance = balanced_accuracy_score(y_train, y_train_preds)
        test_performance = balanced_accuracy_score(y_test, y_test_preds)

        if test_performance is None or test_performance is np.inf:
            test_performance = 0

        res = {
            'train_accuracy': float(train_performance),
            'test_accuracy': float(test_performance),
        }

        return res

    @staticmethod
    def get_default_configspace(seed=11):

        config_space = CS.ConfigurationSpace(seed=seed)
        # learning rate
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                'eta',
                lower=0.001,
                upper=1,
                log=True,
            )
        )
        # l2 regularization
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                'lambda',
                lower=1E-10,
                upper=1,
                log=True,
            )
        )
        # l1 regularization
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                'alpha',
                lower=1E-10,
                upper=1,
                log=True,
            )
        )
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                'num_round',
                lower=1,
                upper=100,
            )
        )
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                'gamma',
                lower=0.1,
                upper=1,
                log=True,
            )
        )
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                'colsample_bylevel',
                lower=0.1,
                upper=1,
            )
        )
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                'colsample_bynode',
                lower=0.1,
                upper=1,
            )
        )
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                'colsample_bytree',
                lower=0.5,
                upper=1,
            )
        )
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                'max_depth',
                lower=1,
                upper=20,
            )
        )
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                'max_delta_step',
                lower=0,
                upper=10,
            )
        )
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                'min_child_weight',
                lower=0.1,
                upper=20,
                log=True,
            )
        )
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                'subsample',
                lower=0.01,
                upper=1,
            )
        )

        return config_space


class TabNetWorker(Worker):

    def __init__(self, *args, param=None, splits=None, categorical_information=None, **kwargs):

        super().__init__(*args, **kwargs)
        self.param=param
        self.splits = splits
        if categorical_information is not None:
            self.categorical_ind = categorical_information['categorical_ind']
            self.categorical_columns = categorical_information['categorical_columns']
            self.categorical_dimensions = categorical_information['categorical_dimensions']



    def compute(self, config, budget, **kwargs):
        """
        Simple example for a compute function
        The loss is just a the config + some noise (that decreases with the budget)
        For dramatization, the function can sleep for a given interval to emphasizes
        the speed ups achievable with parallel workers.
        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train
        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        X_train = self.splits['X_train']
        X_val = self.splits['X_val']
        X_test = self.splits['X_test']
        y_train = self.splits['y_train']
        y_val = self.splits['y_val']
        y_test = self.splits['y_test']



        clf = TabNetClassifier(
            n_a=config['na'],
            n_d=config['na'],
            n_steps=config['nsteps'],
            gamma=config['gamma'],
            lambda_sparse=config['lambda_sparse'],
            momentum=config['mb'],
            cat_idxs=self.categorical_columns,
            cat_dims=self.categorical_dimensions,
            seed=self.param['seed'],
            optimizer_params={
                'lr': config['learning_rate'],
            },
            scheduler_params={
                'step_size': config['decay_iterations'],
                'gamma': config['decay_rate'],
            },
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
        )
        batch_size = config['batch_size']
        if batch_size == 32768:
            vbatch_size = config['vbatch_size1']
        elif batch_size == 16384:
            vbatch_size = config['vbatch_size2']
        elif batch_size == 8192:
            vbatch_size = config['vbatch_size3']
        elif batch_size == 4096:
            vbatch_size = config['vbatch_size4']
        elif batch_size == 2048:
            vbatch_size = config['vbatch_size5']
        elif batch_size == 1024:
            vbatch_size = config['vbatch_size6']
        elif batch_size == 512:
            vbatch_size = config['vbatch_size7']
        elif batch_size == 256:
            vbatch_size = config['vbatch_size8']
        else:
            raise ValueError('Illegal batch size given')

        clf.fit(
            X_train=X_train, y_train=y_train,
            batch_size=batch_size,
            virtual_batch_size=vbatch_size,
            eval_set=[(X_val, y_val)],
            eval_name=['Validation'],
            eval_metric=['balanced_accuracy'],
            max_epochs=200,
            patience=0,
        )

        y_train_preds = clf.predict(X_train)
        y_val_preds = clf.predict(X_val)
        y_test_preds = clf.predict(X_test)

        train_performance = balanced_accuracy_score(y_train, y_train_preds)
        val_performance = balanced_accuracy_score(y_val, y_val_preds)
        test_performance = balanced_accuracy_score(y_test, y_test_preds)

        if val_performance is None or val_performance is np.inf:
            val_error_rate = 1
        else:
            val_error_rate = 1 - val_performance

        res = {
            'train_accuracy': float(train_performance),
            'val_accuracy': float(val_performance),
            'test_accuracy': float(test_performance),
        }

        return ({
            'loss': float(val_error_rate),  # this is the a mandatory field to run hyperband
            'info': res  # can be used for any user-defined information - also mandatory
        })

    def refit(self, config):

        X_train = self.splits['X_train']
        X_test = self.splits['X_test']
        y_train = self.splits['y_train']
        y_test = self.splits['y_test']

        categorical_columns = []
        categorical_dimensions = []

        for index, categorical_column in enumerate(self.categorical_ind):
            if categorical_column:
                column_unique_values = len(set(X_train[:, index]))
                column_max_index = int(max(X_train[:, index]))
                # categorical columns with only one unique value
                # do not need an embedding.
                if column_unique_values == 1:
                    continue
                categorical_columns.append(index)
                categorical_dimensions.append(column_max_index + 1)

        clf = TabNetClassifier(
            n_a=config['na'],
            n_d=config['na'],
            n_steps=config['nsteps'],
            gamma=config['gamma'],
            lambda_sparse=config['lambda_sparse'],
            momentum=config['mb'],
            cat_idxs=self.categorical_columns,
            cat_dims=self.categorical_dimensions,
            seed=self.param['seed'],
            optimizer_params={
                'lr': config['learning_rate'],
            },
            scheduler_params={
                'step_size': config['decay_iterations'],
                'gamma': config['decay_rate'],
            },
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
        )
        batch_size = config['batch_size']
        if batch_size == 32768:
            vbatch_size = config['vbatch_size1']
        elif batch_size == 16384:
            vbatch_size = config['vbatch_size2']
        elif batch_size == 8192:
            vbatch_size = config['vbatch_size3']
        elif batch_size == 4096:
            vbatch_size = config['vbatch_size4']
        elif batch_size == 2048:
            vbatch_size = config['vbatch_size5']
        elif batch_size == 1024:
            vbatch_size = config['vbatch_size6']
        elif batch_size == 512:
            vbatch_size = config['vbatch_size7']
        elif batch_size == 256:
            vbatch_size = config['vbatch_size8']
        else:
            raise ValueError('Illegal batch size given')

        clf.fit(
            X_train=X_train, y_train=y_train,
            batch_size=batch_size,
            virtual_batch_size=vbatch_size,
            eval_metric=['balanced_accuracy'],
            max_epochs=200,
        )

        y_train_preds = clf.predict(X_train)
        y_test_preds = clf.predict(X_test)

        train_performance = balanced_accuracy_score(y_train, y_train_preds)
        test_performance = balanced_accuracy_score(y_test, y_test_preds)

        if test_performance is None or test_performance is np.inf:
            test_performance = 0

        res = {
            'train_accuracy': float(train_performance),
            'test_accuracy': float(test_performance),
        }

        return res

    @staticmethod
    def get_default_configspace(seed=11):

        config_space = CS.ConfigurationSpace(seed=seed)
        # learning rate
        config_space.add_hyperparameter(
            CS.CategoricalHyperparameter(
                'na',
                choices=[8, 16, 24, 32, 64, 128],
            )
        )
        config_space.add_hyperparameter(
            CS.CategoricalHyperparameter(
                'learning_rate',
                choices=[0.005, 0.01, 0.02, 0.025],
            )
        )
        config_space.add_hyperparameter(
            CS.CategoricalHyperparameter(
                'gamma',
                choices=[1.0, 1.2, 1.5, 2.0],
            )
        )
        config_space.add_hyperparameter(
            CS.CategoricalHyperparameter(
                'nsteps',
                choices=[3, 4, 5, 6, 7, 8, 9, 10],
            )
        )
        config_space.add_hyperparameter(
            CS.CategoricalHyperparameter(
                'lambda_sparse',
                choices=[0, 0.000001, 0.0001, 0.001, 0.01, 0.1],
            )
        )
        batch_size = CS.CategoricalHyperparameter(
            'batch_size',
            choices=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        )
        vbatch_size1 = CS.CategoricalHyperparameter(
            'vbatch_size1',
            choices=[256, 512, 1024, 2048, 4096],
        )
        vbatch_size2 = CS.CategoricalHyperparameter(
            'vbatch_size2',
            choices=[256, 512, 1024, 2048, 4096],
        )
        vbatch_size3 = CS.CategoricalHyperparameter(
            'vbatch_size3',
            choices=[256, 512, 1024, 2048, 4096],
        )
        vbatch_size4 = CS.CategoricalHyperparameter(
            'vbatch_size4',
            choices=[256, 512, 1024, 2048],
        )
        vbatch_size5 = CS.CategoricalHyperparameter(
            'vbatch_size5',
            choices=[256, 512, 1024],
        )
        vbatch_size6 = CS.CategoricalHyperparameter(
            'vbatch_size6',
            choices=[256, 512],
        )
        vbatch_size7 = CS.Constant(
            'vbatch_size7',
            256
        )
        vbatch_size8 = CS.Constant(
            'vbatch_size8',
            256
        )
        config_space.add_hyperparameter(
            batch_size
        )
        config_space.add_hyperparameters(
            [
                vbatch_size1,
                vbatch_size2,
                vbatch_size3,
                vbatch_size4,
                vbatch_size5,
                vbatch_size6,
                vbatch_size7,
                vbatch_size8,
            ]
        )
        config_space.add_hyperparameter(
            CS.CategoricalHyperparameter(
                'decay_rate',
                choices=[0.4, 0.8, 0.9, 0.95],
            )
        )
        config_space.add_hyperparameter(
            CS.CategoricalHyperparameter(
                'decay_iterations',
                choices=[500, 2000, 8000, 10000, 20000],
            )
        )
        config_space.add_hyperparameter(
            CS.CategoricalHyperparameter(
                'mb',
                choices=[0.6, 0.7, 0.8, 0.9, 0.95, 0.98],
            )
        )

        config_space.add_condition(
            CS.EqualsCondition(
                vbatch_size1,
                batch_size,
                32768,
            )
        )
        config_space.add_condition(
            CS.EqualsCondition(
                vbatch_size2,
                batch_size,
                16384,
            )
        )
        config_space.add_condition(
            CS.EqualsCondition(
                vbatch_size3,
                batch_size,
                8192,
            )
        )
        config_space.add_condition(
            CS.EqualsCondition(
                vbatch_size4,
                batch_size,
                4096,
            )
        )
        config_space.add_condition(
            CS.EqualsCondition(
                vbatch_size5,
                batch_size,
                2048,
            )
        )
        config_space.add_condition(
            CS.EqualsCondition(
                vbatch_size6,
                batch_size,
                1024,
            )
        )
        config_space.add_condition(
            CS.EqualsCondition(
                vbatch_size7,
                batch_size,
                512,
            )
        )
        config_space.add_condition(
            CS.EqualsCondition(
                vbatch_size8,
                batch_size,
                256,
            )
        )

        return config_space
