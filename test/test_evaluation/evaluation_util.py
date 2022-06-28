import functools
import traceback
import unittest

import numpy as np
from numpy.linalg import LinAlgError

import scipy.sparse

import sklearn.datasets
import sklearn.model_selection
from sklearn import preprocessing

from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.data.time_series_forecasting_validator import TimeSeriesForecastingInputValidator
from autoPyTorch.datasets.resampling_strategy import HoldoutValTypes
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.datasets.time_series_dataset import TimeSeriesForecastingDataset
from autoPyTorch.pipeline.components.training.metrics.metrics import (
    accuracy,
    balanced_accuracy,
    log_loss
)

SCORER_LIST = [accuracy, balanced_accuracy, log_loss]

N_TEST_RUNS = 5


def get_dataset(dataset='iris', make_sparse=False, add_NaNs=False,
                train_size_maximum=150, make_multilabel=False,
                make_binary=False):
    iris = getattr(sklearn.datasets, "load_%s" % dataset)()
    X = iris.data.astype(np.float32)
    Y = iris.target
    rs = np.random.RandomState(42)
    indices = np.arange(X.shape[0])
    train_size = min(int(len(indices) / 3. * 2.), train_size_maximum)
    rs.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_test = X[train_size:]
    Y_test = Y[train_size:]
    if add_NaNs:
        mask = rs.choice([True, False], size=(X_train.shape))
        X_train[mask] = np.NaN
    if make_sparse:
        X_train[:, 0] = 0
        X_train[rs.random_sample(X_train.shape) > 0.5] = 0
        X_train = scipy.sparse.csc_matrix(X_train)
        X_train.eliminate_zeros()
        X_test[:, 0] = 0
        X_test[rs.random_sample(X_test.shape) > 0.5] = 0
        X_test = scipy.sparse.csc_matrix(X_test)
        X_test.eliminate_zeros()
    if make_binary and make_multilabel:
        raise ValueError('Can convert dataset only to one of the two '
                         'options binary or multilabel!')
    if make_binary:
        Y_train[Y_train > 1] = 1
        Y_test[Y_test > 1] = 1
    if make_multilabel:
        num_classes = len(np.unique(Y))
        Y_train_ = np.zeros((Y_train.shape[0], num_classes))
        for i in range(Y_train.shape[0]):
            Y_train_[i, Y_train[i]] = 1
        Y_train = Y_train_
        Y_test_ = np.zeros((Y_test.shape[0], num_classes))
        for i in range(Y_test.shape[0]):
            Y_test_[i, Y_test[i]] = 1
        Y_test = Y_test_
    return X_train, Y_train, X_test, Y_test


class Dummy(object):
    def __init__(self):
        self.name = 'Dummy'


class BaseEvaluatorTest(unittest.TestCase):
    def __init__(self, methodName):
        super(BaseEvaluatorTest, self).__init__(methodName)
        self.output_directories = []

    def _fit(self, evaluator):
        return self.__fit(evaluator.search)

    def _partial_fit(self, evaluator, fold):
        partial_fit = functools.partial(evaluator.partial_fit, fold=fold)
        return self.__fit(partial_fit)

    def __fit(self, function_handle):
        """Allow us to catch known and valid exceptions for all evaluate
        scripts."""
        try:
            function_handle()
            return True
        except KeyError as e:
            if 'Floating-point under-/overflow occurred at epoch' in \
                    e.args[0] or \
                    'removed all features' in e.args[0] or \
                    'failed to create intent' in e.args[0]:
                pass
            else:
                traceback.print_exc()
                raise e
        except ValueError as e:
            if 'Floating-point under-/overflow occurred at epoch' in e.args[0]:
                pass
            elif 'removed all features' in e.args[0]:
                pass
            elif 'failed to create intent' in e.args[0]:
                pass
            else:
                raise e
        except LinAlgError as e:
            if 'not positive definite, even with jitter' in e.args[0]:
                pass
            else:
                raise e
        except RuntimeWarning as e:
            if 'invalid value encountered in sqrt' in e.args[0]:
                pass
            elif 'divide by zero encountered in divide' in e.args[0]:
                pass
            else:
                raise e
        except UserWarning as e:
            if 'FastICA did not converge' in e.args[0]:
                pass
            else:
                raise e


def get_multiclass_classification_datamanager(resampling_strategy=HoldoutValTypes.holdout_validation):
    X_train, Y_train, X_test, Y_test = get_dataset('iris')
    indices = list(range(X_train.shape[0]))
    np.random.seed(1)
    np.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    validator = TabularInputValidator(is_classification=True).fit(X_train, Y_train)
    dataset = TabularDataset(
        X=X_train, Y=Y_train,
        X_test=X_test, Y_test=Y_test,
        validator=validator,
        resampling_strategy=resampling_strategy
    )
    return dataset


def get_abalone_datamanager(resampling_strategy=HoldoutValTypes.holdout_validation):
    # https://www.openml.org/d/183
    X, y = sklearn.datasets.fetch_openml(data_id=183, return_X_y=True, as_frame=False)
    y = preprocessing.LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=1
    )

    validator = TabularInputValidator(is_classification=True).fit(X_train, y_train)
    dataset = TabularDataset(
        X=X_train, Y=y_train,
        validator=validator,
        X_test=X_test, Y_test=y_test,
        resampling_strategy=resampling_strategy
    )
    return dataset


def get_binary_classification_datamanager(resampling_strategy=HoldoutValTypes.holdout_validation):
    X_train, Y_train, X_test, Y_test = get_dataset('iris')
    indices = list(range(X_train.shape[0]))
    np.random.seed(1)
    np.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    eliminate_class_two = Y_train != 2
    X_train = X_train[eliminate_class_two]
    Y_train = Y_train[eliminate_class_two]

    eliminate_class_two = Y_test != 2
    X_test = X_test[eliminate_class_two]
    Y_test = Y_test[eliminate_class_two]

    validator = TabularInputValidator(is_classification=True).fit(X_train, Y_train)
    dataset = TabularDataset(
        X=X_train, Y=Y_train,
        X_test=X_test, Y_test=Y_test,
        validator=validator,
        resampling_strategy=resampling_strategy
    )
    return dataset


def get_regression_datamanager(resampling_strategy=HoldoutValTypes.holdout_validation):
    X_train, Y_train, X_test, Y_test = get_dataset('boston')
    indices = list(range(X_train.shape[0]))
    np.random.seed(1)
    np.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    validator = TabularInputValidator(is_classification=True).fit(X_train, Y_train)
    dataset = TabularDataset(
        X=X_train, Y=Y_train,
        X_test=X_test, Y_test=Y_test,
        validator=validator,
        resampling_strategy=resampling_strategy
    )
    return dataset


def get_500_classes_datamanager(resampling_strategy=HoldoutValTypes.holdout_validation):
    weights = ([0.002] * 475) + ([0.001] * 25)
    X, Y = sklearn.datasets.make_classification(n_samples=1000,
                                                n_features=20,
                                                n_classes=500,
                                                n_clusters_per_class=1,
                                                n_informative=15,
                                                n_redundant=5,
                                                n_repeated=0,
                                                weights=weights,
                                                flip_y=0,
                                                class_sep=1.0,
                                                hypercube=True,
                                                shift=None,
                                                scale=1.0,
                                                shuffle=True,
                                                random_state=1)

    validator = TabularInputValidator(is_classification=True).fit(X, Y)
    dataset = TabularDataset(
        X=X[:700], Y=Y[:700],
        X_test=X[700:], Y_test=Y[710:],
        validator=validator,
        resampling_strategy=resampling_strategy
    )

    return dataset


def get_forecasting_dataset(n_seq=10,
                            n_prediction_steps=3,
                            resampling_strategy=HoldoutValTypes.time_series_hold_out_validation):
    base_length = 50
    X = []
    targets = []
    X_test = []
    Y_test = []

    for i in range(n_seq):
        series_length = base_length + i * 10

        targets.append(np.arange(i * 1000, series_length + i * 1000))
        X.append(targets[-1] - 1)
        X_test.append(np.arange(X[-1][-1] + 1, X[-1][-1] + 1 + n_prediction_steps))
        Y_test.append(np.arange(targets[-1][-1] + 1, targets[-1][-1] + 1 + n_prediction_steps))

    input_validator = TimeSeriesForecastingInputValidator(is_classification=False).fit(X, targets)
    return TimeSeriesForecastingDataset(X=X, Y=targets, X_test=X_test,
                                        Y_test=Y_test,
                                        known_future_features=(0,),
                                        validator=input_validator,
                                        resampling_strategy=resampling_strategy,
                                        n_prediction_steps=n_prediction_steps
                                        )


def get_dataset_getters():
    return [get_binary_classification_datamanager,
            get_multiclass_classification_datamanager,
            get_500_classes_datamanager,
            get_abalone_datamanager,
            get_regression_datamanager,
            get_forecasting_dataset]
