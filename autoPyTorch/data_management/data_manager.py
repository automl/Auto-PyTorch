from __future__ import print_function, division
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import sys
# from autoPyTorch.core.autonet_config import map_configs

from autoPyTorch.data_management.data_reader import CSVReader, OpenMlReader, AutoMlReader
from sklearn.datasets import make_regression, make_multilabel_classification

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

from enum import Enum
class ProblemType(Enum):
    FeatureClassification = 1
    FeatureRegression = 3
    FeatureMultilabel = 4

class DataManager(object):
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.X_train, self.Y_train = None, None
        self.X_test, self.Y_test = None, None
        self.X_valid, self.Y_valid = None, None
        self.is_classification = None
        self.is_multilabel = None
        self.metric = None
        self.max_runtime = None
        self.categorical_features = None

    def read_data(self, file_name, test_split=0.0, is_classification=None, random_seed=0, **kwargs):
        print("Read:" + file_name)
        reader = self._get_reader(file_name, is_classification)
        reader.read()
        self.categorical_features = reader.categorical_features

        if reader.is_multilabel:
            self.problem_type = ProblemType.FeatureMultilabel
        elif reader.is_classification:
            self.problem_type = ProblemType.FeatureClassification
        else:
            self.problem_type = ProblemType.FeatureRegression

        self.X, self.Y = reader.X, reader.Y
        self.X_valid, self.Y_valid = reader.X_valid, reader.Y_valid
        self.X_test, self.Y_test = reader.X_test, reader.Y_test
        self.max_runtime = reader.max_runtime
        self.metric = reader.metric
        self._split_data(test_split, random_seed)

    def _get_reader(self, file_name, is_classification):
        if file_name.endswith(".csv"):
            reader = CSVReader(file_name, is_classification=is_classification)
        elif file_name.startswith("openml:"):
            dataset_id = int(file_name.split(":")[1])
            reader = OpenMlReader(dataset_id, is_classification=is_classification)
        elif file_name.endswith(".info"):
            reader = AutoMlReader(file_name)
        else:
            raise ValueError("That filetype is not supported: " + file_name)
        return reader

    def generate_classification(self, num_classes, num_features, num_samples, test_split=0.1, seed=0):
        #X, Y = make_classification(n_samples=800, n_features=num_feats, n_classes=num_classes, n_informative=4)
        X, y = make_multilabel_classification(
            n_samples=num_samples, n_features=num_features, n_classes=num_classes, n_labels=0.01,
            length=50, allow_unlabeled=False, sparse=False, return_indicator='dense',
            return_distributions=False, random_state=seed
        )
        Y = np.argmax(y, axis=1)
        self.categorical_features = [False] * num_features
        self.problem_type = ProblemType.FeatureClassification
        self.X, self.Y = X, Y
        self._split_data(test_split, seed)

    def generate_regression(self, num_features, num_samples, test_split=0.1, seed=0):
        X, Y = make_regression(n_samples=num_samples, n_features=num_features, random_state=seed)
        self.categorical_features = [False] * num_features
        self.problem_type = ProblemType.FeatureRegression
        self.X, self.Y = X, Y
        self._split_data(test_split, seed)
        
    def _split_data(self, test_split, seed):
        valid_specified = self.X_valid is not None and self.Y_valid is not None
        test_specified = self.X_test is not None and self.Y_test is not None

        if not valid_specified and not test_specified:
            self.X, self.Y, self.X_train, self.Y_train, self.X_test, self.Y_test = deterministic_shuffle_and_split(self.X, self.Y, test_split, seed=seed)
            return
        if not test_specified:
            # assume only interested in validation performance
            self.X_test = self.X_valid
            self.Y_test = self.Y_valid
        self.X_train = self.X
        self.Y_train = self.Y

def deterministic_shuffle_and_split(X, Y, split, seed):
    rng = np.random.RandomState(seed)
    p = rng.permutation(X.shape[0])

    X = X[p]
    Y = Y[p]
    if 0. < split < 1.:
        split = int(split * X.shape[0])
        return X, Y, X[0:-split], Y[0:-split], X[-split:], Y[-split:]
    else:
        return X, Y, X, Y, None, None