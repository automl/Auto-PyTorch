__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import unittest
import numpy as np
import time

import torch
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from autonet.utils.configspace_wrapper import ConfigWrapper
from autonet.pipeline.nodes.imputation import Imputation
from numpy.testing import assert_array_equal



class TestImputation(unittest.TestCase):


    def test_imputation(self):
        X_train = np.array([[1, np.nan, 3], [4, 5, 6], [7, 8, np.nan]])
        X_valid = np.array([[np.nan, 2, 3], [4, 5, np.nan], [7, np.nan, 9]])
        categorical_features = [False, True, False]
        hyperparameter_config = {Imputation.get_name() +  ConfigWrapper.delimiter + "strategy": "median"}

        imputation_node = Imputation()

        fit_result = imputation_node.fit(hyperparameter_config=hyperparameter_config, X_train=X_train, X_valid=X_valid,
            categorical_features=categorical_features)

        assert_array_equal(fit_result['X_train'], np.array([[1, 3, 9], [4, 6, 5], [7, 4.5, 8]]))
        assert_array_equal(fit_result['X_valid'], np.array([[4, 3, 2], [4, 4.5, 5], [7, 9, 9]]))
        assert_array_equal(fit_result['categorical_features'], [False, False, True])

        X_test = np.array([[np.nan, np.nan, np.nan]])

        predict_result = imputation_node.predict(X=X_test, imputation_preprocessor=fit_result['imputation_preprocessor'])
        assert_array_equal(predict_result['X'], np.array([[4, 4.5, 9]]))