__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import unittest
import numpy as np
import time

import torch
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.pipeline.nodes.resampling_strategy_selector import ResamplingStrategySelector
from numpy.testing import assert_array_almost_equal
from autoPyTorch.components.preprocessing.resampling import TargetSizeStrategyUpsample, \
    RandomOverSamplingWithReplacement, RandomUnderSamplingWithReplacement


class TestResamplingStrategySelector(unittest.TestCase):


    def test_resampling_strategy_selector(self):
        X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        Y_train = np.array([[0, 1], [1, 0], [1, 0]])
        hyperparameter_config = {
            ResamplingStrategySelector.get_name() +  ConfigWrapper.delimiter + "over_sampling_method": "random_over_sampling",
            ResamplingStrategySelector.get_name() +  ConfigWrapper.delimiter + "under_sampling_method": "random_under_sampling",
            ResamplingStrategySelector.get_name() +  ConfigWrapper.delimiter + "target_size_strategy": "up",
        }

        resampler_node = ResamplingStrategySelector()
        resampler_node.add_over_sampling_method("random_over_sampling", RandomOverSamplingWithReplacement)
        resampler_node.add_under_sampling_method("random_under_sampling", RandomUnderSamplingWithReplacement)
        resampler_node.add_target_size_strategy("up", TargetSizeStrategyUpsample)

        fit_result = resampler_node.fit(hyperparameter_config=hyperparameter_config, X_train=X_train, Y_train=Y_train)

        num_0 = 0
        num_1 = 0
        for i in range(fit_result['X_train'].shape[0]):
            x = fit_result['X_train'][i, :]
            y = fit_result['Y_train'][i, :]

            if np.all(y == np.array([0, 1])):
                assert_array_almost_equal(x, np.array([1, 2, 3]))
                num_0 += 1
            else:
                self.assertTrue(np.all(x == np.array([4, 5, 6])) or  np.all(x == np.array([7, 8, 9])))
                num_1 += 1
        self.assertEqual(num_0, 2)
        self.assertEqual(num_1, 2)
        
