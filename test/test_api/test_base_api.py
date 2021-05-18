import os
import pathlib
import pickle
import sys
import unittest
from test.test_api.utils import dummy_do_dummy_prediction, dummy_eval_function

import numpy as np

import pandas as pd

import pytest


import sklearn
import sklearn.datasets
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier, VotingRegressor

from smac.runhistory.runhistory import RunHistory

from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.api.tabular_regression import TabularRegressionTask
from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
    HoldoutValTypes,
)
from autoPyTorch.optimizer.smbo import AutoMLSMBO
from autoPyTorch.pipeline.components.training.metrics.metrics import accuracy

from autoPyTorch.api.base_task import BaseTask


# ====
# Test
# ====
@pytest.mark.parametrize("fit_dictionary_tabular", ['classification_categorical_only'], indirect=True)
def test_nonsupported_arguments(fit_dictionary_tabular):
    with pytest.raises(ValueError, match=r".*Expected search space updates to be of instance.*"):
        api = BaseTask(search_space_updates='None')

    api = BaseTask()
    with pytest.raises(ValueError, match=r".*Invalid configuration arguments given.*"):
        api.set_pipeline_config(unsupported=True)
    with pytest.raises(ValueError, match=r".*No search space initialised and no dataset.*"):
        api.get_search_space()
    api.resampling_strategy = None
    with pytest.raises(ValueError, match=r".*Resampling strategy is needed to determine.*"):
        api._load_models()
    api.resampling_strategy = unittest.mock.MagicMock()
    with pytest.raises(ValueError, match=r".*Providing a metric to AutoPytorch is required.*"):
        api._load_models()
    api.ensemble_ = unittest.mock.MagicMock()
    with pytest.raises(ValueError, match=r".*No metric found. Either fit/search has not been.*"):
        api.score(np.ones(10), np.ones(10))
    api._metric = unittest.mock.MagicMock()
    with pytest.raises(ValueError, match=r".*No valid model found in run history.*"):
        api._load_models()
    dataset = fit_dictionary_tabular['backend'].load_datamanager()
    with pytest.raises(ValueError, match=r".*Incompatible dataset entered for current task.*"):
        api._search('accuracy', dataset)

    def returnfalse():
        return False

    api._load_models = returnfalse
    with pytest.raises(ValueError, match=r".*No ensemble found. Either fit has not yet.*"):
        api.predict(np.ones((10, 10)))
    with pytest.raises(ValueError, match=r".*No ensemble found. Either fit has not yet.*"):
        api.predict(np.ones((10, 10)))
