import logging
import re
import unittest
from unittest.mock import MagicMock

import numpy as np

import pytest

from autoPyTorch.api.base_task import BaseTask, _pipeline_predict
from autoPyTorch.constants import TABULAR_CLASSIFICATION, TABULAR_REGRESSION
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline


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
    api.task_type = None
    api._metric = MagicMock()
    with pytest.raises(ValueError, match=r".*AutoPytorch failed to infer a task type*"):
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


def test_pipeline_predict_function():
    X = np.ones((10, 10))
    pipeline = MagicMock()
    pipeline.predict.return_value = np.full((10,), 3)
    pipeline.predict_proba.return_value = np.full((10, 2), 3)

    # First handle the classification case
    task = TABULAR_CLASSIFICATION
    with pytest.raises(ValueError, match='prediction probability not within'):
        _pipeline_predict(pipeline, X, 5, logging.getLogger, task)
    pipeline.predict_proba.return_value = np.zeros((10, 2))
    predictions = _pipeline_predict(pipeline, X, 5, logging.getLogger(), task)
    assert np.shape(predictions) == (10, 2)

    task = TABULAR_REGRESSION
    predictions = _pipeline_predict(pipeline, X, 5, logging.getLogger(), task)
    assert np.shape(predictions) == (10,)
    # Trigger warning msg with different shape for prediction
    pipeline.predict.return_value = np.full((12,), 3)
    predictions = _pipeline_predict(pipeline, X, 5, logging.getLogger(), task)


@pytest.mark.parametrize("fit_dictionary_tabular", ['classification_categorical_only'], indirect=True)
def test_show_models(fit_dictionary_tabular):
    api = BaseTask()
    api.ensemble_ = MagicMock()
    api.models_ = [TabularClassificationPipeline(dataset_properties=fit_dictionary_tabular['dataset_properties'])]
    api.ensemble_.get_models_with_weights.return_value = [(1.0, api.models_[0])]
    # Expect the default configuration
    expected = (r"0\s+|\s+SimpleImputer,OneHotEncoder,NoScaler,NoFeaturePreprocessing\s+"
                r"|\s+no embedding,ShapedMLPBackbone,FullyConnectedHead,nn.Sequential\s+|\s+1")
    assert re.search(expected, api.show_models()) is not None


def test_set_pipeline_config():
    # checks if we can correctly change the pipeline options
    estimator = BaseTask()
    pipeline_options = {"device": "cuda",
                        "budget_type": "epochs",
                        "min_epochs": 10,
                        "epochs": 51,
                        "runtime": 360}
    estimator.set_pipeline_config(**pipeline_options)
    assert pipeline_options.items() <= estimator.get_pipeline_options().items()