import ConfigSpace as CS

import numpy as np

import pytest

from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner import _traditional_learners
from autoPyTorch.pipeline.traditional_tabular_classification import (
    TraditionalTabularClassificationPipeline,
)


@pytest.mark.parametrize("fit_dictionary_tabular",
                         ['classification_numerical_and_categorical',
                          'regression_numerical_and_categorical'], indirect=True)
def test_traditional_tabular_pipeline(fit_dictionary_tabular):
    pipeline = TraditionalTabularClassificationPipeline(
        dataset_properties=fit_dictionary_tabular['dataset_properties']
    )
    assert pipeline._get_estimator_hyperparameter_name() == "traditional_tabular_learner"
    cs = pipeline.get_hyperparameter_search_space()
    assert isinstance(cs, CS.ConfigurationSpace)
    config = cs.sample_configuration()
    assert config['model_trainer:tabular_traditional_model:traditional_learner'] in _traditional_learners
    assert pipeline.get_pipeline_representation() == {
        'Preprocessing': 'None',
        'Estimator': 'TabularTraditionalModel',
    }


@pytest.mark.parametrize("fit_dictionary_tabular",
                         ['classification_numerical_and_categorical'], indirect=True)
def test_traditional_tabular_pipeline_predict(fit_dictionary_tabular):
    pipeline = TraditionalTabularClassificationPipeline(
        dataset_properties=fit_dictionary_tabular['dataset_properties']
    )
    assert pipeline._get_estimator_hyperparameter_name() == "traditional_tabular_learner"
    config = pipeline.get_hyperparameter_search_space().get_default_configuration()
    pipeline.set_hyperparameters(config)
    pipeline.fit(fit_dictionary_tabular)
    prediction = pipeline.predict(fit_dictionary_tabular['X_train'])
    assert np.shape(fit_dictionary_tabular['X_train'])[0] == prediction.shape[0]
    assert prediction.shape[1] == 1
    prediction = pipeline.predict(fit_dictionary_tabular['X_train'], batch_size=5)
    assert np.shape(fit_dictionary_tabular['X_train'])[0] == prediction.shape[0]
    prediction = pipeline.predict_proba(fit_dictionary_tabular['X_train'], batch_size=5)
    assert np.shape(fit_dictionary_tabular['X_train'])[0] == prediction.shape[0]
