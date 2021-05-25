import ConfigSpace as CS

import numpy as np

import pytest

from autoPyTorch.pipeline.traditional_tabular_classification import (
    TraditionalTabularClassificationPipeline,
)
from autoPyTorch.pipeline.components.setup.traditional_ml.classifier_models import _classifiers


@pytest.mark.parametrize("fit_dictionary_tabular",
                         ['classification_numerical_and_categorical'], indirect=True)
def test_traditional_tabular_pipeline(fit_dictionary_tabular):
    pipeline = TraditionalTabularClassificationPipeline(
        dataset_properties=fit_dictionary_tabular['dataset_properties']
    )
    assert pipeline._get_estimator_hyperparameter_name() == "tabular_classifier"
    cs = pipeline.get_hyperparameter_search_space()
    assert isinstance(cs, CS.ConfigurationSpace)
    config = cs.sample_configuration()
    assert config['model_trainer:tabular_classifier:classifier'] in _classifiers
    assert pipeline.get_pipeline_representation() == {
        'Preprocessing': 'None',
        'Estimator': 'TabularClassifier',
    }


@pytest.mark.parametrize("fit_dictionary_tabular",
                         ['classification_numerical_and_categorical'], indirect=True)
def test_traditional_tabular_pipeline_predict(fit_dictionary_tabular):
    pipeline = TraditionalTabularClassificationPipeline(
        dataset_properties=fit_dictionary_tabular['dataset_properties']
    )
    assert pipeline._get_estimator_hyperparameter_name() == "tabular_classifier"
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
