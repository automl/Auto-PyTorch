import copy
import os
import sys

import numpy as np

import pytest

from autoPyTorch.pipeline.components.setup.traditional_ml import ModelChoice
from autoPyTorch.pipeline.components.setup.traditional_ml.classifier_models.classifiers import (
    CatboostModel,
    ExtraTreesModel,
    KNNModel,
    LGBModel,
    RFModel,
    SVMModel
)


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


@pytest.fixture(params=[LGBModel(), CatboostModel(), SVMModel(),
                        RFModel(), ExtraTreesModel(), KNNModel()])
def classifier(request):
    return request.param


@pytest.fixture
def dataset_properties(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def dataset_properties_num_only():
    return {'numerical_columns': list(range(5))}


@pytest.fixture
def dataset_properties_categorical_only():
    return {'numerical_columns': list(range(0))}


@pytest.mark.parametrize("dataset_properties", ['dataset_properties_num_only',
                                                'dataset_properties_categorical_only'], indirect=True)
class TestModelChoice:
    def test_get_set_config_space(self, dataset_properties):
        """Make sure that we can setup a valid choice in the encoder
        choice"""
        model_choice = ModelChoice(dataset_properties)
        cs = model_choice.get_hyperparameter_search_space(dataset_properties=dataset_properties)

        # Make sure that all hyperparameters are part of the search space
        assert sorted(cs.get_hyperparameter('__choice__').choices) == sorted(list(model_choice.get_components().keys()))

        # Make sure we can properly set some random configs
        # Whereas just one iteration will make sure the algorithm works,
        # doing five iterations increase the confidence. We will be able to
        # catch component specific crashes
        for i in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            model_choice.set_hyperparameters(config)

            assert model_choice.choice.__class__ == model_choice.get_components()[config_dict['__choice__']]

            # Then check the choice configuration
            selected_choice = config_dict.pop('__choice__', None)
            for key, value in config_dict.items():
                # Remove the selected_choice string from the parameter
                # so we can query in the object for it
                key = key.replace(selected_choice + ':', '')
                assert key in vars(model_choice.choice)['config']
                assert value == model_choice.choice.__dict__['config'][key]


@pytest.mark.parametrize("dataset", ['dataset_traditional_classifier_num_only',
                                     'dataset_traditional_classifier_categorical_only',
                                     'dataset_traditional_classifier_num_categorical'], indirect=True)
class TestTraditionalClassifiers:
    def test_classifier_fit_predict(self, classifier, dataset):
        X, y = dataset

        blockPrint()
        try:
            results = classifier.fit(X_train=X, X_val=X, y_train=y, y_val=y)
        except ValueError as e:
            assert isinstance(classifier, KNNModel)
            assert 'Found array with 0 feature' in e.args[0]
            # KNN classifier works only on numerical data
            pytest.skip()

        enablePrint()

        assert isinstance(results, dict)
        assert 'val_preds' in results.keys()
        assert isinstance(results['val_preds'], list)
        assert len(results['val_preds']) == y.shape[0]
        assert len(results['val_preds'][0]) == len(np.unique(y))
        assert len(np.argwhere(0 > np.array(results['val_preds']).all() > 1)) == 0
        assert 'labels' in results.keys()
        assert len(results['labels']) == y.shape[0]
        assert 'train_score' in results.keys()
        assert isinstance(results['train_score'], float)
        assert 'val_score' in results.keys()
        assert isinstance(results['val_score'], float)

        # Test if classifier can predict on val set and
        # if the result is same as the one in results
        y_pred = classifier.predict(X, predict_proba=True)
        assert np.allclose(y_pred, results['val_preds'], atol=1e-04)
        assert y_pred.shape[0] == y.shape[0]
        # Test if classifier can score and
        # the result is same as in results
        score = classifier.score(X, y)
        assert score == results['val_score']
        # Test if score is greater than 0.8
        assert score >= 0.8
