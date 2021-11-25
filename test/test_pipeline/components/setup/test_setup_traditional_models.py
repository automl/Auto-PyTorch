import copy
import os
import pickle
import sys

import numpy as np

import pytest

from autoPyTorch.pipeline.components.setup.traditional_ml import ModelChoice
from autoPyTorch.pipeline.components.setup.traditional_ml.tabular_traditional_model import TabularTraditionalModel


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


@pytest.fixture(params=['lgb', 'catboost',
                        'random_forest',
                        'extra_trees', 'svm', 'knn'])
def traditional_learner(request):
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


@pytest.mark.parametrize("fit_dictionary_tabular", ['classification_categorical_only',
                                                    'classification_numerical_only',
                                                    'classification_numerical_and_categorical',
                                                    "regression_numerical_only",
                                                    "regression_categorical_only",
                                                    "regression_numerical_and_categorical"
                                                    ], indirect=True)
def test_model_fit_predict_score(traditional_learner, fit_dictionary_tabular):

    if len(fit_dictionary_tabular['dataset_properties']['numerical_columns']) == 0 and traditional_learner == 'knn':
        pytest.skip("knn can not work with categorical only data")

    model = TabularTraditionalModel(traditional_learner=traditional_learner)

    assert isinstance(model.get_properties(), dict)

    blockPrint()
    model.fit(X=fit_dictionary_tabular)
    enablePrint()

    assert isinstance(model.fit_output, dict)
    assert 'val_preds' in model.fit_output.keys()
    assert isinstance(model.fit_output['val_preds'], list)
    assert len(model.fit_output['val_preds']) == len(fit_dictionary_tabular['val_indices'])
    if model.model.is_classification:
        assert len(model.fit_output['val_preds'][0]) == len(np.unique(fit_dictionary_tabular['y_train']))
    assert len(np.argwhere(0 > np.array(model.fit_output['val_preds']).all() > 1)) == 0
    assert 'labels' in model.fit_output.keys()
    assert len(model.fit_output['labels']) == len(fit_dictionary_tabular['val_indices'])
    assert 'train_score' in model.fit_output.keys()
    assert isinstance(model.fit_output['train_score'], float)
    assert 'val_score' in model.fit_output.keys()
    assert isinstance(model.fit_output['val_score'], float)

    # Test if traditional model can predict on val set
    if model.model.is_classification:
        y_pred = model.predict_proba(fit_dictionary_tabular['X_train'][fit_dictionary_tabular['val_indices']])
    else:
        y_pred = model.predict(fit_dictionary_tabular['X_train'][fit_dictionary_tabular['val_indices']])
        with pytest.raises(ValueError, match="Can't predict probabilities for a regressor"):
            model.predict_proba(fit_dictionary_tabular['X_train'][fit_dictionary_tabular['val_indices']])

    assert np.allclose(y_pred.squeeze(), model.fit_output['val_preds'], atol=1e-04)
    assert y_pred.shape[0] == len(fit_dictionary_tabular['val_indices'])
    # Test if classifier can score and
    # the result is same as in results
    score = model.score(fit_dictionary_tabular['X_train'][fit_dictionary_tabular['val_indices']],
                        fit_dictionary_tabular['y_train'][fit_dictionary_tabular['val_indices']])
    assert np.allclose(score, model.fit_output['val_score'], atol=1e-6)

    dump_file = os.path.join(fit_dictionary_tabular['backend'].temporary_directory, 'dump.pkl')

    with open(dump_file, 'wb') as f:
        pickle.dump(model, f)

    with open(dump_file, 'rb') as f:
        restored_estimator = pickle.load(f)
    restored_estimator.predict(fit_dictionary_tabular['X_train'])
