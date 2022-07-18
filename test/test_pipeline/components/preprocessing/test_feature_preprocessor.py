import flaky

import numpy as np

import pytest

from sklearn.base import BaseEstimator
from sklearn.compose import make_column_transformer

from autoPyTorch.constants import CLASSIFICATION_TASKS, REGRESSION_TASKS, STRING_TO_TASK_TYPES
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing import (
    FeatureProprocessorChoice
)
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    NoFeaturePreprocessor import NoFeaturePreprocessor
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline


@pytest.fixture
def random_state():
    return 11


@pytest.fixture(params=['NoFeaturePreprocessor',
                        'FastICA',
                        'KernelPCA',
                        'RandomKitchenSinks',
                        'Nystroem',
                        'PolynomialFeatures',
                        'TruncatedSVD',
                        'ExtraTreesPreprocessorClassification',
                        'ExtraTreesPreprocessorRegression',
                        'FeatureAgglomeration',
                        'RandomTreesEmbedding',
                        'SelectPercentileClassification',
                        'SelectPercentileRegression',
                        'SelectRatesClassification',
                        'SelectRatesRegression',
                        'LibLinearSVCPreprocessor'
                        ])
def preprocessor(request):
    return request.param


@pytest.mark.parametrize("fit_dictionary_tabular", ['classification_numerical_only',
                                                    'classification_numerical_and_categorical',
                                                    'regression_numerical_only'], indirect=True)
class TestFeaturePreprocessors:

    def test_feature_preprocessor(self, fit_dictionary_tabular, preprocessor, random_state):
        task_type = str(fit_dictionary_tabular['dataset_properties']['task_type'])
        if (
            ("Classification" in preprocessor or preprocessor == "LibLinearSVCPreprocessor")
            and STRING_TO_TASK_TYPES[task_type] not in CLASSIFICATION_TASKS
        ):
            pytest.skip("Tests not relevant for {}".format(preprocessor.__class__.__name__))
        elif "Regression" in preprocessor and STRING_TO_TASK_TYPES[task_type] not in REGRESSION_TASKS:
            pytest.skip("Tests not relevant for {}".format(preprocessor.__class__.__name__))
        preprocessor = FeatureProprocessorChoice(
            dataset_properties=fit_dictionary_tabular['dataset_properties']
        ).get_components()[preprocessor]

        configuration = preprocessor. \
            get_hyperparameter_search_space(dataset_properties=fit_dictionary_tabular["dataset_properties"]) \
            .get_default_configuration().get_dictionary()
        preprocessor = preprocessor(**configuration, random_state=random_state)
        preprocessor.fit(fit_dictionary_tabular)
        X = preprocessor.transform(fit_dictionary_tabular)
        sklearn_preprocessor = X['feature_preprocessor']['numerical']

        # check if the fit dictionary X is modified as expected
        assert isinstance(X['feature_preprocessor'], dict)
        if isinstance(preprocessor, NoFeaturePreprocessor):
            assert sklearn_preprocessor is None, sklearn_preprocessor
            pytest.skip("Tests not relevant for {}".format(preprocessor.__class__.__name__))
        assert isinstance(sklearn_preprocessor, BaseEstimator)
        assert (X['feature_preprocessor']['categorical']) is None

        # make column transformer with returned encoder to fit on data
        column_transformer = make_column_transformer((sklearn_preprocessor,
                                                      X['dataset_properties']['numerical_columns']),
                                                     remainder='passthrough')
        column_transformer.fit(X['X_train'], X['y_train'])

        transformed = column_transformer.transform(X['X_train'])
        assert isinstance(transformed, np.ndarray)

    @flaky.flaky(max_runs=3)
    def test_pipeline_fit_include(self, fit_dictionary_tabular, preprocessor):
        """
        This test ensures that a tabular classification
        pipeline can be fit with all preprocessors
        in the include
        """

        task_type = str(fit_dictionary_tabular['dataset_properties']['task_type'])
        if (
            ("Classification" in preprocessor or preprocessor == "LibLinearSVCPreprocessor")
            and STRING_TO_TASK_TYPES[task_type] not in CLASSIFICATION_TASKS
        ):
            pytest.skip("Tests not relevant for {}".format(preprocessor.__class__.__name__))
        elif "Regression" in preprocessor and STRING_TO_TASK_TYPES[task_type] not in REGRESSION_TASKS:
            pytest.skip("Tests not relevant for {}".format(preprocessor.__class__.__name__))
        fit_dictionary_tabular['epochs'] = 1

        pipeline = TabularClassificationPipeline(
            dataset_properties=fit_dictionary_tabular['dataset_properties'],
            include={'feature_preprocessor': [preprocessor]})
        cs = pipeline.get_hyperparameter_search_space()
        config = cs.sample_configuration()
        pipeline.set_hyperparameters(config)
        try:
            pipeline.fit(fit_dictionary_tabular)
        except Exception as e:
            if (
                ("must be non-negative" in e.args[0] or "contains negative values" in e.args[0])
                and not fit_dictionary_tabular['dataset_properties']['issigned']
            ):
                pytest.skip("Failure because scaler made data nonnegative.")
            pytest.fail(f"For config {config} failed with {e}")

        # To make sure we fitted the model, there should be a
        # run summary object with accuracy
        run_summary = pipeline.named_steps['trainer'].run_summary
        assert run_summary is not None

        assert preprocessor == pipeline.named_steps['feature_preprocessor'].choice.__class__.__name__
