import numpy as np

import pytest

from sklearn.base import BaseEstimator
from sklearn.compose import make_column_transformer

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing.FastICA import FastICA
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing.KernelPCA import \
    KernelPCA
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    NoFeaturePreprocessor import NoFeaturePreprocessor
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing.Nystroem import Nystroem
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing.\
    PolynomialFeatures import PolynomialFeatures
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing.\
    PowerTransformer import PowerTransformer
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    RandomKitchenSinks import RandomKitchenSinks
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing.TruncatedSVD import \
    TruncatedSVD


@pytest.fixture(params=[TruncatedSVD(), PolynomialFeatures(), PowerTransformer(),
                        Nystroem(), FastICA(), KernelPCA(), RandomKitchenSinks(), TruncatedSVD()])
def preprocessor(request):
    return request.param


@pytest.mark.parametrize("fit_dictionary", ['fit_dictionary_numerical_only',
                                            'fit_dictionary_num_and_categorical'], indirect=True)
class TestFeaturePreprocessors():

    def test_feature_preprocessor(self, fit_dictionary, preprocessor):
        configuration = preprocessor.\
            get_hyperparameter_search_space(dataset_properties=fit_dictionary["dataset_properties"]) \
            .sample_configuration().get_dictionary()
        preprocessor = preprocessor.set_params(**configuration)
        preprocessor.fit(fit_dictionary)
        X = preprocessor.transform(fit_dictionary)
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
        column_transformer.fit(X['X_train'])
        transformed = column_transformer.transform(X['X_train'])
        assert isinstance(transformed, np.ndarray)
