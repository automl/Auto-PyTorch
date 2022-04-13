from test.test_pipeline.components.preprocessing.base import TabularPipeline

import numpy as np

import pytest

from scipy.sparse import csr_matrix

from sklearn.compose import ColumnTransformer

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.TabularColumnTransformer import (
    TabularColumnTransformer
)


@pytest.mark.parametrize("fit_dictionary_tabular", ['classification_numerical_only',
                                                    'classification_categorical_only',
                                                    'classification_numerical_and_categorical'], indirect=True)
class TestTabularTransformer:
    def test_tabular_preprocess(self, fit_dictionary_tabular):
        pipeline = TabularPipeline(dataset_properties=fit_dictionary_tabular['dataset_properties'])
        pipeline = pipeline.fit(fit_dictionary_tabular)
        X = pipeline.transform(fit_dictionary_tabular)
        column_transformer = X['tabular_transformer']

        # check if transformer was added to fit dictionary
        assert 'tabular_transformer' in X.keys()
        # check if transformer is of expected type
        # In this case we expect the tabular transformer not the actual column transformer
        # as the later is not callable and runs into error in the compose transform
        assert isinstance(column_transformer, TabularColumnTransformer)

        data = column_transformer.preprocessor.fit_transform(X['X_train'])
        assert isinstance(data, np.ndarray)

        # Make sure no columns are unintentionally dropped after preprocessing
        if len(fit_dictionary_tabular['dataset_properties']["numerical_columns"]) == 0:
            categorical_pipeline = column_transformer.preprocessor.named_transformers_['categorical_pipeline']
            categorical_data = categorical_pipeline.transform(X['X_train'])
            assert data.shape[1] == categorical_data.shape[1]
        elif len(fit_dictionary_tabular['dataset_properties']["categorical_columns"]) == 0:
            numerical_pipeline = column_transformer.preprocessor.named_transformers_['numerical_pipeline']
            numerical_data = numerical_pipeline.transform(X['X_train'])
            assert data.shape[1] == numerical_data.shape[1]

    def test_sparse_data(self, fit_dictionary_tabular):

        X = np.random.binomial(1, 0.1, (100, 2000))
        sparse_X = csr_matrix(X)
        y = np.random.randint(0, 1, 100)
        numerical_columns = list(range(2000))
        categorical_columns = []
        train_indices = np.array(range(50))
        dataset_properties = dict(numerical_columns=numerical_columns,
                                  categorical_columns=categorical_columns,
                                  categories=[],
                                  issparse=True)
        X = {
            'X_train': sparse_X,
            'y_train': y,
            'train_indices': train_indices,
            'dataset_properties': dataset_properties
        }

        pipeline = TabularPipeline(dataset_properties=dataset_properties)

        pipeline = pipeline.fit(X)
        X = pipeline.transform(X)
        column_transformer = X['tabular_transformer']

        # check if transformer was added to fit dictionary
        assert 'tabular_transformer' in X.keys()
        # check if transformer is of expected type
        assert isinstance(column_transformer.preprocessor, ColumnTransformer)

        data = column_transformer.preprocessor.fit_transform(X['X_train'])
        assert isinstance(data, csr_matrix)
