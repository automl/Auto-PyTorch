import numpy as np
from numpy.testing import assert_array_equal


from sklearn.base import BaseEstimator
from sklearn.compose import make_column_transformer

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.variance_thresholding. \
    VarianceThreshold import VarianceThreshold


def test_variance_threshold():
    data = np.array([[1, 2, 1],
                     [7, 8, 9],
                     [4, 5, 1],
                     [11, 12, 1],
                     [17, 18, 19],
                     [14, 15, 16]])
    numerical_columns = [0, 1, 2]
    train_indices = np.array([0, 2, 3])
    test_indices = np.array([1, 4, 5])
    dataset_properties = {
        'categorical_columns': [],
        'numerical_columns': numerical_columns,
    }
    X = {
        'X_train': data[train_indices],
        'dataset_properties': dataset_properties
    }
    component = VarianceThreshold()

    component = component.fit(X)
    X = component.transform(X)
    variance_threshold = X['variance_threshold']['numerical']

    # check if the fit dictionary X is modified as expected
    assert isinstance(X['variance_threshold'], dict)
    assert isinstance(variance_threshold, BaseEstimator)

    # make column transformer with returned encoder to fit on data
    column_transformer = make_column_transformer((variance_threshold,
                                                  X['dataset_properties']['numerical_columns']),
                                                 remainder='passthrough')
    column_transformer = column_transformer.fit(X['X_train'])
    transformed = column_transformer.transform(data[test_indices])

    assert_array_equal(transformed, np.array([[7, 8],
                                              [17, 18],
                                              [14, 15]]))
