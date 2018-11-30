import numpy as np

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

class DataConverter(object):
    def __init__(self, is_classification=None,
                 numerical_min_unique_values=3,
                 force_numerical=None,
                 force_categorical=None,
                 is_multilabel=None):
        """
        Initialize the data_converter.
        
        Arguments:
            X: NxM Matrix: N records with M features
            Y: Vector of N labels.
            is_classification: specifies, if it is a classification problem. None for autodetect.
            numerical_min_unique_values: minimum number of unique values for a numerical feature.
                A feature will be interpreted as categorical, if it has less.
            force_numerical: Array of feature indices, which schould be treated as numerical.
            force_categorical: Array of feature indices, which should be trated as categorical.
            is_multilabel: True, if multivariable regression / multilabel classification
        """
        self.is_classification = is_classification
        self.numerical_min_unique_values= numerical_min_unique_values
        self.force_numerical = force_numerical or []
        self.force_categorical = force_categorical or []
        self.is_multilabel = is_multilabel

    def convert(self, X, Y):
        """
        Convert the data.
        
        Returns:
            X_result: The converted X matrix, using one-hot-encoding for categorical features.
            Y_result: The converted Y vector, using integers for categorical featues.
            is_classification: If the problem is a classification problem.
        """
        X_result, categorical = self.convert_matrix(X, self.force_categorical, self.force_numerical)

        if len(Y.shape) == 1 or Y.shape[1] == 1:
            Y_result, Y_categorical = self.convert_matrix(Y.reshape(-1, 1),
                                                                [0] if self.is_classification else [],
                                                                [0] if self.is_classification == False else [])
            self.is_classification = np.any(Y_categorical)
            assert self.is_multilabel != True, "multilabel specified, but only 1-dim output vector given"
            self.is_multilabel = False
        else:
            Y_result = self.check_multi_dim_output(Y)

        if Y_result.shape[1] == 1:
            Y_result = np.reshape(Y_result, (-1, ))
        elif not self.is_multilabel and self.is_classification:
            Y_result = np.argmax(Y_result, axis=1)
        return X_result, Y_result, self.is_classification, self.is_multilabel, categorical

    def convert_matrix(self, matrix, force_categorical, force_numerical):
        """
        Covert the matrix in a matrix of floats.
        Use one-hot-encoding for categorical features.
        Features are categorical if at least one item is a string or it has more
            unique values than specified numerical_min_unique_values
            or it is listed in force_categorical.
        
        Arguments:
            matrix: The matrix to convert.
            force_cateogrical: The list of column indizes, which should be categorical.
            force_numerical: The list of column indizes, which should be numerical.
            
        Result:
            result: the converted matrix
            categorical: boolean vector, that specifies which columns are categorical
        """
        num_rows = len(matrix)
        is_categorical = []
        len_values_and_indices = []
        result_width = 0
        
        # iterate over the columns and get some data
        for i in range(matrix.shape[1]):
            
            # check if it is categorical or numerical
            matrix_column = matrix[0:num_rows, i]
            if matrix.dtype == np.dtype("object"):
                values_occurred = dict()
                values = []
                indices = []
                for v in matrix_column:
                    if v not in values_occurred:
                        values_occurred[v] = len(values)
                        values.append(v)
                    indices.append(values_occurred[v])
                indices = np.array(indices)
                values = np.array(values, dtype=object)
                nan_indices = np.array([i for i, n in enumerate(matrix_column) if n == np.nan])
                valid_value_indices = np.array([i for i, n in enumerate(values) if n != np.nan])
            else:
                values, indices = np.unique(matrix_column, return_inverse=True)
                nan_indices = np.argwhere(np.isnan(matrix_column)).flatten()
                valid_value_indices = np.argwhere(~np.isnan(values)).flatten()

            # check for missing values
            # nan values are additional category in categorical features
            if len(nan_indices) > 0:
                values = np.append(values[valid_value_indices], np.nan)
                indices[nan_indices] = values.shape[0] - 1         

            len_values_and_indices.append((len(values), indices))
            if len(values) == 1:
                is_categorical.append(None)
            elif i in force_categorical or i not in force_numerical and (
                    len(values) < self.numerical_min_unique_values or
                    any(type(value) is str for value in values)):
                # column is categorical
                is_categorical.append(True)
                result_width += 1
            else:
                # column is numerical
                is_categorical.append(False)
                result_width += 1

        # fill the result
        result = np.zeros(shape=(num_rows, result_width), dtype='float32', order='F')
        j = 0
        for i, is_cat in enumerate(is_categorical):
            len_values, indices = len_values_and_indices[i]
            if len_values == 1:
                continue
            if is_cat:
                # column is categorical: convert to int
                result[:, j] = indices
                j += 1
            else:
                # column is numerical
                result[:, j] = matrix[:, i]
                j += 1

        return result.astype('float32', copy=False), [x for x in is_categorical if x is not None]
    
    
    def check_multi_dim_output(self, Y):
        Y = Y.astype('float32', copy=False)
        unique = np.unique(Y)
        if len(unique) == 2 and self.is_classification != False and 0 in unique and 1 in unique:
            self.is_classification = True
            if np.all(np.sum(Y, axis=1) == 1) and self.is_multilabel != True:
                self.is_multilabel = False
            else:
                self.is_multilabel = True
        else:
            assert not np.any(np.isnan(Y)), "NaN in Y"
            self.is_classification = False
        return Y