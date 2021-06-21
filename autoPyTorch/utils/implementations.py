from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

import numpy as np

from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin

import torch


def get_loss_weight_strategy(loss: Type[torch.nn.Module]) -> Callable:
    """
    Utility function that returns strategy for the given loss
    Args:
        loss (Type[torch.nn.Module]): type of the loss function
    Returns:
        (Callable): Relevant Callable strategy
    """
    if loss.__name__ in LossWeightStrategyWeightedBinary.get_properties()['supported_losses']:
        return LossWeightStrategyWeightedBinary()
    elif loss.__name__ in LossWeightStrategyWeighted.get_properties()['supported_losses']:
        return LossWeightStrategyWeighted()
    else:
        raise ValueError("No strategy currently supports the given loss, {}".format(loss.__name__))


class LossWeightStrategyWeighted():
    def __call__(self, y: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy() if y.is_cuda else y.numpy()
        if isinstance(y[0], str):
            y = y.astype('float64')
        counts = np.sum(y, axis=0)
        total_weight = y.shape[0]

        if len(y.shape) > 1:
            weight_per_class = total_weight / y.shape[1]
            weights = (np.ones(y.shape[1]) * weight_per_class) / np.maximum(counts, 1)
        else:
            classes, counts = np.unique(y, axis=0, return_counts=True)
            classes, counts = classes[::-1], counts[::-1]
            weight_per_class = total_weight / classes.shape[0]
            weights = (np.ones(classes.shape[0]) * weight_per_class) / counts

        return weights

    @staticmethod
    def get_properties() -> Dict[str, Any]:
        return {'supported_losses': ['CrossEntropyLoss']}


class LossWeightStrategyWeightedBinary():
    def __call__(self, y: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy() if y.is_cuda else y.numpy()
        if isinstance(y[0], str):
            y = y.astype('float64')
        counts_one = np.sum(y, axis=0)
        counts_zero = counts_one + (-y.shape[0])
        weights = counts_zero / np.maximum(counts_one, 1)

        return np.array(weights)

    @staticmethod
    def get_properties() -> Dict[str, Any]:
        return {'supported_losses': ['BCEWithLogitsLoss']}


class MinorityCoalescing(BaseEstimator, TransformerMixin):
    """ Group together categories which occurrence is less than a specified
    minimum fraction. Coalesced categories get index of one.
    """

    def __init__(self, minimum_fraction: Optional[float] = None):
        self.minimum_fraction = minimum_fraction

    def check_X(self, X: Union[np.ndarray, sparse.csr_matrix]) -> None:
        """
        This estimator takes as input a set of features coming in a tabular fashion.
        The classes in the columns, i.e.features, are coalesced together, if the ratio of
        occurrence of each class is less than self.minimum_fraction.

        Those classes with low occurrence frequency are coalesced into a new class: -2.
        The classification pipeline shifts the classes to be positive through encoding.
        The imputation tags missing elements as -1. Then the coalescer can use -2 as a
        'coalesced' indicator.

        Failure can only happen if the pipeline failed to fit encoding/imputation.

        Args:
        X (np.ndarray):
            The input features from the user, likely transformed by an encoder and imputator.
        """
        X_data = X.data if sparse.issparse(X) else X
        if np.nanmin(X_data) <= -2:
            raise ValueError("The input features to the MinorityCoalescing "
                             "need to contain only integers greater than -2.")

    def fit(self, X: Union[np.ndarray, sparse.csr_matrix],
            y: Optional[np.ndarray] = None) -> 'MinorityCoalescing':
        """
        Trains the estimator to identify low frequency classes on the input train data.

        Args:
        X (Union[np.ndarray, sparse.csr_matrix]):
            The input features from the user, likely transformed by an encoder and imputator.
        y (Optional[np.ndarray]):
            Optional labels for the given task, not used by this estimator.
        """
        self.check_X(X)

        if self.minimum_fraction is None:
            return self

        # Remember which values should not be coalesced
        do_not_coalesce: List[Set[int]] = list()
        for column in range(X.shape[1]):
            do_not_coalesce.append(set())

            if sparse.issparse(X):
                indptr_start = X.indptr[column]
                indptr_end = X.indptr[column + 1]
                unique, counts = np.unique(
                    X.data[indptr_start:indptr_end], return_counts=True)
                colsize = indptr_end - indptr_start
            else:
                unique, counts = np.unique(X[:, column], return_counts=True)
                colsize = X.shape[0]

            for unique_value, count in zip(unique, counts):
                fraction = float(count) / colsize
                if fraction >= self.minimum_fraction:
                    do_not_coalesce[-1].add(unique_value)

        self._do_not_coalesce = do_not_coalesce
        return self

    def transform(self, X: Union[np.ndarray, sparse.csr_matrix]) -> Union[np.ndarray,
                                                                          sparse.csr_matrix]:
        """
        Coalesces categories with low frequency on the input array X.

        Args:
        X (Union[np.ndarray, sparse.csr_matrix]):
            The input features from the user, likely transformed by an encoder and imputator.
        """
        self.check_X(X)

        if self.minimum_fraction is None:
            return X

        for column in range(X.shape[1]):
            if sparse.issparse(X):
                indptr_start = X.indptr[column]
                indptr_end = X.indptr[column + 1]
                unique = np.unique(X.data[indptr_start:indptr_end])
                for unique_value in unique:
                    if unique_value not in self._do_not_coalesce[column]:
                        indptr_start = X.indptr[column]
                        indptr_end = X.indptr[column + 1]
                        X.data[indptr_start:indptr_end][
                            X.data[indptr_start:indptr_end] == unique_value] = -2
            else:
                unique = np.unique(X[:, column])
                unique_values = [unique_value for unique_value in unique
                                 if unique_value not in self._do_not_coalesce[column]]
                mask = np.isin(X[:, column], unique_values)
                # The imputer uses -1 for unknown categories
                # Then -2 means coalesced categories
                X[mask, column] = -2
        return X

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)
