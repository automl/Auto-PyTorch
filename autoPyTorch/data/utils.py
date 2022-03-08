# Implementation used from https://github.com/automl/auto-sklearn/blob/development/autosklearn/util/data.py
import warnings
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast
)

import numpy as np

import pandas as pd

from scipy.sparse import issparse, spmatrix

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.utils import _approximate_mode, check_random_state
from sklearn.utils.validation import _num_samples, check_array

from autoPyTorch.data.base_target_validator import SupportedTargetTypes
from autoPyTorch.utils.common import ispandas

# TODO: TypedDict with python 3.8
#
#   When upgrading to python 3.8 as minimum version, this should be a TypedDict
#   so that mypy can identify the fields types
DatasetCompressionSpec = Dict[str, Union[int, float, List[str]]]
DatasetDTypeContainerType = Union[Type, Dict[str, Type]]
DatasetCompressionInputType = Union[np.ndarray, spmatrix, pd.DataFrame]

# Default specification for arg `dataset_compression`
default_dataset_compression_arg: DatasetCompressionSpec = {
    "memory_allocation": 0.1,
    "methods": ["precision", "subsample"]
}


class CustomStratifiedShuffleSplit(StratifiedShuffleSplit):
    """Splitter that deals with classes with too few samples"""

    def _iter_indices(self, X, y, groups=None):  # type: ignore
        n_samples = _num_samples(X)
        y = check_array(y, ensure_2d=False, dtype=None)
        n_train, n_test = _validate_shuffle_split(
            n_samples,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )

        if y.ndim == 2:
            # for multi-label y, map each distinct row to a string repr
            # using join because str(row) uses an ellipsis if len(row) > 1000
            y = np.array([" ".join(row.astype("str")) for row in y])

        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]

        class_counts = np.bincount(y_indices)

        if n_train < n_classes:
            raise ValueError(
                "The train_size = %d should be greater or "
                "equal to the number of classes = %d" % (n_train, n_classes)
            )
        if n_test < n_classes:
            raise ValueError(
                "The test_size = %d should be greater or "
                "equal to the number of classes = %d" % (n_test, n_classes)
            )

        # Find the sorted list of instances for each class:
        # (np.unique above performs a sort, so code is O(n logn) already)
        class_indices = np.split(
            np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1]
        )

        rng = check_random_state(self.random_state)

        for _ in range(self.n_splits):
            # if there are ties in the class-counts, we want
            # to make sure to break them anew in each iteration
            n_i = _approximate_mode(class_counts, n_train, rng)
            class_counts_remaining = class_counts - n_i
            t_i = _approximate_mode(class_counts_remaining, n_test, rng)
            train = []
            test = []

            # NOTE: Adapting for unique instances
            #
            #   Each list n_i, t_i represent the list of class in the
            #   training_set and test_set resepectively.
            #
            #   n_i = [100, 100, 0, 3]  # 100 of class '0', 0 of class '2'
            #   t_i = [300, 300, 1, 3]  # 300 of class '0', 1 of class '2'
            #
            #  To support unique labels such as class '2', which only has one sample
            #  between both n_i and t_i, we need to make sure that n_i has at least
            #  one sample of all classes. There is also the extra check to ensure
            #  that the sizes stay the same.
            #
            #   n_i = [ 99, 100, 1, 3]  # 100 of class '0', 0 of class '2'
            #            |       ^
            #            v       |
            #   t_i = [301, 300, 0, 3]  # 300 of class '0', 1 of class '2'
            #
            for i, class_count in enumerate(n_i):
                if class_count == 0:
                    t_i[i] -= 1
                    n_i[i] += 1

                    j = np.argmax(n_i)
                    if n_i[j] == 1:
                        warnings.warn(
                            "Can't respect size requirements for split.",
                            " The training set must contain all of the unique"
                            " labels that exist in the dataset.",
                        )
                    else:
                        n_i[j] -= 1
                        t_i[j] += 1

            for i in range(n_classes):
                permutation = rng.permutation(class_counts[i])
                perm_indices_class_i = class_indices[i].take(permutation, mode="clip")

                train.extend(perm_indices_class_i[: n_i[i]])
                test.extend(perm_indices_class_i[n_i[i]: n_i[i] + t_i[i]])

            train = rng.permutation(train)
            test = rng.permutation(test)

            yield train, test


def get_dataset_compression_mapping(
    memory_limit: int,
    dataset_compression: Union[bool, Mapping[str, Any]]
) -> Optional[DatasetCompressionSpec]:
    """
    Internal function to get value for `BaseTask._dataset_compression`
    based on the value of `dataset_compression` passed.

    If True, it returns the default_dataset_compression_arg. In case
    of a mapping, it is validated and returned as a `DatasetCompressionSpec`.

    If False, it returns None.

    Args:
        memory_limit (int):
            memory limit of the current search.
        dataset_compression (Union[bool, Mapping[str, Any]]):
            mapping passed to the `search` function.

    Returns:
        Optional[DatasetCompressionSpec]:
            Validated data compression spec or None.
    """
    dataset_compression_mapping: Optional[Mapping[str, Any]] = None

    if not isinstance(dataset_compression, bool):
        dataset_compression_mapping = dataset_compression
    elif dataset_compression:
        dataset_compression_mapping = default_dataset_compression_arg

    if dataset_compression_mapping is not None:
        dataset_compression_mapping = validate_dataset_compression_arg(
            dataset_compression_mapping, memory_limit=memory_limit)

    return dataset_compression_mapping


def validate_dataset_compression_arg(
    dataset_compression: Mapping[str, Any],
    memory_limit: int
) -> DatasetCompressionSpec:
    """Validate and return a correct dataset_compression argument

    The returned value can be safely used with `reduce_dataset_size_if_too_large`.

    Args:
        dataset_compression: Mapping[str, Any]
            The argumnents to validate

    Returns:
        DatasetCompressionSpec
            The validated and correct dataset compression spec
    """
    if not isinstance(dataset_compression, Mapping):
        raise ValueError(
            f"Unknown type for `dataset_compression` {type(dataset_compression)}"
            f"\ndataset_compression = {dataset_compression}"
        )

    # Fill with defaults if they don't exist
    dataset_compression = {
        **default_dataset_compression_arg,
        **dataset_compression
    }

    # Must contain known keys
    if set(dataset_compression.keys()) != set(default_dataset_compression_arg.keys()):
        raise ValueError(
            f"Unknown key in dataset_compression, {list(dataset_compression.keys())}."
            f"\nPossible keys are {list(default_dataset_compression_arg.keys())}"
        )

    memory_allocation = dataset_compression["memory_allocation"]

    # "memory_allocation" must be float or int
    if not (isinstance(memory_allocation, float) or isinstance(memory_allocation, int)):
        raise ValueError(
            "key 'memory_allocation' must be an `int` or `float`"
            f"\ntype = {memory_allocation}"
            f"\ndataset_compression = {dataset_compression}"
        )

    # "memory_allocation" if absolute, should be > 0 and < memory_limit
    if isinstance(memory_allocation, int) and not (0 < memory_allocation < memory_limit):
        raise ValueError(
            f"key 'memory_allocation' if int must be in (0, memory_limit={memory_limit})"
            f"\nmemory_allocation = {memory_allocation}"
            f"\ndataset_compression = {dataset_compression}"
        )

    # "memory_allocation" must be in (0,1) if float
    if isinstance(memory_allocation, float):
        if not (0.0 < memory_allocation < 1.0):
            raise ValueError(
                "key 'memory_allocation' if float must be in (0, 1)"
                f"\nmemory_allocation = {memory_allocation}"
                f"\ndataset_compression = {dataset_compression}"
            )
        # convert to required memory so we can directly use
        dataset_compression["memory_allocation"] = memory_allocation * memory_limit

    # "methods" must be non-empty sequence
    if (
        not isinstance(dataset_compression["methods"], Sequence)
        or len(dataset_compression["methods"]) <= 0
    ):
        raise ValueError(
            "key 'methods' must be a non-empty list"
            f"\nmethods = {dataset_compression['methods']}"
            f"\ndataset_compression = {dataset_compression}"
        )

    # "methods" must contain known methods
    if any(
        method not in cast(Sequence, default_dataset_compression_arg["methods"])  # mypy
        for method in dataset_compression["methods"]
    ):
        raise ValueError(
            f"key 'methods' can only contain {default_dataset_compression_arg['methods']}"
            f"\nmethods = {dataset_compression['methods']}"
            f"\ndataset_compression = {dataset_compression}"
        )

    return cast(DatasetCompressionSpec, dataset_compression)


class _DtypeReductionMapping(Mapping):
    """
    Unfortuantly, mappings compare by hash(item) and not the __eq__ operator
    between the key and the item.

    Hence we wrap the dict in a Mapping class and implement our own __getitem__
    such that we do use __eq__ between keys and query items.

    >>> np.float32 == dtype('float32') # True, they are considered equal
    >>>
    >>> mydict = { np.float32: 'hello' }
    >>>
    >>> # Equal by __eq__ but dict operations fail
    >>> np.dtype('float32') in mydict # False
    >>> mydict[dtype('float32')]  # KeyError

    This mapping class fixes that supporting the `in` operator as well as `__getitem__`

    >>> reduction_mapping = _DtypeReductionMapping()
    >>>
    >>> reduction_mapping[np.dtype('float64')] # np.float32
    >>> np.dtype('float32') in reduction_mapping # True
    """

    # Information about dtype support
    _mapping: Dict[type, type] = {
        np.float32: np.float32,
        np.float64: np.float32,
        np.int32: np.int32,
        np.int64: np.int32
    }

    # In spite of the names, np.float96 and np.float128
    # provide only as much precision as np.longdouble,
    # that is, 80 bits on most x86 machines and 64 bits
    # in standard Windows builds.
    _mapping.update({getattr(np, s): np.float64 for s in ['float96', 'float128'] if hasattr(np, s)})

    @classmethod
    def __getitem__(cls, item: type) -> type:
        for k, v in cls._mapping.items():
            if k == item:
                return v
        raise KeyError(item)

    @classmethod
    def __iter__(cls) -> Iterator[type]:
        return iter(cls._mapping.keys())

    @classmethod
    def __len__(cls) -> int:
        return len(cls._mapping)


reduction_mapping = _DtypeReductionMapping()
supported_precision_reductions = list(reduction_mapping)


def reduce_precision(
    X: DatasetCompressionInputType
) -> Tuple[DatasetCompressionInputType, DatasetDTypeContainerType, DatasetDTypeContainerType]:
    """ Reduce the precision of a dataset containing floats or ints

    Note:
        For dataframe, the column's precision is reduced using pd.to_numeric.

    Args:
        X (DatasetCompressionInputType):
            The data to reduce precision of.

    Returns:
        Tuple[DatasetCompressionInputType, DatasetDTypeContainerType, DatasetDTypeContainerType]
            Returns the reduced data X along with the dtypes it and the dtypes it was reduced to.
    """
    reduced_dtypes: Optional[DatasetDTypeContainerType] = None
    if isinstance(X, np.ndarray) or issparse(X):
        dtypes = X.dtype
        if X.dtype not in supported_precision_reductions:
            raise ValueError(f"X.dtype = {X.dtype} not equal to any supported"
                             f" {supported_precision_reductions}")
        reduced_dtypes = reduction_mapping[X.dtype]
        X = X.astype(reduced_dtypes)

    elif ispandas(X):
        dtypes = dict(X.dtypes)

        col_names = X.dtypes.index

        float_cols = col_names[[dt.name.startswith("float") for dt in X.dtypes.values]]
        int_cols = col_names[[dt.name.startswith("int") for dt in X.dtypes.values]]
        X[int_cols] = X[int_cols].apply(lambda column: pd.to_numeric(column, downcast='integer'))
        X[float_cols] = X[float_cols].apply(lambda column: pd.to_numeric(column, downcast='float'))

        reduced_dtypes = dict(X.dtypes)
    else:
        raise ValueError(f"Unrecognised data type of X, expected data type to "
                         f"be in (np.ndarray, spmatrix, pd.DataFrame), but got :{type(X)}")

    return X, reduced_dtypes, dtypes


def subsample(
    X: DatasetCompressionInputType,
    is_classification: bool,
    sample_size: Union[float, int],
    y: Optional[SupportedTargetTypes] = None,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> Tuple[DatasetCompressionInputType, SupportedTargetTypes]:
    """Subsamples data returning the same type as it recieved.

    If `is_classification`, we split using a stratified shuffle split which
    preserves unique labels in the training set.

    NOTE:
    It's highly unadvisable to use lists here. In order to preserve types,
    we convert to a numpy array and then back to a list.

    NOTE2:
    Interestingly enough, StratifiedShuffleSplut and descendants don't support
    sparse `y` in `split(): _check_array` call. Hence, neither do we.

    Args:
        X: DatasetCompressionInputType
            The X's to subsample
        y: SupportedTargetTypes
            The Y's to subsample
        is_classification: bool
            Whether this is classification data or regression data. Required for
            knowing how to split.
        sample_size: float | int
            If float, percentage of data to take otherwise if int, an absolute
            count of samples to take.
        random_state: int | RandomState = None
            The random state to pass to the splitted

    Returns:
        (DatasetCompressionInputType, SupportedTargetTypes)
            The X and y subsampled according to sample_size
    """

    if isinstance(X, List):
        X = np.asarray(X)
    if isinstance(y, List):
        y = np.asarray(y)

    if is_classification and y is not None:
        splitter = CustomStratifiedShuffleSplit(
            train_size=sample_size, random_state=random_state
        )
        indices_to_keep, _ = next(splitter.split(X=X, y=y))
        X, y = _subsample_by_indices(X, y, indices_to_keep)

    elif y is None:
        X, _ = train_test_split(  # type: ignore
            X,
            train_size=sample_size,
            random_state=random_state,
        )
    else:
        X, _, y, _ = train_test_split(  # type: ignore
            X,
            y,
            train_size=sample_size,
            random_state=random_state,
        )

    return X, y


def _subsample_by_indices(
    X: DatasetCompressionInputType,
    y: SupportedTargetTypes,
    indices_to_keep: np.ndarray
) -> Tuple[DatasetCompressionInputType, SupportedTargetTypes]:
    """
    subsample data by given indices
    """
    if ispandas(X):
        idxs = X.index[indices_to_keep]
        X = X.loc[idxs]
    else:
        X = X[indices_to_keep]

    if ispandas(y):
        # Ifnoring types as mypy does not infer y as dataframe.
        idxs = y.index[indices_to_keep]  # type: ignore [index]
        y = y.loc[idxs]  # type: ignore [union-attr]
    else:
        y = y[indices_to_keep]
    return X, y


def megabytes(arr: DatasetCompressionInputType) -> float:

    if isinstance(arr, np.ndarray):
        memory_in_bytes = arr.nbytes
    elif issparse(arr):
        memory_in_bytes = arr.data.nbytes
    elif ispandas(arr):
        memory_in_bytes = arr.memory_usage(index=True, deep=True).sum()
    else:
        raise ValueError(f"Unrecognised data type of X, expected data type to "
                         f"be in (np.ndarray, spmatrix, pd.DataFrame) but got :{type(arr)}")

    return float(memory_in_bytes / (2**20))


def reduce_dataset_size_if_too_large(
    X: DatasetCompressionInputType,
    memory_allocation: Union[int, float],
    is_classification: bool,
    random_state: Union[int, np.random.RandomState],
    y: Optional[SupportedTargetTypes] = None,
    methods: List[str] = ['precision', 'subsample'],
) -> DatasetCompressionInputType:
    f""" Reduces the size of the dataset if it's too close to the memory limit.

    Follows the order of the operations passed in and retains the type of its
    input.

    Precision reduction will only work on the following data types:
    -   {supported_precision_reductions}

    Precision reduction will only perform one level of precision reduction.
    Technically, you could supply multiple rounds of precision reduction, i.e.
    to reduce np.float128 to np.float32 you could use `methods = ['precision'] * 2`.

    However, if that's the use case, it'd be advised to simply use the function
    `autoPyTorch.data.utils.reduce_precision`.

    Args:
        X: DatasetCompressionInputType
            The features of the dataset.

        methods (List[str] = ['precision', 'subsample']):
            A list of operations that are permitted to be performed to reduce
            the size of the dataset.

            **precision**

                Reduce the precision of float types

            **subsample**
                Reduce the amount of samples of the dataset such that it fits into the allocated
                memory. Ensures stratification and that unique labels are present


        memory_allocation (Union[int, float]):
            The amount of memory to allocate to the dataset. It should specify an
            absolute amount.

    Returns:
        DatasetCompressionInputType
            The reduced X if reductions were needed
    """

    for method in methods:
        if megabytes(X) <= memory_allocation:
            break

        if method == 'precision':
            # If the dataset is too big for the allocated memory,
            # we then try to reduce the precision if it's a high precision dataset
            X, reduced_dtypes, dtypes = reduce_precision(X)
            warnings.warn(
                f'Dataset too large for allocated memory {memory_allocation}MB, '
                f'reduced the precision from {dtypes} to {reduced_dtypes}',
            )
        elif method == "subsample":
            # If the dataset is still too big such that we couldn't fit
            # into the allocated memory, we subsample it so that it does

            n_samples_before = X.shape[0]
            sample_percentage = memory_allocation / megabytes(X)

            # NOTE: type ignore
            #
            # Tried the generic `def subsample(X: T) -> T` approach but it was
            # failing elsewhere, keeping it simple for now
            X, y = subsample(  # type: ignore
                X,
                y=y,
                sample_size=sample_percentage,
                is_classification=is_classification,
                random_state=random_state,
            )

            n_samples_after = X.shape[0]
            warnings.warn(
                f"Dataset too large for allocated memory {memory_allocation}MB,"
                f" reduced number of samples from {n_samples_before} to"
                f" {n_samples_after}."
            )

        else:
            raise ValueError(f"Unknown operation `{method}`")

    return X, y
