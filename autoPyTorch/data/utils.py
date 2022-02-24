# Implementation used from https://github.com/automl/auto-sklearn/blob/development/autosklearn/util/data.py
import warnings
from math import floor
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
from pandas.api.types import is_float_dtype, is_numeric_dtype

from scipy.sparse import issparse, spmatrix


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
    "methods": ["precision"]
}


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
    if isinstance(dataset_compression, Mapping):
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
            # convert to int so we can directly use
            dataset_compression["memory_allocation"] = floor(memory_allocation * memory_limit)

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
    else:
        raise ValueError(
            f"Unknown type for `dataset_compression` {type(dataset_compression)}"
            f"\ndataset_compression = {dataset_compression}"
        )


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
    elif hasattr(X, 'iloc'):
        dtypes = dict(X.dtypes)

        integer_columns = []
        float_columns = []

        for col, dtype in dtypes.items():
            if is_numeric_dtype(dtype):
                if is_float_dtype(dtype):
                    float_columns.append(col)
                else:
                    integer_columns.append(col)

        if len(integer_columns) > 0:
            X[integer_columns] = X[integer_columns].apply(lambda column: pd.to_numeric(column, downcast='integer'))
        if len(float_columns) > 0:
            X[float_columns] = X[float_columns].apply(lambda column: pd.to_numeric(column, downcast='float'))
        reduced_dtypes = dict(X.dtypes)
    else:
        raise ValueError(f"Unrecognised data type of X, expected data type to "
                         f"be in (np.ndarray, spmatrix, pd.DataFrame), but got :{type(X)}")

    return X, reduced_dtypes, dtypes


def megabytes(arr: DatasetCompressionInputType) -> float:

    if isinstance(arr, np.ndarray):
        memory_in_bytes = arr.nbytes
    elif issparse(arr):
        memory_in_bytes = arr.data.nbytes
    elif hasattr(arr, 'iloc'):
        memory_in_bytes = arr.memory_usage(index=True, deep=True).sum()
    else:
        raise ValueError(f"Unrecognised data type of X, expected data type to "
                         f"be in (np.ndarray, spmatrix, pd.DataFrame) but got :{type(arr)}")

    return float(memory_in_bytes / (2**20))


def reduce_dataset_size_if_too_large(
    X: DatasetCompressionInputType,
    memory_allocation: int,
    methods: List[str] = ['precision'],
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

        methods: List[str] = ['precision']
            A list of operations that are permitted to be performed to reduce
            the size of the dataset.

            **precision**

            Reduce the precision of float types

        memory_allocation: int
            The amount of memory to allocate to the dataset. It should specify an
            absolute amount.

    Returns:
        DatasetCompressionInputType
            The reduced X if reductions were needed
    """

    for method in methods:

        if method == 'precision':
            # If the dataset is too big for the allocated memory,
            # we then try to reduce the precision if it's a high precision dataset
            if megabytes(X) > memory_allocation:
                X, reduced_dtypes, dtypes = reduce_precision(X)
                warnings.warn(
                    f'Dataset too large for allocated memory {memory_allocation}MB, '
                    f'reduced the precision from {dtypes} to {reduced_dtypes}',
                )
        else:
            raise ValueError(f"Unknown operation `{method}`")

    return X
