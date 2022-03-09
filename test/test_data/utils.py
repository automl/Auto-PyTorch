from typing import List

import numpy as np

import pandas as pd

from scipy.sparse import spmatrix


def convert(arr, objtype):
    if objtype == np.ndarray:
        return arr
    elif objtype == list:
        return arr.tolist()
    else:
        return objtype(arr)


# Function to get the type of an obj
def dtype(obj):
    if isinstance(obj, List):
        return type(obj[0][0]) if isinstance(obj[0], List) else type(obj[0])
    elif isinstance(obj, pd.DataFrame):
        return obj.dtypes
    else:
        return obj.dtype


# Function to get the size of an object
def size(obj):
    if isinstance(obj, spmatrix):  # spmatrix doesn't support __len__
        return obj.shape[0] if obj.shape[0] > 1 else obj.shape[1]
    else:
        return len(obj)
