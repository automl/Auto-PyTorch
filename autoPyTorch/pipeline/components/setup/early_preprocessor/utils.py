import copy
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np

import pandas as pd

from sklearn.utils import check_array

import torchvision.transforms

from autoPyTorch.pipeline.components.preprocessing.base_preprocessing import (
    autoPyTorchPreprocessingComponent as aPTPre,
    autoPyTorchTargetPreprocessingComponent as aPTTPre
)


def get_preprocess_transforms(X: Dict[str, Any],
                              preprocess_type: Union[Type[aPTPre], Type[aPTTPre]] = aPTPre) \
        -> List[Union[Type[aPTPre], Type[aPTTPre]]]:
    candidate_transforms = []
    for key, value in X.items():
        if isinstance(value, preprocess_type):
            candidate_transforms.append(copy.deepcopy(value))

    return candidate_transforms


def preprocess(dataset: np.ndarray, transforms: torchvision.transforms.Compose,
               indices: List[int] = None) -> np.ndarray:

    composite_transforms = torchvision.transforms.Compose(transforms)
    if indices is None:
        dataset = composite_transforms(dataset)
    else:
        dataset[indices, :] = composite_transforms(np.take(dataset, indices, axis=0))
    # In case the configuration space is so that no
    # sklearn transformation is proposed, we perform
    # check array to convert object to float
    return check_array(
        dataset,
        force_all_finite=False,
        accept_sparse='csr',
        ensure_2d=False,
        allow_nd=True,
    )


def time_series_preprocess(dataset: pd.DataFrame, transforms: torchvision.transforms.Compose,
                           indices: Optional[List[int]] = None) -> pd.DataFrame:
    """
    preprocess time series data (both features and targets). Dataset should be pandas DataFrame whose index identifies
    which series the data belongs to.

    Args:
        dataset (pd.DataFrame): a dataset contains multiple series, its index identifies the series number
        transforms (torchvision.transforms.Compose): transformation applied to dataset
        indices (Optional[List[int]]): the indices that the transformer needs to work with

    Returns:

    """
    # TODO consider Numpy implementation
    composite_transforms = torchvision.transforms.Compose(transforms)
    if indices is None:
        index = dataset.index
        dataset = composite_transforms(dataset)
        dataset = pd.DataFrame(dataset, index=index)
    else:
        sub_dataset = dataset.iloc[:, indices]
        sub_dataset = composite_transforms(sub_dataset)
        dataset.iloc[:, indices] = sub_dataset
    return dataset
