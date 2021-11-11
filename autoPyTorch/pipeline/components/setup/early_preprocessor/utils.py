import copy
from typing import Any, Dict, List

import numpy as np

from sklearn.utils import check_array

import torchvision.transforms

from autoPyTorch.pipeline.components.preprocessing.base_preprocessing import autoPyTorchPreprocessingComponent


def get_preprocess_transforms(X: Dict[str, Any]) -> torchvision.transforms.Compose:
    candidate_transforms: List[autoPyTorchPreprocessingComponent] = list()
    for key, value in X.items():
        if isinstance(value, autoPyTorchPreprocessingComponent):
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
