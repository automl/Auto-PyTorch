from typing import Any, Callable, Dict, List

import numpy as np

from sklearn.utils import check_array

import torch

import torchvision

from autoPyTorch.pipeline.components.training.data_loader.base_data_loader import BaseDataLoaderComponent


class ExpandTransform(object):
    """Expand Dimensionality so tabular transformations see
       a 2d Array
    """
    def __call__(self, data: np.ndarray) -> np.ndarray:
        if len(data.shape) <= 1:
            data = np.expand_dims(data, axis=0)
        return data


class ContractTransform(object):
    """Reverses the effect of ExpandTransform"""
    def __call__(self, data: np.ndarray) -> np.ndarray:
        # We make sure that the data is converted from
        # Object regardless of the configuration space
        # (normally the CS will convert a pd.DataFrame->np.array,
        # but the current config might be no preprocessing)
        # Also, Batch normalization expect a flattened input, so
        # we have to squeeze sklearn output which is normally (N, 1)
        data = check_array(
            data,
            force_all_finite=False,
            accept_sparse='csr',
            ensure_2d=False,
            allow_nd=True,
        )
        return np.squeeze(data)


class FeatureDataLoader(BaseDataLoaderComponent):
    """This class is an interface to the PyTorch Dataloader.

    Particularly, this data loader builds transformations for
    tabular data.

    """

    def build_transform(self, X: Dict[str, Any], mode: str) -> torchvision.transforms.Compose:
        """
        Method to build a transformation that can pre-process input data

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            mode (str): train/val/test

        Returns:
            A composition of transformations
        """

        if mode not in ['train', 'val', 'test']:
            raise ValueError("Unsupported mode provided {}. ".format(mode))

        # In the case of feature data, the options currently available
        # for transformations are:
        #   + imputer
        #   + encoder
        #   + scaler
        # This transformations apply for both train/val/test, so no
        # distinction is performed
        candidate_transformations: List[Callable] = []

        if 'test' in mode or not X['dataset_properties']['is_small_preprocess']:
            candidate_transformations.append((ExpandTransform()))
            candidate_transformations.extend(X['preprocess_transforms'])
            candidate_transformations.append((ContractTransform()))

        # Transform to tensor
        candidate_transformations.append(torch.from_numpy)

        return torchvision.transforms.Compose(candidate_transformations)

    def _check_transform_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
        """

        Makes sure that the fit dictionary contains the required transformations
        that the dataset should go through

        Args:
            X (Dict[str, Any]): Dictionary with fitted parameters. It is a message passing
                mechanism, in which during a transform, a components adds relevant information
                so that further stages can be properly fitted
        """
        if not X['dataset_properties']['is_small_preprocess'] and 'preprocess_transforms' not in X:
            raise ValueError("Cannot find the preprocess_transforms in the fit dictionary")
