from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

import numpy as np

import torch
from torch.utils.data import Dataset, TensorDataset

import torchvision.transforms
from torchvision.transforms import functional as TF

from autoPyTorch.constants import (
    CLASSIFICATION_OUTPUTS,
    CLASSIFICATION_TASKS,
    IMAGE_CLASSIFICATION,
    IMAGE_REGRESSION,
    REGRESSION_OUTPUTS,
    STRING_TO_OUTPUT_TYPES,
    STRING_TO_TASK_TYPES,
    TASK_TYPES_TO_STRING,
)
from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
    HoldoutValTypes,
    NoResamplingStrategyTypes
)

IMAGE_DATASET_INPUT = Union[Dataset, Tuple[Union[np.ndarray, List[str]], np.ndarray]]


class ImageDataset(BaseDataset):
    """
    Dataset class for images used in AutoPyTorch
    Args:
        train (Union[Dataset, Tuple[Union[np.ndarray, List[str]], np.ndarray]]):
            training data
        val (Union[Dataset, Tuple[Union[np.ndarray, List[str]], np.ndarray]]):
            validation data
        test (Union[Dataset, Tuple[Union[np.ndarray, List[str]], np.ndarray]]):
            testing data
        resampling_strategy (Union[CrossValTypes, HoldoutValTypes, NoResamplingStrategyTypes]),
            (default=HoldoutValTypes.holdout_validation):
            strategy to split the training data.
        resampling_strategy_args (Optional[Dict[str, Any]]): arguments
            required for the chosen resampling strategy. If None, uses
            the default values provided in DEFAULT_RESAMPLING_PARAMETERS
            in ```datasets/resampling_strategy.py```.
        shuffle:  Whether to shuffle the data before performing splits
        seed (int: default=1): seed to be used for reproducibility.
        train_transforms (Optional[torchvision.transforms.Compose]):
            Additional Transforms to be applied to the training data
        val_transforms (Optional[torchvision.transforms.Compose]):
            Additional Transforms to be applied to the validation/test data
    """
    def __init__(self,
                 train: IMAGE_DATASET_INPUT,
                 val: Optional[IMAGE_DATASET_INPUT] = None,
                 test: Optional[IMAGE_DATASET_INPUT] = None,
                 resampling_strategy: Union[CrossValTypes,
                                            HoldoutValTypes,
                                            NoResamplingStrategyTypes] = HoldoutValTypes.holdout_validation,
                 resampling_strategy_args: Optional[Dict[str, Any]] = None,
                 shuffle: Optional[bool] = True,
                 seed: Optional[int] = 42,
                 train_transforms: Optional[torchvision.transforms.Compose] = None,
                 val_transforms: Optional[torchvision.transforms.Compose] = None,
                 ):
        _check_image_inputs(train=train, val=val)
        train = _create_image_dataset(data=train)
        if val is not None:
            val = _create_image_dataset(data=val)
        if test is not None:
            test = _create_image_dataset(data=test)
        self.mean, self.std = _calc_mean_std(train=train)

        super().__init__(train_tensors=train, val_tensors=val, test_tensors=test, shuffle=shuffle,
                         resampling_strategy=resampling_strategy, resampling_strategy_args=resampling_strategy_args,
                         seed=seed,
                         train_transforms=train_transforms,
                         val_transforms=val_transforms,
                         )
        if self.output_type is not None:
            if STRING_TO_OUTPUT_TYPES[self.output_type] in CLASSIFICATION_OUTPUTS:
                self.task_type = TASK_TYPES_TO_STRING[IMAGE_CLASSIFICATION]
            elif STRING_TO_OUTPUT_TYPES[self.output_type] in REGRESSION_OUTPUTS:
                self.task_type = TASK_TYPES_TO_STRING[IMAGE_REGRESSION]
            else:
                raise ValueError("Output type not currently supported ")
        else:
            raise ValueError("Task type not currently supported ")
        if STRING_TO_TASK_TYPES[self.task_type] in CLASSIFICATION_TASKS:
            self.num_classes: int = len(np.unique(self.train_tensors[1]))


def _calc_mean_std(train: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    calculates channel wise mean of the dataset
    Args:
        train (Dataset): dataset

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (mean, std)
    """
    mean = torch.zeros((3,), dtype=torch.float)
    var = torch.zeros((3,), dtype=torch.float)
    for i in range(len(train)):
        v, m = torch.var_mean(train[i][0])  # 0 used to index images
        mean += m
        var += v
    mean /= len(train)
    var /= len(var)
    std = torch.sqrt(var)
    return mean, std


def _check_image_inputs(train: IMAGE_DATASET_INPUT,
                        val: Optional[IMAGE_DATASET_INPUT] = None) -> None:
    """
    Performs sanity checks on the given data
    Args:
        train (Union[Dataset, Tuple[Union[np.ndarray, List[str]], np.ndarray]]):
            training data
        val (Union[Dataset, Tuple[Union[np.ndarray, List[str]], np.ndarray]]):
            validation data
    Returns:
        None
    """
    if not isinstance(train, Dataset):
        if len(train[0]) != len(train[1]):
            raise ValueError(
                f"expected train inputs to have the same length, but got lengths {len(train[0])} and {len(train[1])}")
        if val is not None:
            if len(val[0]) != len(val[1]):
                raise ValueError(
                    f"expected val inputs to have the same length, but got lengths {len(train[0])} and {len(train[1])}")


def _create_image_dataset(data: IMAGE_DATASET_INPUT) -> Dataset:
    """
    Creates a torch.utils.data.Dataset from different types of data mentioned
    Args:
        data (Union[Dataset, Tuple[Union[np.ndarray, List[str]], np.ndarray]]):
            data
    Returns:
        (Dataset): torch.utils.data.Dataset object of the given data
    """
    # if user already provided a dataset, use it
    if isinstance(data, Dataset):
        return data
    # if user provided list of file paths, create a file path dataset
    if isinstance(data[0], list):
        return _FilePathDataset(file_paths=data[0], targets=data[1])
    # if user provided the images as numpy tensors use them directly
    else:
        return TensorDataset(torch.tensor(data[0]), torch.tensor(data[1]))


class _FilePathDataset(Dataset):
    """
    Internal class used to handle data from file paths
    Args:
        file_paths (List[str]): paths of images
        targets (np.ndarray): targets
    """
    def __init__(self, file_paths: List[str], targets: np.ndarray):
        self.file_paths = file_paths
        self.targets = targets

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with open(self.file_paths[index], "rb") as f:
            img = Image.open(f).convert("RGB")
        return TF.to_tensor(img), torch.tensor(self.targets[index])

    def __len__(self) -> int:
        return len(self.file_paths)
