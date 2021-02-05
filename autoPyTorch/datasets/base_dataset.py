from abc import ABCMeta
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Callable, cast
from typing import NamedTuple

import numpy as np

from scipy.sparse import issparse

from sklearn.utils.multiclass import type_of_target

from torch.utils.data import Dataset, Subset

import torchvision

from autoPyTorch.datasets.resampling_strategy import (
    CrossValFuncs,
    CrossValTypes,
    CrossValParameters,
    HoldOutFuncs,
    HoldOutTypes,
    HoldOutParameters
)
from autoPyTorch.utils.common import FitRequirement, hash_array_or_matrix, BaseNamedTuple

BaseDatasetType = Union[Tuple[np.ndarray, np.ndarray], Dataset]
SplitFunc = Callable[[int, np.ndarray, Any], List[Tuple[np.ndarray, np.ndarray]]]


def check_valid_data(data: Any) -> None:
    if not all(hasattr(data, attr) for attr in ['__getitem__', '__len__']):
        raise ValueError(
            'The specified Data for Dataset must have both __getitem__ and __len__ attribute.')


def type_check(train_tensors: BaseDatasetType, val_tensors: Optional[BaseDatasetType] = None) -> None:
    for train_tensor in train_tensors:
        check_valid_data(train_tensor)
    if val_tensors is not None:
        for val_tensor in val_tensors:
            check_valid_data(val_tensor)


class TransformSubset(Subset):
    """
    Because the BaseDataset contains all the data (train/val/test), the transformations
    have to be applied with some directions. That is, if yielding train data,
    we expect to apply train transformation (which have augmentations exclusively).

    We achieve so by adding a train flag to the pytorch subset
    """
    def __init__(self, dataset: Dataset, indices: Sequence[int], train: bool) -> None:
        self.dataset = dataset
        self.indices = indices
        self.train = train

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.dataset.__getitem__(self.indices[idx], self.train)


class _DatasetSpecificProperties(BaseNamedTuple, NamedTuple):
    """TODO: doc-string"""
    task_type: Optional[str]
    output_type: str
    issparse: bool
    input_shape: Tuple[int]
    output_shape: Tuple[int]
    num_classes: Optional[int]


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        train_tensors: BaseDatasetType,
        dataset_name: Optional[str] = None,
        val_tensors: Optional[BaseDatasetType] = None,
        test_tensors: Optional[BaseDatasetType] = None,
        splitting_type: Union[CrossValTypes, HoldOutTypes] = HoldOutTypes.holdout_validation,
        splitting_params: Optional[Dict[str, Any]] = None,
        shuffle: Optional[bool] = True,
        random_state: Optional[int] = 42,
        train_transforms: Optional[torchvision.transforms.Compose] = None,
        val_transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        """
        Base class for datasets used in AutoPyTorch
        Args:
            train_tensors (A tuple of objects that have __len__ and __getitem__ attribute):
                training data (A tuple of training features and labels)
            dataset_name (str): name of the dataset, used as experiment name.
            val_tensors (An optional tuple of objects that have __len__ and __getitem__ attribute):
                validation data (A tuple of validation features and labels)
            test_tensors (An optional tuple of objects that have __len__ and __getitem__ attribute):
                test data (A tuple of test features and labels)
            
            TODO: resampling_strategy, resampling_strategy_args
            resampling_strategy (Union[CrossValTypes, HoldOutTypes]),
                (default=HoldOutTypes.holdout_validation):
                strategy to split the training data.
            resampling_strategy_args (Optional[Dict[str, Any]]): 
                arguments required for the chosen resampling strategy. 
                If None, uses the default values provided in DEFAULT_RESAMPLING_PARAMETERS
                in ```datasets/resampling_strategy.py```.

            shuffle:  Whether to shuffle the data before performing splits
            seed (int), (default=1): seed to be used for reproducibility.
            train_transforms (Optional[torchvision.transforms.Compose]):
                Additional Transforms to be applied to the training data
            val_transforms (Optional[torchvision.transforms.Compose]):
                Additional Transforms to be applied to the validation/test data
        """

        self.dataset_name = dataset_name if dataset_name is not None \
            else hash_array_or_matrix(train_tensors[0])

        if not hasattr(train_tensors[0], 'shape'):
            type_check(train_tensors, val_tensors)

        self.train_tensors, self.val_tensors, self.test_tensors = train_tensors, val_tensors, test_tensors
        self.train_transform, self.val_transform = train_transforms, val_transforms

        self.random_state, self.shuffle = random_state, shuffle
        self.rng = np.random.RandomState(seed=self.random_state)

        """SHUHEI TODO: move to HoldOut- and CrossVal- Parameters"""
        # Dict[str, SplitFunc]
        self.cross_validators = CrossValFuncs.get_cross_validators(*CrossValTypes)
        self.holdout_validators = HoldOutFuncs.get_holdout_validators(*HoldOutTypes)

        self.splitting_type, self.splitting_params = splitting_type, splitting_params
        self.convert_splitting_prams_to_namedtuple()
        self.splits = self.get_splits()

        self.task_type: Optional[str] = None
        self.issparse: bool = issparse(self.train_tensors[0])
        self.input_shape: Tuple[int] = train_tensors[0].shape[1:]
        self.num_classes: Optional[int] = None

        if len(train_tensors) == 2 and train_tensors[1] is not None:
            self.output_type: str = type_of_target(self.train_tensors[1])
            self.output_shape: int = train_tensors[1].shape[1] if len(train_tensors[1].shape) == 2 else 1

        # TODO: Look for a criteria to define small enough to preprocess
        self.is_small_preprocess = True
    
    def convert_splitting_prams_to_namedtuple(self):
        if not isinstance(self.splitting_params, dict) and self.splitting_params is not None:
            raise TypeError(f"splitting_params must be dict or None, but got {type(self.splitting_params)}")

        self.splitting_params = {} if self.splitting_params is None else self.splitting_params

        if isinstance(self.splitting_type, HoldOutTypes):
            self.splitting_params = HoldOutParameters(**self.splitting_params,
                                                      random_state=self.random_state)
        elif isinstance(self.splitting_type, CrossValTypes):
            self.splitting_params = CrossValParameters(**self.splitting_params,
                                                       random_state=self.random_state)
        else:
            raise ValueError(f"splitting_type {self.splitting_type} is not supported.")


    def update_transform(self, transform: Optional[torchvision.transforms.Compose],
                         train: bool = True) -> 'BaseDataset':
        """
        During the pipeline execution, the pipeline object might propose transformations
        as a product of the current pipeline configuration being tested.

        This utility allows to return a self with the updated transformation, so that
        a dataloader can yield this dataset with the desired transformations

        Args:
            transform (torchvision.transforms.Compose): 
                The transformations proposed by the current pipeline
            train (bool):
                Whether to update the train or validation transform

        Returns:
            self: A copy of the update pipeline
        """
        if train:
            self.train_transform = transform
        else:
            self.val_transform = transform
        return self

    def __getitem__(self, index: int, train: bool = True) -> Tuple[np.ndarray, ...]:
        """
        The base dataset uses a Subset of the data. Nevertheless, the base dataset expect
        both validation and test data to be present in the same dataset, which motivated the
        need to dynamically give train/test data with the __getitem__ command.

        This method yields a datapoint of the whole data (after a Subset has selected a given
        item, based on the resampling strategy) and applies a train/testing transformation, if any.

        Args:
            index (int): what element to yield from all the train/test tensors
            train (bool): Whether to apply a train or test transformation, if any

        Returns:
            A transformed single point prediction
        """

        X = self.train_tensors[0].iloc[[index]] if hasattr(self.train_tensors[0], 'loc') \
            else self.train_tensors[0][index]

        if self.train_transform is not None and train:
            X = self.train_transform(X)
        elif self.val_transform is not None and not train:
            X = self.val_transform(X)

        # In case of prediction, the targets are not provided
        Y = self.train_tensors[1][index] if self.train_tensors[1] is not None \
            else None

        return X, Y

    def __len__(self) -> int:
        return self.train_tensors[0].shape[0]

    def _get_indices(self) -> np.ndarray:
        return self.rng.permutation(len(self)) if self.shuffle \
            else np.arange(len(self))

    def get_splits(self) -> List[Tuple[List[int], List[int]]]:
        """
        Creates a set of splits based on a resampling strategy provided

        Returns
            (List[Tuple[List[int], List[int]]]): splits in the [train_indices, val_indices] format
        """

        stratify = self.train_tensors[-1] if self.splitting_type.is_stratified() else None

        """TODO: Think about the usage of validation data. It is not used now."""
        if isinstance(self.splitting_type, CrossValTypes):
            splits = self.cross_validators[self.splitting_type.name](cv_params=self.splitting_params,
                                                                     indices=self._get_indices(),
                                                                     stratify=stratify)
        elif isinstance(self.splitting_type, HoldOutTypes):
            splits = self.holdout_validators[self.splitting_type.name](holdout_params=self.splitting_params,
                                                                       indices=self._get_indices(),
                                                                       stratify=stratify)
        return splits

    def get_dataset_for_training(self, split_id: int) -> Tuple[Dataset, Dataset]:
        """
        The above split methods employ the Subset to internally subsample the whole dataset.

        During training, we need access to one of those splits. This is a handy function
        to provide training data to fit a pipeline

        Args:
            split (int): The desired subset of the dataset to split and use

        Returns:
            Dataset: the reduced dataset to be used for testing
        """
        # Subset creates a dataset. Splits is a (train_indices, test_indices) tuple
        return (TransformSubset(self, self.splits[split_id][0], train=True),
                TransformSubset(self, self.splits[split_id][1], train=False))

    def replace_data(self, X_train: BaseDatasetType, X_test: Optional[BaseDatasetType]) -> 'BaseDataset':
        """
        To speed up the training of small dataset, early pre-processing of the data
        can be made on the fly by the pipeline.

        In this case, we replace the original train/test tensors by this pre-processed version
        TODO: X_test is None => training is True? or validation step?

        Args:
            X_train (np.ndarray): the pre-processed (imputation/encoding/...) train data
            X_test (np.ndarray): the pre-processed (imputation/encoding/...) test data

        Returns:
            self
        """
        self.train_tensors = (X_train, self.train_tensors[1])
        if X_test is not None and self.test_tensors is not None:
            self.test_tensors = (X_test, self.test_tensors[1])
        return self

    def get_dataset_properties(self, dataset_requirements: List[FitRequirement]) -> Dict[str, Any]:
        """
        Gets the dataset properties required in the fit dictionary
        Args:
            dataset_requirements (List[FitRequirement]): List of
                fit requirements that the dataset properties must
                contain.

        Returns:
            dataset_properties (Dict[str, Any]):
                Dict of the dataset properties.
        """
        dataset_properties = dict()
        # SHUHEI TODO: check dataset_requirements, FitRequirement
        for dataset_requirement in dataset_requirements:
            dataset_properties[dataset_requirement.name] = getattr(self, dataset_requirement.name)

        # Add task type, output type and issparse to dataset properties as
        # they are not dataset requirements in the pipeline
        dataset_specific_properties = _DatasetSpecificProperties(task_type=self.task_type,
                                                                 output_type=self.output_type,
                                                                 issparse=self.issparse,
                                                                 input_shape=self.input_shape,
                                                                 output_shape=self.output_shape,
                                                                 num_classes=self.num_classes)
        dataset_properties.update(**dataset_specific_properties._asdict())
        return dataset_properties
