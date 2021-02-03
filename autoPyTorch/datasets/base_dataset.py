from abc import ABCMeta
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast
from typing import NamedTuple

import numpy as np

from scipy.sparse import issparse

from sklearn.utils.multiclass import type_of_target

from torch.utils.data import Dataset, Subset

import torchvision

from autoPyTorch.datasets.resampling_strategy import (
    CROSS_VAL_FN,
    CrossValTypes,
    DEFAULT_RESAMPLING_PARAMETERS,
    HOLDOUT_FN,
    HoldoutValTypes,
    get_cross_validators,
    get_holdout_validators,
    is_stratified
)
from autoPyTorch.utils.common import FitRequirement, hash_array_or_matrix, BaseNamedTuple, ConstantKeys

BASE_DATASET_TYPE = Union[Tuple[np.ndarray, np.ndarray], Dataset]
NUM_SPLITS, VAL_SHARE = ConstantKeys.NUM_SPLITS, ConstantKeys.VAL_SHARE
STRATIFY, STRATIFIED = ConstantKeys.STRATIFY, ConstantKeys.STRATIFIED


def check_valid_data(data: Any) -> None:
    if not (hasattr(data, '__getitem__') and hasattr(data, '__len__')):
        raise ValueError(
            'The specified Data for Dataset does either not have a __getitem__ or a __len__ attribute.')


def type_check(train_tensors: BASE_DATASET_TYPE, val_tensors: Optional[BASE_DATASET_TYPE] = None) -> None:
    for i in range(len(train_tensors)):
        check_valid_data(train_tensors[i])
    if val_tensors is not None:
        for i in range(len(val_tensors)):
            check_valid_data(val_tensors[i])


class _DatasetSpecificProperties(
    BaseNamedTuple,
    NamedTuple(
        '_DatasetSpecificProperties',
        [
            ('task_type', Optional[str]),
            ('output_type', str),
            ('issparse', bool),
            ('input_shape', Tuple[int]),
            ('output_shape', Tuple[int]),
            ('num_classes', Optional[int])
        ],
    )):
    pass


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


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        train_tensors: BASE_DATASET_TYPE,
        dataset_name: Optional[str] = None,
        val_tensors: Optional[BASE_DATASET_TYPE] = None,
        test_tensors: Optional[BASE_DATASET_TYPE] = None,
        resampling_strategy: Union[CrossValTypes, HoldoutValTypes] = HoldoutValTypes.holdout_validation,
        resampling_strategy_args: Optional[Dict[str, Any]] = None,
        shuffle: Optional[bool] = True,
        seed: Optional[int] = 42,
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
            resampling_strategy (Union[CrossValTypes, HoldoutValTypes]),
                (default=HoldoutValTypes.holdout_validation):
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
        if dataset_name is not None:
            self.dataset_name = dataset_name
        else:
            self.dataset_name = hash_array_or_matrix(train_tensors[0])
        if not hasattr(train_tensors[0], 'shape'):
            type_check(train_tensors, val_tensors)
        self.train_tensors = train_tensors
        self.val_tensors = val_tensors
        self.test_tensors = test_tensors
        self.cross_validators: Dict[str, CROSS_VAL_FN] = {}
        self.holdout_validators: Dict[str, HOLDOUT_FN] = {}
        self.rand = np.random.RandomState(seed=seed)
        self.shuffle = shuffle
        self.resampling_strategy = resampling_strategy
        self.resampling_strategy_args = resampling_strategy_args
        self.task_type: Optional[str] = None
        self.issparse: bool = issparse(self.train_tensors[0])
        self.input_shape: Tuple[int] = train_tensors[0].shape[1:]
        self.num_classes: Optional[int] = None
        if len(train_tensors) == 2 and train_tensors[1] is not None:
            self.output_type: str = type_of_target(self.train_tensors[1])
            # SHUHEI MEMO: What do you wanna do here? shape is always tuple, right?
            """Old: train_tensors[1].shape[1] if train_tensors[1].shape == 2 else 1"""
            self.output_shape: int = train_tensors[1].shape[1] if len(train_tensors[1].shape) == 2 else 1

        # TODO: Look for a criteria to define small enough to preprocess
        self.is_small_preprocess = True

        # Make sure cross validation splits are created once
        self.cross_validators = get_cross_validators(
            CrossValTypes.stratified_k_fold_cross_validation,
            CrossValTypes.k_fold_cross_validation,
            CrossValTypes.shuffle_split_cross_validation,
            CrossValTypes.stratified_shuffle_split_cross_validation
        )
        self.holdout_validators = get_holdout_validators(
            HoldoutValTypes.holdout_validation,
            HoldoutValTypes.stratified_holdout_validation
        )
        self.splits = self.get_splits_from_resampling_strategy()

        # We also need to be able to transform the data, be it for pre-processing
        # or for augmentation
        self.train_transform = train_transforms
        self.val_transform = val_transforms

    def update_transform(self, transform: Optional[torchvision.transforms.Compose],
                         train: bool = True,
                         ) -> 'BaseDataset':
        """
        During the pipeline execution, the pipeline object might propose transformations
        as a product of the current pipeline configuration being tested.

        This utility allows to return a self with the updated transformation, so that
        a dataloader can yield this dataset with the desired transformations

        Args:
            transform (torchvision.transforms.Compose): The transformations proposed
                by the current pipeline
            train (bool): Whether to update the train or validation transform

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

        if hasattr(self.train_tensors[0], 'loc'):
            X = self.train_tensors[0].iloc[[index]]
        else:
            X = self.train_tensors[0][index]

        if self.train_transform is not None and train:
            X = self.train_transform(X)
        elif self.val_transform is not None and not train:
            X = self.val_transform(X)

        # In case of prediction, the targets are not provided
        Y = self.train_tensors[1]
        if Y is not None:
            Y = Y[index]
        else:
            Y = None

        return X, Y

    def __len__(self) -> int:
        return self.train_tensors[0].shape[0]

    def _get_indices(self) -> np.ndarray:
        if self.shuffle:
            indices = self.rand.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices

    def get_splits_from_resampling_strategy(self) -> List[Tuple[List[int], List[int]]]:
        """
        Creates a set of splits based on a resampling strategy provided

        Returns
            (List[Tuple[List[int], List[int]]]): splits in the [train_indices, val_indices] format
        """
        splits = []
        if isinstance(self.resampling_strategy, HoldoutValTypes):
            val_share = DEFAULT_RESAMPLING_PARAMETERS[self.resampling_strategy].get(
                VAL_SHARE, None)
            if self.resampling_strategy_args is not None:
                val_share = self.resampling_strategy_args.get(VAL_SHARE, val_share)
            splits.append(
                self.create_holdout_val_split(
                    holdout_val_type=self.resampling_strategy,
                    val_share=val_share,
                )
            )
        elif isinstance(self.resampling_strategy, CrossValTypes):
            num_splits = DEFAULT_RESAMPLING_PARAMETERS[self.resampling_strategy].get(
                NUM_SPLITS, None)
            if self.resampling_strategy_args is not None:
                num_splits = self.resampling_strategy_args.get(NUM_SPLITS, num_splits)
            # Create the split if it was not created before
            splits.extend(
                self.create_cross_val_splits(
                    cross_val_type=self.resampling_strategy,
                    num_splits=cast(int, num_splits),
                )
            )
        else:
            raise ValueError(f"Unsupported resampling strategy={self.resampling_strategy}")
        return splits

    def create_cross_val_splits(
        self,
        cross_val_type: CrossValTypes,
        num_splits: int
    ) -> List[Tuple[Union[List[int], np.ndarray], Union[List[int], np.ndarray]]]:
        """
        This function creates the cross validation split for the given task.

        It is done once per dataset to have comparable results among pipelines
        Args:
            cross_val_type (CrossValTypes):
            num_splits (int): number of splits to be created

        Returns:
            (List[Tuple[Union[List[int], np.ndarray], Union[List[int], np.ndarray]]]):
                list containing 'num_splits' splits.
        """
        # Create just the split once
        # This is gonna be called multiple times, because the current dataset
        # is being used for multiple pipelines. That is, to be efficient with memory
        # we dump the dataset to memory and read it on a need basis. So this function
        # should be robust against multiple calls, and it does so by remembering the splits
        if not isinstance(cross_val_type, CrossValTypes):
            raise NotImplementedError(f'The selected `cross_val_type` "{cross_val_type}" is not implemented.')
        kwargs = {}
        if is_stratified(cross_val_type):
            # we need additional information about the data for stratification
            kwargs[STRATIFY] = self.train_tensors[-1]
        splits = self.cross_validators[cross_val_type.name](
            num_splits, self._get_indices(), **kwargs)
        return splits

    def create_holdout_val_split(
        self,
        holdout_val_type: HoldoutValTypes,
        val_share: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function creates the holdout split for the given task.

        It is done once per dataset to have comparable results among pipelines
        Args:
            holdout_val_type (HoldoutValTypes):
            val_share (float): share of the validation data

        Returns:
            (Tuple[np.ndarray, np.ndarray]): Tuple containing (train_indices, val_indices)
        """
        if holdout_val_type is None:
            raise ValueError(
                '`val_share` specified, but `holdout_val_type` not specified.'
            )
        if self.val_tensors is not None:
            raise ValueError(
                '`val_share` specified, but the Dataset was a given a pre-defined split at initialization already.')
        if val_share < 0 or val_share > 1:
            raise ValueError(f"`val_share` must be between 0 and 1, got {val_share}.")
        if not isinstance(holdout_val_type, HoldoutValTypes):
            raise NotImplementedError(f'The specified `holdout_val_type` "{holdout_val_type}" is not supported.')
        kwargs = {}
        if is_stratified(holdout_val_type):
            # we need additional information about the data for stratification
            kwargs[STRATIFY] = self.train_tensors[-1]
        # SHUHEI MEMO: What is this function?
        train, val = self.holdout_validators[holdout_val_type.name](val_share, self._get_indices(), **kwargs)
        return train, val

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

    def replace_data(self, X_train: BASE_DATASET_TYPE, X_test: Optional[BASE_DATASET_TYPE]) -> 'BaseDataset':
        """
        To speed up the training of small dataset, early pre-processing of the data
        can be made on the fly by the pipeline.

        In this case, we replace the original train/test tensors by this pre-processed version

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

        """
        dataset_properties = dict()
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
