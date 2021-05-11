import os
import uuid
from abc import ABCMeta
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from scipy.sparse import issparse

from sklearn.utils.multiclass import type_of_target

from torch.utils.data import Dataset, Subset

import torchvision

from autoPyTorch.constants import CLASSIFICATION_OUTPUTS, STRING_TO_OUTPUT_TYPES
from autoPyTorch.datasets.resampling_strategy import CrossValTypes, HoldoutValTypes
from autoPyTorch.utils.common import FitRequirement

BaseDatasetInputType = Union[Tuple[np.ndarray, np.ndarray], Dataset]


def check_valid_data(data: Any) -> None:
    if not all(hasattr(data, attr) for attr in ['__getitem__', '__len__']):
        raise ValueError(
            'The specified Data for Dataset must have both __getitem__ and __len__ attribute.')


def type_check(train_tensors: BaseDatasetInputType,
               val_tensors: Optional[BaseDatasetInputType] = None) -> None:
    """To avoid unexpected behavior, we use loops over indices."""
    for i in range(len(train_tensors)):
        check_valid_data(train_tensors[i])
    if val_tensors is not None:
        for i in range(len(val_tensors)):
            check_valid_data(val_tensors[i])


class TransformSubset(Subset):
    """Wrapper of BaseDataset for splitted datasets

    Since the BaseDataset contains all the data points (train/val/test),
    we require different transformation for each data point.
    This class helps to take the subset of the dataset
    with either training or validation transformation.
    The TransformSubset allows to add train flags
    while indexing the main dataset towards this goal.

    Attributes:
        dataset (BaseDataset/Dataset): Dataset to sample the subset
        indices names (Sequence[int]): Indices to sample from the dataset
        train (bool): If we apply train or validation transformation

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
        train_tensors: BaseDatasetInputType,
        dataset_name: Optional[str] = None,
        val_tensors: Optional[BaseDatasetInputType] = None,
        test_tensors: Optional[BaseDatasetInputType] = None,
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
            train_tensors (A tuple of objects that have a __len__ and a __getitem__ attribute):
                training data
            dataset_name (str): name of the dataset, used as experiment name.
            val_tensors (An optional tuple of objects that have a __len__ and a __getitem__ attribute):
                validation data
            test_tensors (An optional tuple of objects that have a __len__ and a __getitem__ attribute):
                test data
            resampling_strategy (Union[CrossValTypes, HoldoutValTypes]),
                (default=HoldoutValTypes.holdout_validation):
                strategy to split the training data.
            resampling_strategy_args (Optional[Dict[str, Any]]):
                arguments required for the chosen resampling strategy.
                The details are provided in autoPytorch/datasets/resampling_strategy.py
            shuffle:  Whether to shuffle the data before performing splits
            seed (int), (default=1): seed to be used for reproducibility.
            train_transforms (Optional[torchvision.transforms.Compose]):
                Additional Transforms to be applied to the training data
            val_transforms (Optional[torchvision.transforms.Compose]):
                Additional Transforms to be applied to the validation/test data
        """
        self.dataset_name = dataset_name

        if self.dataset_name is None:
            self.dataset_name = str(uuid.uuid1(clock_seq=os.getpid()))

        if not hasattr(train_tensors[0], 'shape'):
            type_check(train_tensors, val_tensors)
        self.train_tensors, self.val_tensors, self.test_tensors = train_tensors, val_tensors, test_tensors
        self.random_state = np.random.RandomState(seed=seed)
        self.shuffle = shuffle

        self.resampling_strategy = resampling_strategy
        self.resampling_strategy_args: Dict[str, Any] = {}
        if resampling_strategy_args is not None:
            self.resampling_strategy_args = resampling_strategy_args

        self.shuffle_split = self.resampling_strategy_args.get('shuffle', False)
        self.is_stratify = self.resampling_strategy_args.get('stratify', False)

        self.task_type: Optional[str] = None
        self.issparse: bool = issparse(self.train_tensors[0])
        self.input_shape: Tuple[int] = self.train_tensors[0].shape[1:]

        if len(self.train_tensors) == 2 and self.train_tensors[1] is not None:
            self.output_type: str = type_of_target(self.train_tensors[1])

            if STRING_TO_OUTPUT_TYPES[self.output_type] in CLASSIFICATION_OUTPUTS:
                self.output_shape = len(np.unique(self.train_tensors[1]))
            else:
                self.output_shape = self.train_tensors[1].shape[-1] if self.train_tensors[1].ndim > 1 else 1

        # TODO: Look for a criteria to define small enough to preprocess
        self.is_small_preprocess = True

        self.splits = self.get_splits_from_resampling_strategy()

        # We also need to be able to transform the data, be it for pre-processing
        # or for augmentation
        self.train_transform = train_transforms
        self.val_transform = val_transforms

    def update_transform(self, transform: Optional[torchvision.transforms.Compose],
                         train: bool = True) -> 'BaseDataset':
        """
        During the pipeline execution, the pipeline object might propose transformations
        as a product of the current pipeline configuration being tested.

        This utility allows to return self with the updated transformation, so that
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
        The base dataset uses a Subset of the data. Nevertheless, the base dataset expects
        both validation and test data to be present in the same dataset, which motivates
        the need to dynamically give train/test data with the __getitem__ command.

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
        Y = self.train_tensors[1][index] if self.train_tensors[1] is not None else None

        return X, Y

    def __len__(self) -> int:
        return self.train_tensors[0].shape[0]

    def _get_indices(self) -> np.ndarray:
        return self.random_state.permutation(len(self)) if self.shuffle else np.arange(len(self))

    def _check_resampling_strategy_args(self) -> None:
        if not any(isinstance(self.resampling_strategy, val_type)
                   for val_type in [HoldoutValTypes, CrossValTypes]):
            raise ValueError(f"resampling_strategy {self.resampling_strategy} is not supported.")

        if self.resampling_strategy_args is not None and \
           not isinstance(self.resampling_strategy_args, dict):

            raise TypeError("resampling_strategy_args must be dict or None,"
                            f" but got {type(self.resampling_strategy_args)}")

        val_share = self.resampling_strategy_args.get('val_share', None)
        num_splits = self.resampling_strategy_args.get('num_splits', None)

        if val_share is not None and (val_share < 0 or val_share > 1):
            raise ValueError(f"`val_share` must be between 0 and 1, got {val_share}.")

        if num_splits is not None:
            if num_splits <= 0:
                raise ValueError(f"`num_splits` must be a positive integer, got {num_splits}.")
            elif not isinstance(num_splits, int):
                raise ValueError(f"`num_splits` must be an integer, got {num_splits}.")

    def get_splits_from_resampling_strategy(self) -> List[Tuple[List[int], List[int]]]:
        """
        Creates a set of splits based on a resampling strategy provided

        Returns
            (List[Tuple[List[int], List[int]]]): splits in the [train_indices, val_indices] format
        """
        # check if the requirements are met and if we can get splits
        self._check_resampling_strategy_args()

        labels_to_stratify = self.train_tensors[-1] if self.is_stratify else None

        if isinstance(self.resampling_strategy, HoldoutValTypes):
            val_share = self.resampling_strategy_args.get('val_share', None)

            return self.resampling_strategy(
                random_state=self.random_state,
                val_share=val_share,
                shuffle=self.shuffle_split,
                indices=self._get_indices(),
                labels_to_stratify=labels_to_stratify
            )
        elif isinstance(self.resampling_strategy, CrossValTypes):
            num_splits = self.resampling_strategy_args.get('num_splits', None)

            return self.resampling_strategy(
                random_state=self.random_state,
                num_splits=num_splits,
                shuffle=self.shuffle_split,
                indices=self._get_indices(),
                labels_to_stratify=labels_to_stratify
            )
        else:
            raise ValueError(f"Unsupported resampling strategy={self.resampling_strategy}")

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

    def replace_data(self, X_train: BaseDatasetInputType,
                     X_test: Optional[BaseDatasetInputType]) -> 'BaseDataset':
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
        Gets the dataset properties required in the fit dictionary.
        This depends on the components that are active in the
        pipeline and returns the properties they need about the dataset.
        Information of the required properties of each component
        can be found in their documentation.
        Args:
            dataset_requirements (List[FitRequirement]): List of
                fit requirements that the dataset properties must
                contain. This is created using the `get_dataset_requirements
                function in
                <https://github.com/automl/Auto-PyTorch/blob/refactor_development/autoPyTorch/utils/pipeline.py#L25>`

        Returns:
            dataset_properties (Dict[str, Any]):
                Dict of the dataset properties.
        """
        dataset_properties = dict()
        for dataset_requirement in dataset_requirements:
            dataset_properties[dataset_requirement.name] = getattr(self, dataset_requirement.name)

        # Add the required dataset info to dataset properties as
        # they might not be a dataset requirement in the pipeline
        dataset_properties.update(self.get_required_dataset_info())
        return dataset_properties

    def get_required_dataset_info(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing required dataset
        properties to instantiate a pipeline.
        """
        info = {'output_type': self.output_type,
                'issparse': self.issparse}
        return info
