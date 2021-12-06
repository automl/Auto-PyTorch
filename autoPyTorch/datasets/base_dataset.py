import os
import uuid
from abc import ABCMeta
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np

from scipy.sparse import issparse

from sklearn.utils.multiclass import type_of_target

from torch.utils.data import Dataset, Subset

import torchvision

from autoPyTorch.constants import CLASSIFICATION_OUTPUTS, STRING_TO_OUTPUT_TYPES
from autoPyTorch.datasets.resampling_strategy import (
    CrossValFunc,
    CrossValFuncs,
    CrossValTypes,
    DEFAULT_RESAMPLING_PARAMETERS,
    HoldOutFunc,
    HoldOutFuncs,
    HoldoutValTypes
)
from autoPyTorch.utils.common import FitRequirement

BaseDatasetInputType = Union[Tuple[np.ndarray, np.ndarray], Dataset]
BaseDatasetPropertiesType = Union[int, float, str, List, bool]


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


def _return_output_shape(output_type: str, target_labels: np.ndarray) -> int:
    """
    Returns the shape of output given a target_labels and output_type.

    Args:
        output_type (str):
            The output type according to sklearn specification.
        target_labels (np.ndarray):
            Target labels of the training dataset.

    Returns:
        output_shape (int):
            The shape of outputs.
    """
    if STRING_TO_OUTPUT_TYPES.get(output_type, None) in CLASSIFICATION_OUTPUTS:
        return len(np.unique(target_labels))
    elif target_labels.ndim > 1:
        return target_labels.shape[-1]

    return 1


def _double_check_and_return_property_of_target(train_tensors: BaseDatasetInputType) -> Tuple[str, int]:
    """
    Since task type inference by sklearn (see Reference below) for continuous is
    not suitable for AutoPytorch, we double-check the task type in the case
    when we get `multiclass` from the inference.

    Args:
        train_tensors (BaseDatasetInputType):
            feature and label tensors that are used for training.

    Returns:
        output_type (str):
            output_type of the label tensor.
            The return will be either:
                - continuous
                - binary
                - multiclass
                - multiclass-multioutput
                - continuous-multioutput  # TODO: Check usecases
                - multilabel-indicator
        output_shape (int):
            The shape of the output.

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.type_of_target.html#exampleshttps://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.type_of_target.html  # noqa: E501
    """
    if len(train_tensors) != 2 or train_tensors[1] is None:
        raise ValueError(
            'Unsupervised learning has not been supported yet. '
            'Make sure that your dataset has labels and the format is correct.'
        )

    if isinstance(train_tensors, Dataset):
        target_labels = np.array([sample[-1] for sample in train_tensors])
    else:
        target_labels = np.array(train_tensors[1])

    output_type: str = type_of_target(target_labels)
    is_numerical_multiclass = (output_type == 'multiclass' and target_labels.dtype != np.dtype('O'))

    if is_numerical_multiclass:
        # From sklearn design, it is guaranteed that elements are not float-like.
        target_labels = target_labels.astype(np.int64)
        unique_labels = np.unique(target_labels)

        # We assume that every label vector includes [0, C) if the task is C-class classification.
        # e.g. labels = [0, 1, 3, 1, 3] will be viewed as regression since it does not have `2`.
        is_continuous = np.any(unique_labels != np.arange(unique_labels.size))

        output_type = output_type if not is_continuous else 'continuous'

    return output_type, _return_output_shape(output_type, target_labels)


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
        self.dataset_name = dataset_name

        if self.dataset_name is None:
            self.dataset_name = str(uuid.uuid1(clock_seq=os.getpid()))

        if not hasattr(train_tensors[0], 'shape'):
            type_check(train_tensors, val_tensors)
        self.train_tensors, self.val_tensors, self.test_tensors = train_tensors, val_tensors, test_tensors
        self.cross_validators: Dict[str, CrossValFunc] = {}
        self.holdout_validators: Dict[str, HoldOutFunc] = {}
        self.random_state = np.random.RandomState(seed=seed)
        self.shuffle = shuffle
        self.resampling_strategy = resampling_strategy
        self.resampling_strategy_args = resampling_strategy_args
        self.task_type: Optional[str] = None
        self.issparse: bool = issparse(self.train_tensors[0])
        self.input_shape: Tuple[int] = self.train_tensors[0].shape[1:]

        self.output_type, self.output_shape = _double_check_and_return_property_of_target(train_tensors)

        # TODO: Look for a criteria to define small enough to preprocess
        self.is_small_preprocess = True

        # Make sure cross validation splits are created once
        self.cross_validators = CrossValFuncs.get_cross_validators(*CrossValTypes)
        self.holdout_validators = HoldOutFuncs.get_holdout_validators(*HoldoutValTypes)
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
        return int(self.train_tensors[0].shape[0])

    def _get_indices(self) -> np.ndarray:
        return self.random_state.permutation(len(self)) if self.shuffle else np.arange(len(self))

    def get_splits_from_resampling_strategy(self) -> List[Tuple[List[int], List[int]]]:
        """
        Creates a set of splits based on a resampling strategy provided

        Returns
            (List[Tuple[List[int], List[int]]]): splits in the [train_indices, val_indices] format
        """
        splits = []
        if isinstance(self.resampling_strategy, HoldoutValTypes):
            val_share = DEFAULT_RESAMPLING_PARAMETERS[self.resampling_strategy].get(
                'val_share', None)
            if self.resampling_strategy_args is not None:
                val_share = self.resampling_strategy_args.get('val_share', val_share)
            splits.append(
                self.create_holdout_val_split(
                    holdout_val_type=self.resampling_strategy,
                    val_share=val_share,
                )
            )
        elif isinstance(self.resampling_strategy, CrossValTypes):
            num_splits = DEFAULT_RESAMPLING_PARAMETERS[self.resampling_strategy].get(
                'num_splits', None)
            if self.resampling_strategy_args is not None:
                num_splits = self.resampling_strategy_args.get('num_splits', num_splits)
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
        if cross_val_type.is_stratified():
            # we need additional information about the data for stratification
            kwargs["stratify"] = self.train_tensors[-1]
        splits = self.cross_validators[cross_val_type.name](
            self.random_state, num_splits, self._get_indices(), **kwargs)
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
        if holdout_val_type.is_stratified():
            # we need additional information about the data for stratification
            kwargs["stratify"] = self.train_tensors[-1]
        train, val = self.holdout_validators[holdout_val_type.name](
            self.random_state, val_share, self._get_indices(), **kwargs)
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

    def get_dataset_properties(
        self, dataset_requirements: List[FitRequirement]
    ) -> Dict[str, BaseDatasetPropertiesType]:
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
            dataset_properties (Dict[str, BaseDatasetPropertiesType]):
                Dict of the dataset properties.
        """
        dataset_properties = dict()
        for dataset_requirement in dataset_requirements:
            dataset_properties[dataset_requirement.name] = getattr(self, dataset_requirement.name)

        # Add the required dataset info to dataset properties as
        # they might not be a dataset requirement in the pipeline
        dataset_properties.update(self.get_required_dataset_info())
        return dataset_properties

    def get_required_dataset_info(self) -> Dict[str, BaseDatasetPropertiesType]:
        """
        Returns a dictionary containing required dataset
        properties to instantiate a pipeline.
        """
        info: Dict[str, BaseDatasetPropertiesType] = {'output_type': self.output_type,
                                                      'issparse': self.issparse}
        return info
