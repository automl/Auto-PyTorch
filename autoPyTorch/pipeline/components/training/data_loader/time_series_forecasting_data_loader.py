from typing import Any, Dict, Optional, Tuple

from autoPyTorch.pipeline.components.training.data_loader.time_series_data_loader import TimeSeriesDataLoader


from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter, Constant
)

import numpy as np

import torch

import torchvision

import warnings


from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.time_series_dataset import TimeSeriesForecastingDataset
from autoPyTorch.pipeline.components.training.base_training import autoPyTorchTrainingComponent
from autoPyTorch.utils.backend import Backend
from autoPyTorch.utils.common import FitRequirement, custom_collate_fn


class TimeSeriesForecastingDataLoader(TimeSeriesDataLoader):
    """This class is an interface to the PyTorch Dataloader.

    It gives the possibility to read various types of mapped
    datasets as described in:
    https://pytorch.org/docs/stable/data.html

    """

    def __init__(self,
                 batch_size: int = 64,
                 sequence_length: int = 1,
                 #sample_interval: int = 1,
                 upper_sequence_length: int = np.iinfo(np.int32).max,
                 n_prediction_steps: int = 1) -> None:
        """
        initialize a dataloader
        Args:
            batch_size: batch size
            sequence_length: length of each sequence
            sample_interval: sample interval ,its value is the interval of the resolution
            upper_sequence_length: upper limit of sequence length, to avoid a sequence length larger than dataset length
            or specified by the users
            n_prediction_steps: how many stpes to predict in advance
        """
        super().__init__(batch_size=batch_size)
        self.sequence_length: int = sequence_length
        self.upper_seuqnce_length = upper_sequence_length
        self.n_prediction_steps = n_prediction_steps
        self.sample_interval = 1
        # length of the tail, for instance if a sequence_length = 2, sample_interval =2, n_prediction = 2,
        # the time sequence should look like: [X, y, X, y, y] [test_data](values in tail is marked with X)
        self.tail_length = (self.sequence_length * self.sample_interval) + self.n_prediction_steps - 1

    def transform(self, X: np.ndarray) -> np.ndarray:
        """The transform function calls the transform function of the
        underlying model and returns the transformed array.

        Args:
            X (np.ndarray): input features

        Returns:
            np.ndarray: Transformed features
        """
        X.update({'train_data_loader': self.train_data_loader,
                  'val_data_loader': self.val_data_loader,
                  'X_train': self.datamanager.train_tensors[0],
                  'y_train': self.datamanager.train_tensors[1]})
        if self.datamanager.val_tensors is not None and 'X_val' in X:
            X.update({'X_val': self.datamanager.val_tensors[0],
                      'y_val': self.datamanager.val_tensors[1]})
        if self.datamanager.test_tensors is not None and 'X_test' in X:
            X.update({'X_test': self.datamanager.test_tensors[0],
                      'y_test': self.datamanager.test_tensors[1]})

        return X

    def fit(self, X: Dict[str, Any], y: Any = None) -> torch.utils.data.DataLoader:
        """
        Fits a component by using an input dictionary with pre-requisites

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API

        Returns:
            A instance of self
        """
        fraction_subset = X.get('fraction_subset', 1.0)
        self.sample_interval = int(np.ceil(1.0 / fraction_subset))
        print("!"*50)
        print(self.sample_interval)
        print("#"*50)
        self.tail_length = (self.sequence_length * self.sample_interval) + self.n_prediction_steps - 1

        # Make sure there is an optimizer
        self.check_requirements(X, y)

        # Incorporate the transform to the dataset
        datamanager = X['backend'].load_datamanager()
        datamanager = self._update_dataset(datamanager)

        self.train_transform = self.build_transform(X, mode='train')
        self.val_transform = self.build_transform(X, mode='val')
        self.test_transform = self.build_transform(X, mode='test')
        datamanager.update_transform(
            self.train_transform,
            train=True,
        )
        datamanager.update_transform(
            self.val_transform,
            train=False,
        )

        if X['dataset_properties']["is_small_preprocess"]:
            # This parameter indicates that the data has been pre-processed for speed
            # Overwrite the datamanager with the pre-processes data
            datamanager.replace_data(X['X_train'], X['X_test'] if 'X_test' in X else None)
        train_dataset, val_dataset = datamanager.get_dataset_for_training(split_id=X['split_id'])

        self.datamanager = datamanager

        self.train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=True,
            num_workers=X.get('num_workers', 0),
            pin_memory=X.get('pin_memory', True),
            drop_last=X.get('drop_last', True),
            collate_fn=custom_collate_fn,
        )

        self.val_data_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=min(self.batch_size, len(val_dataset)),
            shuffle=False,
            num_workers=X.get('num_workers', 0),
            pin_memory=X.get('pin_memory', True),
            drop_last=X.get('drop_last', False),
            collate_fn=custom_collate_fn,
        )

        return self

    def _update_dataset(self, datamanager: TimeSeriesForecastingDataset):
        """
        update the dataset to build time sequence
        """
        num_features = datamanager.num_features
        population_size = datamanager.population_size
        num_target = datamanager.num_target

        X_train, y_train = datamanager.train_tensors
        val_tensors = datamanager.val_tensors
        test_tensors = datamanager.test_tensors
        n_prediction_steps = datamanager.n_prediction_steps

        X_train = X_train.reshape([-1, population_size, num_features])
        y_train = y_train.reshape([-1, population_size, num_target])

        time_series_length = X_train.shape[0]
        self.population_size = population_size
        self.num_features = num_features
        num_datapoints_train = time_series_length - (self.sequence_length - 1) * self.sample_interval - n_prediction_steps + 1
        num_targets = y_train.shape[-1]

        y_train = y_train[-num_datapoints_train:, :]
        if test_tensors is not None:
            X_test, y_test = test_tensors

            X_test = X_test.reshape([-1, population_size, num_features])
            y_test = y_test.reshape([-1, population_size, num_target])

            if val_tensors is not None:
                X_val, y_val = val_tensors

                X_val = X_val.reshape([-1, population_size, num_features])
                y_val = y_val.reshape([-1, population_size, num_target])

                num_datapoints_val = X_val.shape[0]

                X_val = np.concatenate([X_train[-self.tail_length:], X_val])
                X_test = np.concatenate([X_val[-self.tail_length:], X_test])
                val_tensors = self._ser2seq(X_val, y_val, num_datapoints_val, num_features, num_targets)
                datamanager.val_tensors = val_tensors

            num_datapoints_test = X_test.shape[0]

            X_test = np.concatenate([X_train[-self.tail_length:], X_test])
            self.X_val_tail = X_test[-self.tail_length:] if self.tail_length > 1 \
                else np.zeros((0, population_size, num_features)).astype(dtype=X_test.dtype)

            test_tensors = self._ser2seq(X_test, y_test, num_datapoints_test, num_features, num_targets)
            datamanager.test_tensors = test_tensors

        elif val_tensors is not None:
            X_val, y_val = val_tensors
            X_val = np.concatenate([X_train[-self.tail_length:], X_val])

            # used for prediction
            self.X_val_tail = X_val[-self.tail_length:]
            val_tensors = self._ser2seq(X_val, y_val, num_datapoints_train, num_features, num_targets)
            datamanager.val_tensors = val_tensors
        else:
            self.X_val_tail = X_train[-self.tail_length:]

        train_tensors = self._ser2seq(X_train, y_train, num_datapoints_train, num_features, num_targets)
        datamanager.train_tensors = train_tensors
        datamanager.splits = datamanager.get_splits_from_resampling_strategy()
        return datamanager

    def _ser2seq(self, X_in, y_in, num_datapoints, num_features, num_targets):
        """
        build a sliding window transformer for the given data
         Args:
            X_in (np.ndarray): input feature array to be transformed with shape
             [time_series_length, population_size, num_features]
            y_in (np.ndarray): input target array with shape [time_series_length, population_size, num_targets]
            num_datapoints: number of actual data points stored in the dataset
            num_features: number of features
            num_targets: number of targets
        Returns:
            X_in_trans: transformed input featuer array with shpae
            [num_datapoints * population_size, sequence_length, num_features]
            y_in_trans: transformed input target array with shape
            [num_datapoints * population_size, num_targets]
        """
        X_in = np.concatenate([np.roll(X_in, shift=i * self.sample_interval, axis=0) for i in range(0, -self.sequence_length, -1)],
                              axis=2).astype(np.float32)[:num_datapoints]
        X_in = X_in.reshape((-1, self.sequence_length, num_features))
        y_in = y_in.reshape((-1, num_targets))
        return X_in, y_in

    def get_loader(self, X: np.ndarray, y: Optional[np.ndarray] = None, batch_size: int = np.inf,
                   ) -> torch.utils.data.DataLoader:
        """
        Creates a data loader object from the provided data,
        applying the transformations meant to validation objects
        """
        if X.ndim == 3:
            X_shape = X.shape
            if X_shape[-1] != self.num_features:
                raise ValueError("the features of test data is incompatible with the training data")
            if X_shape[1] == self.population_size:
                num_points_X_in = X_shape[0]
            elif X_shape[0] == self.population_size:
                num_points_X_in = X_shape[1]
                X = np.swapaxes(X, 0, 1)
            elif X_shape[1] == 1:
                X = X.reshape([-1, self.population_size, self.num_features])
                num_points_X_in = X_shape[0]
            else:
                raise ValueError("test shape is incompatible with the training shape")
        else:
            raise ValueError(
                "The test data for time series forecasting has to be a three-dimensional tensor of shape PxLxM.")

        X = np.concatenate([self.X_val_tail, X])
        X = np.concatenate([np.roll(X, shift=i * self.sample_interval, axis=0) for i in range(0, -self.sequence_length, -1)],
                           axis=2).astype(np.float32)[:num_points_X_in]
        X = X.reshape((-1, self.sequence_length, self.num_features))


        dataset = BaseDataset(
            train_tensors=(X, y),
            # This dataset is used for loading test data in a batched format
            train_transforms=self.test_transform,
            val_transforms=self.test_transform,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

    def get_train_data_loader(self) -> torch.utils.data.DataLoader:
        """Returns a data loader object for the train data

        Returns:
            torch.utils.data.DataLoader: A train data loader
        """
        assert self.train_data_loader is not None, "No train data loader fitted"
        return self.train_data_loader

    def get_val_data_loader(self) -> torch.utils.data.DataLoader:
        """Returns a data loader object for the validation data

        Returns:
            torch.utils.data.DataLoader: A validation data loader
        """
        assert self.val_data_loader is not None, "No val data loader fitted"
        return self.val_data_loader

    def get_test_data_loader(self) -> torch.utils.data.DataLoader:
        """Returns a data loader object for the test data

        Returns:
            torch.utils.data.DataLoader: A validation data loader
        """
        return self.test_data_loader

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None,
                                        batch_size: Tuple[Tuple, int] = ((32, 320), 64),
                                        sequence_length: Tuple[Tuple, int] = ((1, 20), 1)
                                        ) -> ConfigurationSpace:
        batch_size = UniformIntegerHyperparameter(
            "batch_size", batch_size[0][0], batch_size[0][1], default_value=batch_size[1])
        if "upper_sequence_length" not in dataset_properties:
            warnings.warn('max_sequence_length is not given in dataset property , might exists the risk of selecting '
                          'length that is greater than the maximal allowed length of the dataset')
            upper_sequence_length = min(np.iinfo(np.int32).max, sequence_length[0][1])
        else:
            upper_sequence_length = min(dataset_properties["upper_sequence_length"], sequence_length[0][1])
        if sequence_length[0][0] >= upper_sequence_length:
            warnings.warn("the lower bound of sequence length is greater than the upper bound")
            sequence_length = Constant("sequence_length", upper_sequence_length)
        else:
            sequence_length = UniformIntegerHyperparameter("sequence_length",
                                                           lower=sequence_length[0][0],
                                                           upper=upper_sequence_length,
                                                           default_value=sequence_length[1])
        cs = ConfigurationSpace()
        cs.add_hyperparameters([batch_size, sequence_length])
        return cs

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.train_data_loader.__class__.__name__
        return string
