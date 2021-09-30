import typing

import numpy as np

# import torch

from autoPyTorch.pipeline.components.training.trainer.base_trainer import BaseTrainerComponent
from autoPyTorch.pipeline.components.training.trainer.cutout_utils import CutOut


class RowCutOutTrainer(CutOut, BaseTrainerComponent):
    NUMERICAL_VALUE = 0
    CATEGORICAL_VALUE = -1

    def data_preparation(self, X: np.ndarray, y: np.ndarray,
                         ) -> typing.Tuple[np.ndarray, typing.Dict[str, np.ndarray]]:
        """
        Depending on the trainer choice, data fed to the network might be pre-processed
        on a different way. That is, in standard training we provide the data to the
        network as we receive it to the loader. Some regularization techniques, like mixup
        alter the data.

        Args:
            X (np.ndarray): The batch training features
            y (np.ndarray): The batch training labels

        Returns:
            np.ndarray: that processes data
            typing.Dict[str, np.ndarray]: arguments to the criterion function
        """

        r = self.random_state.rand(1)
        if r > self.cutout_prob:
            y_a = y
            y_b = y
            lam = 1
            return X, {'y_a': y_a, 'y_b': y_b, 'lam': lam}

        size = X.shape[1]
        indices = self.random_state.choice(range(size), max(1, np.int32(size * self.patch_ratio)),
                                           replace=False)

        """
        if not isinstance(self.numerical_columns, typing.Iterable):
            raise ValueError("{} requires numerical columns information of {}"
                             "to prepare data got {}.".format(self.__class__.__name__,
                                                              typing.Iterable,
                                                              self.numerical_columns))
        numerical_indices = torch.tensor(self.numerical_columns)
        categorical_indices = torch.tensor([index for index in indices if index not in self.numerical_columns])

        # We use an ordinal encoder on the categorical columns of tabular data
        # -1 is the conceptual equivalent to 0 in a image, that does not
        # have color as a feature and hence the network has to learn to deal
        # without this data. For numerical columns we use 0 to cutout the features
        # similar to the effect that setting 0 as a pixel value in an image.
        X[:, categorical_indices.long()] = self.CATEGORICAL_VALUE
        X[:, numerical_indices.long()] = self.NUMERICAL_VALUE
        """
        X[:, indices] = 0
        lam = 1
        y_a = y
        y_b = y
        return X, {'y_a': y_a, 'y_b': y_b, 'lam': lam}

    @staticmethod
    def get_properties(dataset_properties: typing.Optional[typing.Dict[str, typing.Any]] = None
                       ) -> typing.Dict[str, typing.Union[str, bool]]:
        return {
            'shortname': 'RowCutOutTrainer',
            'name': 'RowCutOutTrainer',
            'handles_tabular': True,
            'handles_image': False,
            'handles_time_series': False,
        }
