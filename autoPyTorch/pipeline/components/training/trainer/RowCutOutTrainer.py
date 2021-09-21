import typing

import numpy as np

import torch

from autoPyTorch.pipeline.components.training.trainer.base_trainer import BaseTrainerComponent
from autoPyTorch.pipeline.components.training.trainer.cutout_utils import CutOut


class RowCutOutTrainer(CutOut, BaseTrainerComponent):
    """
    References:
        Title: Improved Regularization of Convolutional Neural Networks with Cutout
        Authors: Terrance DeVries and Graham W. Taylor
        URL: https://arxiv.org/pdf/1708.04552.pdf
        Github URL: https://github.com/hysts/pytorch_cutout/blob/master/dataloader.py#L36-L68
    """

    # 0 is non-informative in image data
    NUMERICAL_VALUE = 0
    # -1 is the conceptually equivalent to 0 in a image, i.e. 0-pad
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

        # (batch_size (permutation of rows), col_size) = X.shape
        col_size = X.shape[1]
        col_indices = self.random_state.choice(range(col_size), max(1, int(col_size * self.patch_ratio)),
                                               replace=False)

        if not isinstance(self.numerical_columns, typing.Iterable):
            raise ValueError("numerical_columns in {} must be iterable, "
                             "but got {}.".format(self.__class__.__name__,
                                                  self.numerical_columns))

        numerical_indices = torch.tensor(self.numerical_columns)
        categorical_indices = torch.tensor([idx for idx in col_indices if idx not in self.numerical_columns])

        X[:, categorical_indices.long()] = self.CATEGORICAL_VALUE
        X[:, numerical_indices.long()] = self.NUMERICAL_VALUE

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
