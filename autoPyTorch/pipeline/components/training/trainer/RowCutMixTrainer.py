import typing

import numpy as np

import torch

from autoPyTorch.pipeline.components.training.trainer.base_trainer import BaseTrainerComponent
from autoPyTorch.pipeline.components.training.trainer.mixup_utils import MixUp


class RowCutMixTrainer(MixUp, BaseTrainerComponent):

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
        alpha, beta = 1.0, 1.0
        lam = self.random_state.beta(alpha, beta)
        batch_size = X.shape[0]
        device = torch.device('cuda' if X.is_cuda else 'cpu')
        batch_indices = torch.randperm(batch_size).to(device)

        r = self.random_state.rand(1)
        if beta <= 0 or r > self.alpha:
            return X, {'y_a': y, 'y_b': y[batch_indices], 'lam': 1}

        row_size = X.shape[1]
        row_indices = torch.tensor(
            self.random_state.choice(
                range(1, row_size),
                max(1, int(row_size * lam)),
                replace=False
            )
        )

        X[:, row_indices] = X[batch_indices, :][:, row_indices]

        # Adjust lam
        lam = 1 - len(row_indices) / X.shape[1]

        y_a, y_b = y, y[batch_indices]

        return X, {'y_a': y_a, 'y_b': y_b, 'lam': lam}

    @staticmethod
    def get_properties(dataset_properties: typing.Optional[typing.Dict[str, typing.Any]] = None
                       ) -> typing.Dict[str, typing.Union[str, bool]]:
        return {
            'shortname': 'RowCutMixTrainer',
            'name': 'MixUp Regularized with Cutoff Tabular Trainer',
            'handles_tabular': True,
            'handles_image': False,
            'handles_time_series': False,
        }
