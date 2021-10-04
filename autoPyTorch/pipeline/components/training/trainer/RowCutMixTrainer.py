from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

import torch

from autoPyTorch.pipeline.components.training.trainer.base_trainer import BaseTrainerComponent
from autoPyTorch.pipeline.components.training.trainer.mixup_utils import MixUp


class RowCutMixTrainer(MixUp, BaseTrainerComponent):

    def data_preparation(self, X: np.ndarray, y: np.ndarray,
                         ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
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
        beta = 1.0
        lam = self.random_state.beta(beta, beta)
        batch_size, n_columns = np.shape(X)
        index = torch.randperm(batch_size).cuda() if X.is_cuda else torch.randperm(batch_size)

        r = self.random_state.rand(1)
        if beta <= 0 or r > self.alpha:
            return X, {'y_a': y, 'y_b': y[index], 'lam': 1}

        indices = torch.tensor(self.random_state.choice(range(batch_size), max(1, np.int32(n_columns * lam)),
                                                        replace=False))

        X[:, indices] = X[index, :][:, indices]

        # Adjust lam
        lam = 1 - (len(indices) / n_columns)

        y_a, y_b = y, y[index]

        return X, {'y_a': y_a, 'y_b': y_b, 'lam': lam}

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'RowCutMixTrainer',
            'name': 'MixUp Regularized with Cutoff Tabular Trainer',
            'handles_tabular': True,
            'handles_image': False,
            'handles_time_series': False,
        }
