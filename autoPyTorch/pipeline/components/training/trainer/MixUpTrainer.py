from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch

from autoPyTorch.pipeline.components.training.trainer.base_trainer import BaseTrainerComponent
from autoPyTorch.pipeline.components.training.trainer.mixup_utils import MixUp


class MixUpTrainer(MixUp, BaseTrainerComponent):
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
            Dict[str, np.ndarray]: arguments to the criterion function
        """
        lam = self.random_state.beta(self.alpha, self.alpha) if self.alpha > 0. else 1.
        batch_size = X.size()[0]
        index = torch.randperm(batch_size).cuda() if X.is_cuda else torch.randperm(batch_size)

        mixed_x = lam * X + (1 - lam) * X[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, {'y_a': y_a, 'y_b': y_b, 'lam': lam}

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Union[str, bool]]:

        return {
            'shortname': 'MixUpTrainer',
            'name': 'MixUp Regularized Trainer',
            'handles_tabular': True,
            'handles_image': True,
            'handles_time_series': True,
        }
