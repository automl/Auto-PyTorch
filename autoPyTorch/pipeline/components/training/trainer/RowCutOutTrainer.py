from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from autoPyTorch.pipeline.components.training.trainer.base_trainer import BaseTrainerComponent
from autoPyTorch.pipeline.components.training.trainer.cutout_utils import CutOut


class RowCutOutTrainer(CutOut, BaseTrainerComponent):

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

        r = self.random_state.rand(1)
        if r > self.cutout_prob:
            y_a = y
            y_b = y
            lam = 1
            return X, {'y_a': y_a, 'y_b': y_b, 'lam': lam}

        size: int = np.shape(X)[1]
        indices = self.random_state.choice(range(size), max(1, np.int32(size * self.patch_ratio)),
                                           replace=False)

        X[:, indices] = 0
        lam = 1
        y_a = y
        y_b = y
        return X, {'y_a': y_a, 'y_b': y_b, 'lam': lam}

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'RowCutOutTrainer',
            'name': 'RowCutOutTrainer',
            'handles_tabular': True,
            'handles_image': False,
            'handles_time_series': False,
        }
