import typing

import numpy as np

from autoPyTorch.pipeline.components.training.trainer.base_trainer import BaseTrainerComponent
from autoPyTorch.pipeline.components.training.trainer.cutout_utils import CutOut


class GridCutOutTrainer(CutOut, BaseTrainerComponent):

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
        r = np.random.rand(1)
        batch_size, channel, W, H = X.size()
        if r > self.cutout_prob:
            return X, {'y_a': y, 'y_b': y, 'lam': 1}

        # Draw parameters of a random bounding box
        # Where to cut basically
        cut_rat = np.sqrt(1. - self.patch_ratio)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        X[:, :, bbx1:bbx2, bby1:bby2] = 0.0

        return X, {'y_a': y, 'y_b': y, 'lam': 1}

    @staticmethod
    def get_properties(dataset_properties: typing.Optional[typing.Dict[str, typing.Any]] = None
                       ) -> typing.Dict[str, typing.Union[str, bool]]:
        return {
            'shortname': 'GridCutOutTrainer',
            'name': 'GridCutOutTrainer',
            'handles_tabular': False,
            'handles_image': True,
            'handles_time_series': False,
        }
