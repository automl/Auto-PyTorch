import typing

import numpy as np

import torch

from autoPyTorch.pipeline.components.training.trainer.base_trainer import BaseTrainerComponent
from autoPyTorch.pipeline.components.training.trainer.mixup_utils import MixUp


class GridCutMixTrainer(MixUp, BaseTrainerComponent):
    """  # noqa
    References:
        Title: CutMix: Regularization Strategy to Train Strong Classifiers
               with Localizable Features
        Authors: Sangdoo Yun et. al.
        URL: https://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf
        Github URL: https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py#L227-L244
    """

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
        batch_size, _, W, H = X.shape
        device = torch.device('cuda' if X.is_cuda else 'cpu')
        batch_indices = torch.randperm(batch_size).to(device)

        r = self.random_state.rand(1)
        if beta <= 0 or r > self.alpha:
            return X, {'y_a': y, 'y_b': y[batch_indices], 'lam': 1}

        # Draw parameters of a random bounding box
        # Where to cut basically
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        cx = self.random_state.randint(W)
        cy = self.random_state.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        X[:, :, bbx1:bbx2, bby1:bby2] = X[batch_indices, :, bbx1:bbx2, bby1:bby2]

        # Adjust lam
        pixel_size = W * H
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / pixel_size)

        y_a, y_b = y, y[batch_indices]

        return X, {'y_a': y_a, 'y_b': y_b, 'lam': lam}

    @staticmethod
    def get_properties(dataset_properties: typing.Optional[typing.Dict[str, typing.Any]] = None
                       ) -> typing.Dict[str, typing.Union[str, bool]]:
        return {
            'shortname': 'GridCutMixTrainer',
            'name': 'GridCutMixTrainer',
            'handles_tabular': False,
            'handles_image': True,
            'handles_time_series': False,
        }
