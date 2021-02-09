from typing import Callable, Union

import numpy as np

import torch

from autoPyTorch.constants import BINARY


def get_loss_weight_strategy(output_type: int) -> Callable:
    if output_type == BINARY:
        return LossWeightStrategyWeightedBinary()
    else:
        return LossWeightStrategyWeighted()


class LossWeightStrategyWeighted():
    def __call__(self, y: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy() if y.is_cuda else y.numpy()
        if isinstance(y[0], str):
            y = y.astype('float64')
        counts = np.sum(y, axis=0)
        total_weight = y.shape[0]

        if len(y.shape) > 1:
            weight_per_class = total_weight / y.shape[1]
            weights = (np.ones(y.shape[1]) * weight_per_class) / np.maximum(counts, 1)
        else:
            classes, counts = np.unique(y, axis=0, return_counts=True)
            classes, counts = classes[::-1], counts[::-1]
            weight_per_class = total_weight / classes.shape[0]
            weights = (np.ones(classes.shape[0]) * weight_per_class) / counts

        return weights


class LossWeightStrategyWeightedBinary():
    def __call__(self, y: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy() if y.is_cuda else y.numpy()
        if isinstance(y[0], str):
            y = y.astype('float64')
        counts_one = np.sum(y, axis=0)
        counts_zero = counts_one + (-y.shape[0])
        weights = counts_zero / np.maximum(counts_one, 1)

        return np.array(weights)
