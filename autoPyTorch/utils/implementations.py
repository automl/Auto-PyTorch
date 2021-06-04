from typing import Any, Callable, Dict, Type, Union

import numpy as np

import torch


def get_loss_weight_strategy(loss: Type[torch.nn.Module]) -> Callable:
    """
    Utility function that returns strategy for the given loss
    Args:
        loss (Type[torch.nn.Module]): type of the loss function
    Returns:
        (Callable): Relevant Callable strategy
    """
    if loss.__name__ in LossWeightStrategyWeightedBinary.get_properties()['supported_losses']:
        return LossWeightStrategyWeightedBinary()
    elif loss.__name__ in LossWeightStrategyWeighted.get_properties()['supported_losses']:
        return LossWeightStrategyWeighted()
    else:
        raise ValueError("No strategy currently supports the given loss, {}".format(loss.__name__))


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
            #classes, counts = classes[::-1], counts[::-1]
            weight_per_class = total_weight / classes.shape[0]
            weights = (np.ones(classes.shape[0]) * weight_per_class) / counts

        return weights

    @staticmethod
    def get_properties() -> Dict[str, Any]:
        return {'supported_losses': ['CrossEntropyLoss']}


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

    @staticmethod
    def get_properties() -> Dict[str, Any]:
        return {'supported_losses': ['BCEWithLogitsLoss']}
