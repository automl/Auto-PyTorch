from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from autoPyTorch.pipeline.components.training.trainer.base_trainer import BaseTrainerComponent


class StandardTrainer(BaseTrainerComponent):
    def __init__(self, weighted_loss: bool = False,
                 use_stochastic_weight_averaging: bool = False,
                 use_snapshot_ensemble: bool = False,
                 se_lastk: int = 3,
                 use_lookahead_optimizer: bool = True,
                 random_state: Optional[Union[np.random.RandomState, int]] = None,
                 **lookahead_config: Dict[str, Any]):
        """
        This class handles the training of a network for a single given epoch.

        Args:
            weighted_loss (bool): whether to use weighted loss

        """
        super().__init__(random_state=random_state,
                         weighted_loss=weighted_loss,
                         use_stochastic_weight_averaging=use_stochastic_weight_averaging,
                         use_snapshot_ensemble=use_snapshot_ensemble,
                         se_lastk=se_lastk,
                         use_lookahead_optimizer=use_lookahead_optimizer,
                         **lookahead_config)

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
        return X, {'y_a': y}

    def criterion_preparation(self, y_a: np.ndarray, y_b: np.ndarray = None, lam: float = 1.0
                              ) -> Callable:
        return lambda criterion, pred: criterion(pred, y_a)

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'StandardTrainer',
            'name': 'StandardTrainer',
            'handles_tabular': True,
            'handles_image': True,
            'handles_time_series': True,
        }
