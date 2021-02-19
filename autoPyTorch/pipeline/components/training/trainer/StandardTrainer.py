import typing

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

from autoPyTorch.pipeline.components.training.trainer.base_trainer import BaseTrainerComponent


class StandardTrainer(BaseTrainerComponent):
    def __init__(self, weighted_loss: bool = False,
                 use_swa: bool = False,
                 random_state: typing.Optional[np.random.RandomState] = None):
        """
        This class handles the training of a network for a single given epoch.

        Args:
            weighted_loss (bool): whether to use weighted loss

        """
        super().__init__(random_state=random_state, weighted_loss=weighted_loss, use_swa=use_swa)

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
        return X, {'y_a': y}

    def criterion_preparation(self, y_a: np.ndarray, y_b: np.ndarray = None, lam: float = 1.0
                              ) -> typing.Callable:
        return lambda criterion, pred: criterion(pred, y_a)

    @staticmethod
    def get_properties(dataset_properties: typing.Optional[typing.Dict[str, typing.Any]] = None
                       ) -> typing.Dict[str, typing.Union[str, bool]]:
        return {
            'shortname': 'StandardTrainer',
            'name': 'StandardTrainer',
            'handles_tabular': True,
            'handles_image': True,
            'handles_time_series': True,
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: typing.Optional[typing.Dict] = None,
                                        weighted_loss: typing.Tuple[typing.Tuple, bool] = ((True, False), True),
                                        use_swa: typing.Tuple[typing.Tuple, bool] = ((True, False), True),
                                        ) -> ConfigurationSpace:
        cs = super(StandardTrainer, StandardTrainer). \
            get_hyperparameter_search_space(dataset_properties=dataset_properties,
                                            weighted_loss=weighted_loss, use_swa=use_swa)

        return cs
