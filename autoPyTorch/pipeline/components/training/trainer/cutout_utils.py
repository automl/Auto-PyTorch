from typing import Any, Callable, Dict, Optional, Tuple

from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
)

import numpy as np

from autoPyTorch.constants import CLASSIFICATION_TASKS, STRING_TO_TASK_TYPES
from autoPyTorch.pipeline.components.training.trainer.utils import Lookahead


class CutOut:
    def __init__(self, patch_ratio: float,
                 cutout_prob: float,
                 weighted_loss: bool = False,
                 random_state: Optional[np.random.RandomState] = None,
                 use_stochastic_weight_averaging: bool = False,
                 use_snapshot_ensemble: bool = False,
                 se_lastk: int = 3,
                 use_lookahead_optimizer: bool = True,
                 **lookahead_config: Any):
        """
        This class handles the training of a network for a single given epoch.

        Args:
            patch_ratio (float): Defines the size of the cut off
            cutout_prob (float): The probability of occurrence of this regulatization

        """
        self.use_stochastic_weight_averaging = use_stochastic_weight_averaging
        self.weighted_loss = weighted_loss
        self.random_state = random_state
        self.use_snapshot_ensemble = use_snapshot_ensemble
        self.se_lastk = se_lastk
        self.use_lookahead_optimizer = use_lookahead_optimizer
        # Add default values for the lookahead optimizer
        if len(lookahead_config) == 0:
            lookahead_config = {f'{Lookahead.__name__}:la_steps': 6,
                                f'{Lookahead.__name__}:la_alpha': 0.6}
        self.lookahead_config = lookahead_config
        self.patch_ratio = patch_ratio
        self.cutout_prob = cutout_prob

    def criterion_preparation(self, y_a: np.ndarray, y_b: np.ndarray = None, lam: float = 1.0
                              ) -> Callable:
        return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict] = None,
        weighted_loss: Tuple[Tuple, bool] = ((True, False), True),
        use_stochastic_weight_averaging: Tuple[Tuple, bool] = ((True, False), True),
        use_snapshot_ensemble: Tuple[Tuple, bool] = ((True, False), True),
        se_lastk: Tuple[Tuple, int] = ((3,), 3),
        use_lookahead_optimizer: Tuple[Tuple, bool] = ((True, False), True),
        la_steps: Tuple[Tuple, int, bool] = ((5, 10), 6, False),
        la_alpha: Tuple[Tuple, float, bool] = ((0.5, 0.8), 0.6, False),
        patch_ratio: Tuple[Tuple[float, float], float] = ((0.0, 1.0), 0.2),
        cutout_prob: Tuple[Tuple[float, float], float] = ((0.0, 1.0), 0.2),
    ) -> ConfigurationSpace:
        weighted_loss = CategoricalHyperparameter("weighted_loss", choices=weighted_loss[0],
                                                  default_value=weighted_loss[1])
        use_swa = CategoricalHyperparameter("use_stochastic_weight_averaging",
                                            choices=use_stochastic_weight_averaging[0],
                                            default_value=use_stochastic_weight_averaging[1])
        use_se = CategoricalHyperparameter("use_snapshot_ensemble",
                                           choices=use_snapshot_ensemble[0],
                                           default_value=use_snapshot_ensemble[1])

        # Note, this is not easy to be considered as a hyperparameter.
        # When used with cyclic learning rates, it depends on the number
        # of restarts.
        se_lastk = Constant('se_lastk', se_lastk[1])

        use_lookahead_optimizer = CategoricalHyperparameter("use_lookahead_optimizer",
                                                            choices=use_lookahead_optimizer[0],
                                                            default_value=use_lookahead_optimizer[1])

        config_space = Lookahead.get_hyperparameter_search_space(la_steps=la_steps,
                                                                 la_alpha=la_alpha)
        parent_hyperparameter = {'parent': use_lookahead_optimizer, 'value': True}

        cs = ConfigurationSpace()
        cs.add_hyperparameters([use_swa, use_se, se_lastk, use_lookahead_optimizer])
        cs.add_configuration_space(
            Lookahead.__name__,
            config_space,
            parent_hyperparameter=parent_hyperparameter
        )
        cond = EqualsCondition(se_lastk, use_se, True)
        cs.add_condition(cond)

        if dataset_properties is not None:
            if STRING_TO_TASK_TYPES[dataset_properties['task_type']] in CLASSIFICATION_TASKS:
                cs.add_hyperparameters([weighted_loss])

        patch_ratio = UniformFloatHyperparameter(
            "patch_ratio", patch_ratio[0][0], patch_ratio[0][1], default_value=patch_ratio[1])
        cutout_prob = UniformFloatHyperparameter(
            "cutout_prob", cutout_prob[0][0], cutout_prob[0][1], default_value=cutout_prob[1])

        cs.add_hyperparameters([patch_ratio, cutout_prob])

        return cs
