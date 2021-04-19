from typing import Any, Callable, Dict, Optional

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
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


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
            weighted_loss: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="weighted_loss",
                value_range=(True, False),
                default_value=True),
            la_steps: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="la_steps",
                value_range=(5, 10),
                default_value=6,
                log=False),
            la_alpha: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="la_alpha",
                value_range=(0.5, 0.8),
                default_value=0.6,
                log=False),
            use_lookahead_optimizer: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="use_lookahead_optimizer",
                value_range=(True, False),
                default_value=True),
            use_stochastic_weight_averaging: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="use_stochastic_weight_averaging",
                value_range=(True, False),
                default_value=True),
            use_snapshot_ensemble: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="use_snapshot_ensemble",
                value_range=(True, False),
                default_value=True),
            se_lastk: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="se_lastk",
                value_range=(3,),
                default_value=3),
            patch_ratio: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="patch_ratio",
                value_range=(0, 1),
                default_value=0.2),
            cutout_prob: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="cutout_prob",
                value_range=(0, 1),
                default_value=0.2),
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()

        add_hyperparameter(cs, patch_ratio, UniformFloatHyperparameter)
        add_hyperparameter(cs, cutout_prob, UniformFloatHyperparameter)
        add_hyperparameter(cs, use_stochastic_weight_averaging, CategoricalHyperparameter)
        use_snapshot_ensemble = get_hyperparameter(use_snapshot_ensemble, CategoricalHyperparameter)
        se_lastk = get_hyperparameter(se_lastk, Constant)
        cs.add_hyperparameters([use_snapshot_ensemble, se_lastk])
        cond = EqualsCondition(se_lastk, use_snapshot_ensemble, True)
        cs.add_condition(cond)

        use_lookahead_optimizer = get_hyperparameter(use_lookahead_optimizer, CategoricalHyperparameter)
        cs.add_hyperparameter(use_lookahead_optimizer)
        la_config_space = Lookahead.get_hyperparameter_search_space(la_steps=la_steps,
                                                                    la_alpha=la_alpha)
        parent_hyperparameter = {'parent': use_lookahead_optimizer, 'value': True}
        cs.add_configuration_space(
            Lookahead.__name__,
            la_config_space,
            parent_hyperparameter=parent_hyperparameter
        )

        if dataset_properties is not None:
            if STRING_TO_TASK_TYPES[dataset_properties['task_type']] in CLASSIFICATION_TASKS:
                add_hyperparameter(cs, weighted_loss, CategoricalHyperparameter)

        return cs
